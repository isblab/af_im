import numpy as np
import pandas as pd
import os
import glob
import pickle as pkl
from collections import OrderedDict
import ml_collections as mlc
import time
import copy
import random

from typing import Dict, Tuple, Optional

import torch
from torch import nn

from openfold.data import feature_pipeline
from openfold.model.structure_module import StructureModule
from openfold.model.heads import AuxiliaryHeads
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.feats import atom14_to_atom37
from openfold.np import protein

from model import get_model
from loss import LossFunction
from metrics import Metrics
from optimizer import Optimizer
from pdb_utils import SaveModels



def parse_nested_dict( dict_: Dict, action: str, 
						device: Optional[str] = "cuda" ):
	for k in dict_:
		if isinstance( dict_[k], Dict ):
			dict_[k] = parse_nested_dict( dict_[k], action, device )
		else:
			if isinstance( dict_[k], torch.Tensor ):
				if action == "add_dim":
					dict_[k] = dict_[k].unsqueeze( 0 )
				elif action == "add_to_device":
					dict_[k] = dict_[k].to( device )
				elif action == "detach":
					dict_[k] = dict_[k].detach().cpu()
	return dict_


class FitToData():
	def __init__( self, ofold_config: mlc.ConfigDict, 
					topology: mlc.ConfigDict, 
					mode: str, 
					system_features: Dict, 
					ofold_output_dir: str,
					output_dir: str,
					prec: int,
					seed_worker,
					device: str ):
		self.ofold_config = ofold_config
		self.topology = topology
		self.is_multimer = self.ofold_config.globals.is_multimer,
		self.mode = mode
		self.prec = prec
		self.device = device
		self.system_features = system_features
		self.ofold_output_dir = ofold_output_dir
		self.output_dir = output_dir
		self.ensemble_file = os.path.join( self.output_dir, f"2ayo_output_models" )

		seed_worker()

		# Add a singleton batch dim.
		self.add_batch_dim()

		self.loss_fn = LossFunction( self.topology["loss"], self.device )
		self.loss_dict = {}
		self.metrics_fn = Metrics( self.topology["metrics"], self.system_features["restraint_features"] )
		self.scalar_metric_dict = {}
		self.other_metric_dict = {}



	def forward( self ):
		"""
		"""
		self.load_feature_dict()
		self.fit()


	def ensemble_exists( self ):
		"""
		Check if the ensemble file already exists.
			If exists --> Do not run the finetuning process
		"""
		return os.path.exists( f"{self.ensemble_file}.pdb" )



	def load_feature_dict( self ) -> None:
		"""
		Load the feature_dict saved as a .pkl file in the system's director.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )
		feature_dict_path = glob.glob( f"{self.ofold_output_dir}/predictions/*feature_dict.pkl" )
		if len( feature_dict_path ) == 0:
			raise Exception( f"Incorrect path -- {feature_dict_path}..." )

		with open( feature_dict_path[0], "rb" ) as f:
			self.feature_dict = pkl.load( f )



	def get_system_embeddings( self ) -> Dict[str, torch.Tensor]:
		"""
		Obtain the MSA, Pair, and Single representation from OpenFold.
		These are stored in a .pkl file in the OpenFold output directory.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		msa_rep --> MSA representation for the system [n, r, 256].
		pair_rep --> Pair representation for the system [r, r, 128].
		single_rep --> Single representation for the system [r, 384].
		( n --> no. of seq in MSA; r --> no. of residues in system. )
		"""
		print( "\nLoding evoformer output MSA, Pair, Single representations..." )
		pkl_path = glob.glob( f"{self.ofold_output_dir}/predictions/*output_dict.pkl" )
		if len( pkl_path ) == 0:
			raise Exception( f"Incorrect path -- {pkl_path}..." )

		pkl_path = pkl_path[0]

		with open( pkl_path, "rb" ) as f:
			ofold_output = pkl.load( f )

		msa_rep = ofold_output["msa"]
		pair_rep = ofold_output["pair"]
		single_rep = ofold_output["single"]

		self.evo_output = {
		"msa": torch.from_numpy( msa_rep ),
		"pair": torch.from_numpy( pair_rep ),
		"single": torch.from_numpy( single_rep )
		}
		print( f"MSA rep: {msa_rep.shape} \t Pair rep: {pair_rep.shape} \t Single rep: {single_rep.shape}" )

		# return evo_output


	def add_batch_dim( self ) -> None:
		"""
		Adds a singleton batch dimension to all tensors.

		****
		This is needed because downstream functions (multi_chain_permutation_align)
			assume that the tensors always have a batch dimension (which will be there while training).
		The reasoning to add a batch dim is speculative.
			I assume this because, in openfold.utils.multi_chain_permutation.multi_chain_permutation_align(),
				Line 421: anchor_true_pos = torch.index_select(true_ca_poses[anchor_gt_idx], 1, anchor_gt_residue)
			tries selecting the dim=1 (hardcoded) in the tensor true_ca_poses (shape = [Nres]) which does not exist.
			Going through the code the only reason for this to happen can be the existence of a batch dim, 
				that would exist while training in mini-batches but does not exist in our case.
		****
		"""
		print( "\nAdding singleton batch dim to all tensors..." )

		# def parse_nested_dict( dict_: Dict, action: str ):
		# 	for k in dict_:
		# 		if isinstance( dict_[k], Dict ):
		# 			dict_[k] = parse_nested_dict( dict_[k] )
		# 		else:
		# 			if isinstance( dict_[k], torch.Tensor ):
		# 				if action == "add_dim":
		# 					dict_[k] = dict_[k].unsqueeze( 0 )
		# 				elif action == "add_to_device":
		# 					dict_[k] = dict_[k].to( self.device )
		# 				elif action == "detach":
		# 					dict_[k] = dict_[k].detach()
		# 	return dict_
		
		with torch.no_grad():
			self.system_features = parse_nested_dict( self.system_features, "add_dim" )
			# dtype = torch.int64 is needed for torch.nn.functional.one_hot() in violation_loss calculation.
			self.system_features["residue_index"] = self.system_features["residue_index"].to( torch.int64 )


	def add_to_device( self ):
		"""
		Add all tensors to device.
		"""
		self.evo_output = parse_nested_dict( self.evo_output, "add_to_device", self.device )
		self.system_features = parse_nested_dict( self.system_features, "add_to_device", self.device )


	def remove_from_device( self, outputs: Dict[str, torch.Tensor] ):
		"""
		Add all tensors to device.
		"""
		self.evo_output = parse_nested_dict( self.evo_output, "detach" )
		self.system_features = parse_nested_dict( self.system_features, "detach" )

		outputs = parse_nested_dict( outputs, "detach" )

		return outputs


	def fit( self ) -> None:
		"""
		Fine-tune weights for the structure module for fit to data.
		Get the Evoformer output: MSA, Single, Pair representations.
			MSA representation not required though.
		Add a singleton batch dim.
		Initialize:
			Model
			A SaveModel object to add and save predicted models to a CIF file.
			Optimizer
		Run the for max_epochs.
			Get predicted output.
			Calculate loss and update parameters.
			Save model to a PDB file.
		"""
		t = time.time()
		self.get_system_embeddings()

		# Craete a SaveModel object.
		save_model_obj = SaveModels( title = "2ayo", 
									output_format = "pdb",
									output_path = self.ensemble_file )
		# Initialize the System object.
		save_model_obj.initialize_system()
		
		# Get model.
		# 	mode and is_multimer can be removed as we plan to stick to multimers only.
		model = get_model( self.topology.model,
							self.system_features, 
							self.ofold_config,
							self.mode, self.is_multimer, 
							self.device )

		# Initialize the specified optimizer.
		optimizer = Optimizer( self.topology.optimizer ).forward( model.params() )

		for epoch in range( self.topology.train.max_epochs ):
			t_start = time.time()
			print( f"\nEpoch: {epoch} --------------------------" )

			self.add_to_device()

			batch = copy.deepcopy( self.system_features )   # Just to keep in sync with OpenFold implementation.
			# Separate out the ground truth features - as in OpenFold training_step.
			gt_features = batch.pop( "gt_features", None )
			# Separate out the restraint features.
			restraint_features = batch.pop( "restraint_features", None )

			# evo_output["single"] = d( evo_output["single"] )
			outputs, batch = model.predict( self.evo_output, gt_features, batch )
			# outputs, batch = self.predict( batch, evo_output, gt_features )

			# This was used in training AF2 to permutes chains in ground truth before calculating the loss
			# 	because the mapping between the predicted and ground-truth will become arbitrary.
			# 	The model cannot be assumed to predict chains in the same order as the ground truth.
			if self.is_multimer and self.topology.train.allow_mcpa:
				# mcpa --> multi chain permutation align
				print( "--> Performing multi-chain permutation alignment..." )
				batch = multi_chain_permutation_align( out = outputs,
														features = batch,
														ground_truth = gt_features )

			self.add_model( save_model_obj, outputs, epoch )
			self.step( outputs, batch, restraint_features, optimizer, epoch )
			t_end = time.time()
			t_ = time.time()
			print( f"Time taken: {( t_end - t_start )}  seconds" )

		self.save_model( save_model_obj )

		t_ = time.time()
		print( f"\n --> Time taken for fitting: {( t_ - t )}  seconds" )



	def compute_loss( self, out: Dict[str, torch.Tensor], 
						batch: Dict[str, torch.Tensor], 
						restraint_features: Dict 
				) -> Tuple[torch.Tensor, Dict[str, float]]:
		"""
		Compute the loss and return the cumulative loss and a dict containing all loss terms per epoch.

		"""
		cum_loss, losses = self.loss_fn.forward( out, batch, restraint_features )
		# print( losses )

		return cum_loss, losses



	def update_loss_dict( self, losses: Dict ):
		"""
		Keep a tab on the loss per epoch for all individual loss 
			terms and the cumulative loss.
		"""
		if self.loss_dict == {}:
			self.loss_dict = {k: [] for k in losses.keys()}
			# self.loss_dict = {k:[round( v.item(), self.prec )] for k, v in losses.items()}
		
		str_ = ""
		for k, v in losses.items():
			v = round( v.item(), self.prec )
			str_ += f"{k}: {v} \t"
			self.loss_dict[k].append( v )
		print( f"Losses: {str_}" )



	def compute_metrics( self, out: Dict[str, torch.Tensor], 
						batch: Dict[str, torch.Tensor], 
						restraint_features: Dict,
						last_epoch: bool
				) -> Dict[str, Dict]:
		"""
		Compute all the required metrics.
		"""
		scalar_metric_dict = self.metrics_fn.forward( out, last_epoch )

		return scalar_metric_dict


	def update_metric_dict( self, scalar_metric_dict: Dict[str, float],
								other_metric_dict: Dict ) -> None:
		"""
		Keep a tab on the metric values per epoch for all individual merics.
		"""
		if self.scalar_metric_dict == {}:
			self.scalar_metric_dict = {k: [] for k in scalar_metric_dict.keys()}
			# self.metric_dict = {k:[round( v.item(), self.prec )] for k, v in metric_dict.items()}

		str_ = ""		
		for k, v in scalar_metric_dict.items():
			v = round( v.item(), self.prec )
			str_ += f"{k}: {v} \t"
			
			self.scalar_metric_dict[k].append( v )
		print( f"Metrics: {str_}" )

		for k, v in other_metric_dict.items():
			self.other_metric_dict[k] = v



	def step( self, outputs: Dict[str, torch.Tensor], 
				batch: Dict[str, torch.Tensor], 
				restraint_features: Dict, 
				optimizer,
				epoch: int ) -> None:
		"""
		Compute the loss for the finetuned output (need to add that yet).
		Keep track of per-epoch final loss and for each individual loss terms.
		Update the parameters.
		"""
		cum_loss, losses = self.compute_loss( outputs, batch, restraint_features )
		self.update_loss_dict( losses )

		if self.topology.train.allow_grad_update:
			optimizer.zero_grad()
			cum_loss.backward()
			optimizer.step()

		# Detach and unload all tensors from device.
		outputs = self.remove_from_device( outputs )
		
		if epoch == self.topology.train.max_epochs-1:
			last_epoch = True
		else:
			last_epoch = False

		scalar_metric_dict, other_metric_dict = self.compute_metrics( outputs, 
																		batch, 
																		restraint_features, 
																		last_epoch )
		self.update_metric_dict( scalar_metric_dict, other_metric_dict )



	def add_model( self, save_model_obj: SaveModels, 
					outputs: Dict[str, torch.Tensor], 
					epoch: int ) -> None:
		"""
		Create a Protein object using the predicted model output.
		For pdb: write the model as a pdb string.
		For cif: add the predicted structure as a model to a modelcif object.
		"""
		unrelaxed_protein = save_model_obj.prep_protein( 
													outputs = outputs, 
													feature_dict = self.feature_dict, 
			                                		feature_processor = self.feature_processor )
		# if epoch == 0:
			# save_model_obj.create_attributes( unrelaxed_protein )

		# save_model_obj.add_to_modelcif( unrelaxed_protein, epoch )
		save_model_obj.add_model( prot = unrelaxed_protein, epoch = epoch )



	def save_model( self, save_model: SaveModels ) -> None:
		"""
		Save to PDB or CIF file.
		"""
		save_model.save()


