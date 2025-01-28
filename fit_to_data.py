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

from typing import Dict

import torch
from torch import nn

from openfold.data import feature_pipeline
from openfold.model.structure_module import StructureModule
from openfold.model.heads import AuxiliaryHeads
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.feats import atom14_to_atom37
from openfold.np import protein

from model import Model1, Model2
from loss import LossFunction
from optimizer import Optimizer
from pdb_utils import SaveModels



class FitToData():
	def __init__( self, ofold_config: mlc.ConfigDict, 
					sys_config: mlc.ConfigDict, 
					mode: str, 
					system_features: Dict, 
					ofold_output_dir: str,
					output_dir: str,
					seed_worker ):
		self.ofold_config = ofold_config
		self.sys_config = sys_config
		self.is_multimer = self.ofold_config.globals.is_multimer,
		self.mode = mode
		self.system_features = system_features
		self.ofold_output_dir = ofold_output_dir
		self.output_dir = output_dir
		self.models_file = os.path.join( self.output_dir, f"2ayo_output_models" )

		self.loss_fn = LossFunction( self.sys_config["loss"] )
		self.loss_dict = {}

		seed_worker()


	def forward( self ):
		"""
		"""
		self.load_feature_dict()
		# Now loading the models.
		# self.load_models()
		self.fit()


	def load_feature_dict( self ):
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



	def get_system_embeddings( self ):
		"""
		Obtain the MSA, Pair, and Single representation from OpenFold.
		These are stored in a .pkl file in the OpenFold output directory.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		msa_rep --> MSA representation for the system [N, L, 256].
		pair_rep --> Pair representation for the system [L, L, 128].
		single_rep --> Single representation for the system [L, 384].
		( N --> no. of seq in MSA; L --> no. of residues in system. )
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

		evo_output = {
		"msa": torch.from_numpy( msa_rep ),
		"pair": torch.from_numpy( pair_rep ),
		"single": torch.from_numpy( single_rep )
		}
		print( f"MSA rep: {msa_rep.shape} \t Pair rep: {pair_rep.shape} \t Single rep: {single_rep.shape}" )

		return evo_output


	def add_batch_dim( self ):
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

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		print( "\nAdding singleton batch dim to all tensors..." )

		def parse_nested_dict( dict_: Dict ):
			for k in dict_:
				if isinstance( dict_[k], Dict ):
					dict_[k] = parse_nested_dict( dict_[k] )
				else:
					if isinstance( dict_[k], torch.Tensor ):
						dict_[k] = dict_[k].unsqueeze( 0 )
			return dict_
		
		with torch.no_grad():
			self.system_features = parse_nested_dict( self.system_features )
			# dtype = torch.int64 is needed for torch.nn.functional.one_hot() in violation_loss calculation.
			self.system_features["residue_index"] = self.system_features["residue_index"].to( torch.int64 )


	def fit( self ):
		"""
		Fine-tune weights for the structure module for fit to data.
		Get the Evoformer output: MSA, Single, Pair representations.
			MSA representation not required though.
		Add a singleton batch dim.
		Temporary implemntation: Create XL restraint feature.
		Initialize:
			A SaveModel object to add and save predicted models to a CIF file.
			Optimizer
		Run the finetuning for max_epochs.
			Get predicted output.
			Add predicted model to model group.
			Calculate loss and update parameters.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		t = time.time()
		evo_output = self.get_system_embeddings()

		# Add a singleton batch dim.
		self.add_batch_dim()
		
		batch = copy.deepcopy( self.system_features )   # Just to keep in sync with OpenFold implementation.
		# Separate out the ground truth features.
		gt_features = batch.pop( "gt_features", None )
		gt_features_keys = gt_features.keys()
		# Separate out the restraint features.
		restraint_features = batch.pop( "restraint_features", None )

		# Craete a SaveModel object.
		save_model_obj = SaveModels( title = "2ayo", 
									output_format = "pdb",
									output_path = self.models_file )
		# Initialize the System object.
		save_model_obj.initialize_system()
		
		# Get model.
		# model = Model1( self.system_features, self.ofold_config, self.mode, self.is_multimer )
		model = Model2( self.system_features, self.ofold_config, self.mode, self.is_multimer )

		# Initialize the specified optimizer.
		optimizer = Optimizer( self.sys_config.optimizer ).forward( model.params() )

		# d = nn.Dropout1d( p = 0.05 )

		for epoch in range( self.sys_config.train.max_epochs ):
			print( f"Epoch: {epoch}" )

			# evo_output["single"] = d( evo_output["single"] )
			outputs, batch = model.predict( evo_output, gt_features, batch )
			# outputs, batch = self.predict( batch, evo_output, gt_features )

			# This was used in training AF2 to permutes chains in ground truth before calculating the loss
			# 	because the mapping between the predicted and ground-truth will become arbitrary.
			# 	The model cannot be assumed to predict chains in the same order as the ground truth.
			if self.is_multimer:
				print( "\nPerforming multi-chain permutation alignment..." )
				batch = multi_chain_permutation_align( out = outputs,
														features = batch,
														ground_truth = gt_features )

			self.add_model( save_model_obj, outputs, epoch )
			self.step( outputs, batch, restraint_features, optimizer )

		self.save_model( save_model_obj )

		t_ = time.time()
		print( ( t_ - t ), " seconds" )



	def compute_loss( self, out: Dict, batch: Dict, restraint_features: Dict ):
		"""
		Calculates the cumulative loss which includes:
			FAPE - backbone and sidechain
			Supervised chi
			Violation
			Chain centre of mass
			Restraints

		"""
		cum_loss, losses = self.loss_fn.forward( out, batch, restraint_features )
		# print( losses )

		return cum_loss, losses
		# return cum_loss, np.array( [v.reshape( -1 ) for k, v in losses.items()] )



	def update_loss_dict( self, losses: Dict ):
		"""
		Keep a tab on the loss per epoch for all individual loss 
			terms and the cumulative loss.
		"""
		if self.loss_dict == {}:
			self.loss_dict = {k:[v.item()] for k, v in losses.items()}
		else:
			for k, v in losses.items():
				self.loss_dict[k].append( v.item() )



	def step( self, outputs: Dict, batch: Dict, restraint_features: Dict, optimizer ):
		"""
		Compute the loss for the finetuned output (need to add that yet).
		Keep track of per-epoch final loss and for each individual loss terms.
		Update the parameters.
		"""
		cum_loss, losses = self.compute_loss( outputs, batch, restraint_features )
		self.update_loss_dict( losses )

		if self.sys_config.train.allow_grad_update:
			cum_loss.backward()
			optimizer.step()



	def add_model( self, save_model_obj: SaveModels, outputs: Dict, epoch: int ):
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



	def save_model( self, save_model: SaveModels ):
		"""
		Save to PDB or CIF file.
		"""
		save_model.save()


