import numpy as np
import pandas as pd
import os
import glob
import pickle as pkl
from collections import OrderedDict
import ml_collections
import time
import copy
import random

from typing import Dict

import torch
from torch import nn

from openfold.data import feature_pipeline
from openfold.model.structure_module import StructureModule
from openfold.model.heads import AuxiliaryHeads
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.feats import atom14_to_atom37
from openfold.np import protein

from loss import LossFunction
from optimizer import Optimizer
from pdb_utils import SaveModels


class FitToData():
	def __init__( self, ofold_config: ml_collections.ConfigDict, 
					sys_config: ml_collections.ConfigDict, 
					mode: str, 
					system_features: Dict, 
					ofold_output_dir: str,
					output_dir: str ):
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

		self.seed = 1


	def seed_worker( self ):
		torch.manual_seed( self.seed )
		# torch.cuda.manual_seed( worker_seed )
		torch.cuda.manual_seed_all( self.seed )
		np.random.seed( self.seed )
		random.seed( self.seed )


	def forward( self ):
		"""
		"""
		self.load_feature_dict()
		# Now loading the models.
		self.load_models()
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


	def load_models( self ):
		"""
		Load the Structure module and the auxillary heads module.
		Intialize with the pretrained weights.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		print( "\nLoading the structure module and auxillary heads..." )
		sm_weights, aux_heads_weights  = self.get_pretrained_weights()
		
		# Instantiate the structure module and aux heads.
		self.structure_module = StructureModule(
			is_multimer = self.is_multimer,
			**self.ofold_config["model"]["structure_module"]
		)
		self.aux_heads = AuxiliaryHeads(
		    self.ofold_config["model"]["heads"],
		)

		# Initialize the models with the pretrained weights.
		self.structure_module.load_state_dict( sm_weights )
		self.aux_heads.load_state_dict( aux_heads_weights )

		# Set the structure module to train mode.
		self.structure_module.train()

		# Set the aux heads to eval mode.
		self.aux_heads.eval()



	def get_pretrained_weights( self ):
		"""
		Get the weights for the pretrained OpenFold monomer and multimer models.
		Extract weights for only Structure modeule, pLDDT head, and TM head.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		if self.mode == "mono":
			pretrained_weights = torch.load( "../../monomer_params.pt" )

		elif self.mode == "multi":
			pretrained_weights = torch.load( os.path.abspath( "../../multimer_params.pt" ) )

		# Obtain weights for the structure module only.
		sm_weights = OrderedDict( 
			( ".".join( key.split( "." )[1:] ), pretrained_weights[key] ) 
			for key in pretrained_weights.keys() if "structure_module" in key 
			)
		# Obtain weights for the auxillary heads module.
		aux_heads_weights = OrderedDict( 
			( ".".join( key.split( "." )[1:] ), pretrained_weights[key] ) 
			for key in pretrained_weights.keys() if "aux_heads" in key 
			)

		return sm_weights, aux_heads_weights



	def get_model_output( self, evo_output: Dict, gt_features: Dict ):
		"""
		Run the structure module and auxillary heads module.

		Input:
		----------
		evo_output --> dict containing MSA, Pair, Single representtaions.
		gt_features --> dict contaiining the ground truth features.

		Returns:
		----------
		outputs --> dict containing the output from structure module and auxillary heads module.
		"""
		outputs = {}
		# Don't need the full Evoformer dict, just the Pair and Single representation.
		outputs["sm"] = self.structure_module.forward( evoformer_output_dict = evo_output, 
														aatype = gt_features["aatype"],
														mask = self.system_features["seq_mask"].to( dtype = evo_output["single"].dtype ) )

		# The  dim=0 in all structure module outputs represents the no. of 
		# 	structure module blocks (default = 8).
		outputs["final_atom_positions"] = atom14_to_atom37(
													outputs["sm"]["positions"][-1],
													gt_features
													)
		outputs["final_atom_mask"] = gt_features["atom37_atom_exists"]
		outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

		with torch.no_grad():
			# The AuxillaryHeads module requires MSA, pair, Single representations in the output dict.
			outputs.update( evo_output )
			outputs.update( self.aux_heads( outputs ) )

		# print( outputs.keys() )
		# print( outputs["sm"].keys() )
		return outputs



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
			# for key in self.system_features.keys():
			# 	if valid_type( self.system_features[key], dict ):
					
			# 		for k in self.system_features[key].keys():
			# 			if valid_type( self.system_features[key][k], dict ):
							
			# 				for m in self.system_features[key][k].keys():
			# 					if valid_type( self.system_features[key][k][m], dict ):
									
			# 						for n in self.system_features[key][k][m].keys():
			# 							if valid_type( self.system_features[key][k][m][n], dict ):
			# 								self.system_features[key][k] = self.system_features[key][k][m][n].unsqueeze( 0 )
			# 								print( key, "  ", k, "  ", self.system_features[key][k][m][n].shape )
										
			# 							else:
			# 								self.system_features[key][k][m][n] = self.system_features[key][k][m][n].unsqueeze( 0 )
			# 								print( key, "  ", k, "  ", self.system_features[key][k][m][n].shape )
								
			# 					else:
			# 						self.system_features[key][k][m] = self.system_features[key][k][m].unsqueeze( 0 )
						
			# 			else:
			# 				self.system_features[key][k] = self.system_features[key][k].unsqueeze( 0 )
				
			# 	else:
			# 		self.system_features[key] = self.system_features[key].unsqueeze( 0 )
			# dtype = torch.int64 is needed for torch.nn.functional.one_hot() in violation_loss calculation.
			self.system_features["residue_index"] = self.system_features["residue_index"].to( torch.int64 )


	# def update_gt_features( self, batch: Dict, gt_features_keys: Dict ):
	# 	"""
	# 	gt_features is a dict nested within batch.
	# 	In OpenFold, for each training step, the gt_features key is split from batch.
	# 	The mul multi-chain_permutation_align() takes batch and gt_features as input separately.
	# 		Post processing all keys in gt_features to batch dict.
	# 	So, gt_features dict needs to be updated from batch dict.
	# 	"""
	# 	gt_features = {}
	# 	for key in gt_features_keys:
	# 		gt_features[key] = batch[key]

	# 	return gt_features



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
		
		batch = self.system_features   # Just to keep in sync with OpenFold implementation.
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
		# Initialize the specified optimizer.
		optimizer = Optimizer( self.sys_config.optimizer ).forward( self.structure_module )

		# d = nn.Dropout1d( p = 0.05 )

		for epoch in range( self.sys_config.train.max_epochs ):
			print( f"Epoch: {epoch}" )

			# evo_output["single"] = d( evo_output["single"] )

			outputs, batch = self.predict( batch, evo_output, gt_features )

			# gt_features = self.update_gt_features( batch, gt_features_keys )

			self.add_model( save_model_obj, outputs, epoch )
			self.step( outputs, batch, restraint_features, optimizer )

		self.save_model( save_model_obj )

		t_ = time.time()
		print( ( t_ - t ), " seconds" )



	def predict( self, batch: Dict, evo_output: Dict, gt_features: Dict ):
		"""
		Obtain model predictions given the input.
		Perform multi-chain permutation align.
		"""
		outputs = self.get_model_output( evo_output, gt_features )

		# for k in outputs["sm"].keys():
		# 	print( f"{k}  -->  {outputs['sm'][k].shape}" )
		# for k in outputs.keys():
		# 	if k != "sm":
		# 		print( f"{k}  -->  {outputs[k].shape}" )
        
        # We are not using recycling so don't need this.
        # Remove the recycling dimension
		# outputs = tensor_tree_map( lambda t: t[..., -1], outputs )
		# self.system_features = tensor_tree_map( lambda t: t[..., -1], self.system_features )

		# This was used in training AF2 to permutes chains in ground truth before calculating the loss
		# 	because the mapping between the predicted and ground-truth will become arbitrary.
		# 	The model cannot be assumed to predict chains in the same order as the ground truth.
		if self.is_multimer:
			print( "\nPerforming multi-chain permutation alignment..." )
			batch = multi_chain_permutation_align( out = outputs,
													features = batch,
													ground_truth = gt_features )

		# Toss out the recycling dimensions --- we don't need them anymore
		# batch = tensor_tree_map(
		# 	lambda x: np.array(x[..., -1].cpu()),
		# 	batch
		# )
		# out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
		return outputs, batch



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


