import numpy as np
import os
import glob
import pickle as pkl
from collections import OrderedDict
import ml_collections
import time

from typing import Dict

import torch

from openfold.model.structure_module import StructureModule
from openfold.model.heads import AuxiliaryHeads
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.feats import atom14_to_atom37

from loss import LossFunction

class FitToData():
	def __init__( self, ofold_config: ml_collections.ConfigDict, 
					sys_config: ml_collections.ConfigDict, 
					mode: str, 
					system_features: Dict, 
					output_dir: str ):
		self.ofold_config = ofold_config
		self.sys_config = sys_config
		self.is_multimer = self.ofold_config.globals.is_multimer,
		self.mode = mode
		self.system_features = system_features
		self.output_dir = output_dir

		self.loss_fn = LossFunction( self.sys_config["loss"] )

		# .pkl file containing OpenFold outputs including MSA, Pair, Single representation.
		# self.pkl_path = glob.glob( f"{self.output_dir}/predictions/*.pkl" )
		# if len( self.pkl_path ) == 0:
		# 	raise Exception( f"Incorrect path -- {self.pkl_path}..." )

	def forward( self ):
		"""
		"""
		# Now loading the models.
		self.load_models()

		self.fit()


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
		pkl_path = glob.glob( f"{self.output_dir}/predictions/*.pkl" )
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
		print( msa_rep.shape, "  ", pair_rep.shape, "  ", single_rep.shape )

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

		print( outputs.keys() )
		print( outputs["sm"].keys() )
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
		with torch.no_grad():
			for key in self.system_features.keys():
				if isinstance( self.system_features[key], dict ):
					for k in self.system_features[key].keys():
						self.system_features[key][k] = self.system_features[key][k].unsqueeze( 0 )
				else:
					self.system_features[key] = self.system_features[key].unsqueeze( 0 )
			# dtype = torch.int64 is needed for torch.nn.functional.one_hot() in violation_loss calculation.
			self.system_features["residue_index"] = self.system_features["residue_index"].to( torch.int64 )


	def fit( self ):
		"""
		Fine-tune weights for the structure module for fit to data.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		evo_output = self.get_system_embeddings()

		# Add a singleton batch dim.
		self.add_batch_dim()

		# for epoch in self.sys_config["train"]["max_epochs"]:
		t = time.time()
		batch = self.system_features   # Just to keep in sync with OpenFold implementation.
		gt_features = self.system_features.pop( "gt_features", None )
		outputs = self.get_model_output( evo_output, gt_features )

		for k in outputs["sm"].keys():
			print( f"{k}  -->  {outputs['sm'][k].shape}" )
		for k in outputs.keys():
			if k != "sm":
				print( f"{k}  -->  {outputs[k].shape}" )
        
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

		cum_loss, losses = self.compute_loss( outputs, batch )
		# cum_loss.backward()
		# optimizer.step()
		
		t_ = time.time()
		print( ( t_ - t ), " seconds" )



	def compute_loss( self, out: Dict, batch: Dict ):
		"""
		Calculates the cumulative loss which includes:
			FAPE - backbone and sidechain
			Supervised chi
			Violation
			Chain centre of mass
			Restraints

		"""
		cum_loss, losses = self.loss_fn.forward( out, batch )

		return cum_loss, np.array( [v.reshape( -1 ) for k, v in losses.items()] )
