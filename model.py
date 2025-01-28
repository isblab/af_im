import numpy as np
import os
import ml_collections as mlc
from collections import OrderedDict
import random

from typing import List, Dict

import torch
from torch import nn

from openfold.model.structure_module import StructureModule
from openfold.model.heads import PerResidueLDDTCaPredictor, DistogramHead
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.feats import atom14_to_atom37
from openfold.np import protein
from openfold.utils.loss import compute_plddt



class LoadState():
	def __init__( self, ofold_config: mlc.ConfigDict, mode: str, is_multimer: bool ):
		self.ofold_config = ofold_config
		self.mode = mode
		self.is_multimer = is_multimer

		self.weights_dict = {}



	def get_pretrained_weights( self, layers: List ):
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

		for layer in layers:
			if layer == "structure_module":
				# Obtain weights for the structure module only.
				self.weights_dict[layer] = OrderedDict( 
							( ".".join( key.split( "." )[1:] ), pretrained_weights[key] ) 
							for key in pretrained_weights.keys() if "structure_module" in key 
							)
			if layer == "lddt":
				# Obtain weights for the plddt head from auxillary heads module.
				self.weights_dict[layer] = OrderedDict( 
							( ".".join( key.split( "." )[2:] ), pretrained_weights[key] ) 
							for key in pretrained_weights.keys() if "aux_heads.plddt" in key 
							)
			if layer == "distogram":
				# Obtain weights for the distogram head from auxillary heads module.
				self.weights_dict[layer] = OrderedDict( 
							( ".".join( key.split( "." )[2:] ), pretrained_weights[key] ) 
							for key in pretrained_weights.keys() if "aux_heads.distogram" in key 
							)

	def load_pretrained_models( self, layers: List ):
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

		print( "\nLoading the structure module..." )
		self.get_pretrained_weights( layers )
		
		for layer in layers:
			if layer == "structure_module":
				# Instantiate the structure module.
				self.structure_module = StructureModule(
					is_multimer = self.is_multimer,
					**self.ofold_config["model"]["structure_module"]
				)

				# Initialize the models with the pretrained weights.
				self.structure_module.load_state_dict( self.weights_dict["structure_module"] )

			if layer == "lddt":
				self.plddt = PerResidueLDDTCaPredictor( **self.ofold_config["model"]["heads"]["lddt"] )
				self.plddt.load_state_dict( self.weights_dict["lddt"] )
			
			if layer == "distogram":
				self.distogram_head = DistogramHead( **self.ofold_config["model"]["heads"]["distogram"] )
				self.distogram_head.load_state_dict( self.weights_dict["distogram"] )



class Model1( LoadState ):
	def __init__( self, system_features: mlc.ConfigDict, 
						ofold_config: mlc.ConfigDict, 
						mode: str, is_multimer: bool ):
		super().__init__( ofold_config, mode, is_multimer )
		
		self.ofold_config = ofold_config
		self.system_features = system_features

		layers = ["structure_module", "lddt"]
		self.load_pretrained_models( layers )

		# Not fine-tuning the structure module here.
		self.structure_module.train()

		# Not fine-tuning lddt head.
		self.plddt.eval()



	def predict( self, evo_output: Dict, gt_features: Dict, batch: Dict ):
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
			# 	Even though not using the full AuxillaryHeads module, but still having this step.
			outputs.update( evo_output )
			# outputs.update( self.aux_heads( outputs ) )
			lddt_logits = self.plddt( outputs["sm"]["single"] )

			# Required for relaxation later on
			outputs["plddt"] = compute_plddt( lddt_logits )

		# # This was used in training AF2 to permutes chains in ground truth before calculating the loss
		# # 	because the mapping between the predicted and ground-truth will become arbitrary.
		# # 	The model cannot be assumed to predict chains in the same order as the ground truth.
		# if self.is_multimer:
		# 	print( "\nPerforming multi-chain permutation alignment..." )
		# 	batch = multi_chain_permutation_align( out = outputs,
		# 											features = batch,
		# 											ground_truth = gt_features )

		return outputs, batch


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.structure_module]



class PairBias( nn.Module ):
	def __init__( self ):
		super().__init__()
		# Feature dim for pair rep (c_z) is 128.
		self.linear = nn.Linear( in_features = 64, out_features = 128, bias = True )
		self.activation = nn.ReLU()
		self.lnorm = nn.LayerNorm( 128 )


	def forward( self, x_in ):
		o = self.linear( x_in )
		o = self.activation( o )
		o = self.lnorm( o )

		return o



class Model2( LoadState ):
	def __init__( self, system_features: mlc.ConfigDict, 
						ofold_config: mlc.ConfigDict, 
						mode: str, is_multimer: bool ):
		super().__init__( ofold_config, mode, is_multimer )
		
		self.ofold_config = ofold_config
		self.system_features = system_features

		layers = ["structure_module", "lddt", "distogram"]
		self.load_pretrained_models( layers )

		# For embedding the input distogram.
		self.pair_bias = PairBias()

		# Not fine-tuning the structure module here.
		self.structure_module.train()

		# Not fine-tuning lddt head.
		self.plddt.eval()
		# Fine-tuning distogram head.
		self.distogram_head.train()



	def predict( self, evo_output: Dict, gt_features: Dict, batch: Dict ):
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
		with torch.no_grad():
			gt_distogram = self.system_features["restraint_features"]["xl_restraint"]["gt_distogram"]
			xl_res_mask = self.system_features["restraint_features"]["xl_restraint"]["xl_res_mask"]
			pair_rep = evo_output.pop( "pair", None )

		# pair = self.pair_bias( gt_distogram )
		evo_output["pair"] = pair + xl_res_mask

		with torch.no_grad():
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

			distogram_logits = self.distogram_head( evo_output["pair"] )
			outputs["distogram_logits"] = distogram_logits

			# The AuxillaryHeads module requires MSA, pair, Single representations in the output dict.
			outputs.update( evo_output )
			lddt_logits = self.plddt( outputs["sm"]["single"] )
			# Required for relaxation later on
			outputs["plddt"] = compute_plddt( lddt_logits )

		return outputs, batch


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.pair_bias, self.structure_module]


