import numpy as np
import os
import ml_collections as mlc
from collections import OrderedDict
import random
from abc import ABC, abstractmethod

from typing import List, Dict, Tuple

import torch
from torch import nn

from openfold.model.structure_module import StructureModule
from openfold.model.heads import PerResidueLDDTCaPredictor, DistogramHead
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.feats import atom14_to_atom37
from openfold.np import protein
from openfold.utils.loss import compute_plddt



def get_model( model_config: mlc.ConfigDict, system_features: mlc.ConfigDict, 
						ofold_config: mlc.ConfigDict, 
						mode: str, is_multimer: bool, device: str ):
	"""
	Return the required model.
	"""
	if model_config.name == "structure_module_finetuning":
		return StructureModuleFineTuning( system_features, ofold_config, model_config, 
											mode, is_multimer, device )
	elif model_config.name == "pair_bias":
		return PairBias( system_features, ofold_config, model_config, 
											mode, is_multimer, device )
	else:
		raise Exception( "Incorrect model type specified..." ) 


class LoadState():
	def __init__( self, ofold_config: mlc.ConfigDict, mode: str, is_multimer: bool, device: str ):
		self.ofold_config = ofold_config
		self.mode = mode
		self.is_multimer = is_multimer
		self.device = device

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
			for key in pretrained_weights.keys():
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
				).to( self.device )

				# Initialize the models with the pretrained weights.
				self.structure_module.load_state_dict( self.weights_dict["structure_module"] )
				self.structure_module

			if layer == "lddt":
				self.plddt = PerResidueLDDTCaPredictor( **self.ofold_config["model"]["heads"]["lddt"] ).to( self.device )
				self.plddt.load_state_dict( self.weights_dict["lddt"] )
			
			if layer == "distogram":
				self.distogram_head = DistogramHead( **self.ofold_config["model"]["heads"]["distogram"] ).to( self.device )
				self.distogram_head.load_state_dict( self.weights_dict["distogram"] )



class Model( ABC ):
	def __init__( self ):
		super().__init__()
	
	@abstractmethod
	def predict( self, evo_output: Dict[str, torch.Tensor], 
					gt_features: Dict[str, torch.Tensor], 
					batch: Dict[str, torch.Tensor], 
					device: str
			) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
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
		pass


	@abstractmethod
	def params( self 
		) -> List[nn.Module]:

		"""
		Return a list of models for the optimizer.
		"""
		pass



class StructureModuleFineTuning( LoadState, Model ):
	def __init__( self, system_features: mlc.ConfigDict, 
						ofold_config: mlc.ConfigDict, 
						model_config: mlc.ConfigDict,
						mode: str, is_multimer: bool, device: str ):
		LoadState.__init__( self, ofold_config, mode, is_multimer, device )
		Model.__init__( self )
		
		self.ofold_config = ofold_config
		self.model_config = model_config
		self.system_features = system_features

		layers = ["structure_module", "lddt"]
		self.load_pretrained_models( layers )

		if self.model_config.mode.sm == "train":
			print( "Using structure module in train mode" )
			self.structure_module.train()
		elif self.model_config.mode.sm == "eval":
			print( "Using structure module in eval mode" )
			self.structure_module.eval()
		else:
			raise Exception( f"Incorrect mode: {self.model_config.mode.sm} for structure module..." )


		if self.model_config.mode.plddt == "train":
			print( "Using plddt head in train mode" )
			self.plddt.train()
		elif self.model_config.mode.plddt == "eval":
			print( "Using plddt head in eval mode" )
			self.plddt.eval()
		else:
			raise Exception( f"Incorrect mode: {self.model_config.mode.sm} for plddt head..." )



	def predict( self, evo_output, gt_features, batch ):
		"""
		Run the structure module and auxillary heads module.

		Input:
		----------
		evo_output --> dict containing MSA, Pair, Single representtaions.
		gt_features --> dict contaiining the ground truth features.

		Returns:
		----------
		outputs --> dict containing the output from structure module and plddt head.
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

		return outputs, batch


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.structure_module]



class PairBias( LoadState, Model ):
	def __init__( self, system_features: mlc.ConfigDict, 
						ofold_config: mlc.ConfigDict, 
						model_config: mlc.ConfigDict,
						mode: str, is_multimer: bool, device: str ):
		LoadState.__init__( self, ofold_config, mode, is_multimer, device )
		Model.__init__( self )
		
		self.ofold_config = ofold_config
		self.model_config = model_config
		self.system_features = system_features

		layers = ["structure_module", "lddt"]
		self.load_pretrained_models( layers )

		if self.model_config.mode.sm == "train":
			print( "Using structure module in train mode" )
			self.structure_module.train()
		elif self.model_config.mode.sm == "eval":
			print( "Using structure module in eval mode" )
			self.structure_module.eval()
		else:
			raise Exception( f"Incorrect mode: {self.model_config.mode.sm} for structure module..." )


		if self.model_config.mode.plddt == "train":
			print( "Using plddt head in train mode" )
			self.plddt.train()
		elif self.model_config.mode.plddt == "eval":
			print( "Using plddt head in eval mode" )
			self.plddt.eval()
		else:
			raise Exception( f"Incorrect mode: {self.model_config.mode.sm} for plddt head..." )



	def predict( self, evo_output, gt_features, batch ):
		"""
		Run the structure module and auxillary heads module.

		Input:
		----------
		evo_output --> dict containing MSA, Pair, Single representtaions.
		gt_features --> dict contaiining the ground truth features.

		Returns:
		----------
		outputs --> dict containing the output from structure module and plddt head.
		"""
		outputs = {}
		with torch.no_grad():
			restraint_features = self.system_features.pop( "restraint_features", None )
			gt_distogram = restraint_features["xl_restraint"]["gt_distogram"]
			xl_res_mask = restraint_features["xl_restraint"]["xl_res_mask"]
			bias_scale_factor = restraint_features["xl_restraint"]["xl_restraint"]
			bias_type = restraint_features["xl_restraint"]["bias_type"]
			
			pair_rep = evo_output.pop( "pair", None )

		if bias_type == "additive":
			evo_output["pair"] = pair_rep + xl_res_mask.squeeze( 0 ).unsqueeze( -1 )
		elif bias_type == "multiplicative":
			evo_output["pair"] = pair_rep * xl_res_mask.squeeze( 0 ).unsqueeze( -1 )
		else:
			raise Exception( f"Incorrect bias type: {bias_type} specified. " +
								"Only 'additive' or 'multiplicative' bias allowed..." )

		with torch.no_grad():
			# Don't need the full Evoformer dict, just the Pair and Single representation.
			outputs["sm"] = self.structure_module( evoformer_output_dict = evo_output, 
													aatype = gt_features["aatype"],
													mask = self.system_features["seq_mask"].to( 
																			dtype = evo_output["single"].dtype )
																			)

			# The  dim=0 in all structure module outputs represents the no. of 
			# 	structure module blocks (default = 8).
			outputs["final_atom_positions"] = atom14_to_atom37(
														outputs["sm"]["positions"][-1],
														gt_features
														)
			outputs["final_atom_mask"] = gt_features["atom37_atom_exists"]
			outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

			# distogram_logits = self.distogram_head( evo_output["pair"] )
			# outputs["distogram_logits"] = distogram_logits

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
		return []



# class PairBias( nn.Module ):
# 	def __init__( self ):
# 		super().__init__()
# 		# Feature dim for pair rep (c_z) is 128.
# 		self.linear = nn.Linear( in_features = 64, out_features = 128, bias = True )
# 		self.activation = nn.ReLU()
# 		self.lnorm = nn.LayerNorm( 128 )


# 	def forward( self, x_in ):
# 		o = self.linear( x_in )
# 		o = self.activation( o )
# 		o = self.lnorm( o )

# 		return o

