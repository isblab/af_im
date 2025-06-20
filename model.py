"""
This script contains model classes.
"""
import os
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from collections import OrderedDict
import ml_collections as mlc

import torch
from torch import nn

from openfold.model.structure_module import StructureModule
from openfold.model.heads import PerResidueLDDTCaPredictor, DistogramHead
# from openfold.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold.utils.feats import atom14_to_atom37
# from openfold.np import protein
from openfold.utils.loss import compute_plddt


def get_model( model_config: mlc.ConfigDict, system_features: mlc.ConfigDict,
						ofold_config: mlc.ConfigDict,
						mode: str, is_multimer: bool, device: str ):
	"""
	Return the required model.
	"""
	if model_config.name == "structure_module_finetuning":
		model = StructureModuleFineTuning( system_features, ofold_config, model_config,
											mode, is_multimer, device )
	elif model_config.name == "fixed_additive_pair_bias":
		model = FixedAdditivePairBias( system_features, ofold_config, model_config,
											mode, is_multimer, device )
	else:
		raise ValueError( "Incorrect model type specified..." )

	return model


class LoadState():
	"""
	Load the parameters for pretrained models.
	"""
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
			pretrained_weights = torch.load( "./monomer_params.pt" )

		elif self.mode == "multi":
			pretrained_weights = torch.load( os.path.abspath( "./multimer_params.pt" ) )
		else:
			raise ValueError( f"Incorrect mode: {self.mode} specified..." )

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

			if layer == "lddt":
				self.plddt = PerResidueLDDTCaPredictor(
													**self.ofold_config["model"]["heads"]["lddt"]
													).to( self.device )
				self.plddt.load_state_dict( self.weights_dict["lddt"] )

			if layer == "distogram":
				self.distogram_head = DistogramHead(
													**self.ofold_config["model"]["heads"]["distogram"]
													).to( self.device )
				self.distogram_head.load_state_dict( self.weights_dict["distogram"] )



class Model( nn.Module, ABC ):
	"""
	Base class for all model classes to specify the necessary methods.
	"""
	def __init__( self ):
		super( Model, self ).__init__()
		pass

	@abstractmethod
	def predict( self, evo_output: Dict[str, torch.Tensor],
					gt_features: Dict[str, torch.Tensor],
					batch: Dict[str, torch.Tensor]
					# device: str
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


	@abstractmethod
	def params( self ) -> List[nn.Module]:

		"""
		Return a list of models for the optimizer.
		"""



class StructureModuleFineTuning( LoadState, Model ):
	"""
	Create a model comprising the OpenFold structure module and plddt head.
	Fine tuning the structure module.
	"""
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
			raise ValueError( f"Incorrect mode: {self.model_config.mode.sm} for structure module..." )


		if self.model_config.mode.plddt == "train":
			print( "Using plddt head in train mode" )
			self.plddt.train()
		elif self.model_config.mode.plddt == "eval":
			print( "Using plddt head in eval mode" )
			self.plddt.eval()
		else:
			raise ValueError( f"Incorrect mode: {self.model_config.mode.sm} for plddt head..." )



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
														mask = self.system_features["seq_mask"].to(
															dtype = evo_output["single"].dtype ) )

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

			# Required for saving the structure later on.
			outputs["plddt"] = compute_plddt( lddt_logits )

		return outputs, batch


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.structure_module]

