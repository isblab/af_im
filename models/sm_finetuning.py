"""
Contains wrapper around structure module.
Given the Evoformer MSA and Pair representation, predict the structure.
"""
from typing import List, Tuple, Dict, Any
import ml_collections as mlc

import torch

from models.base_model import Model
from models.loader import LoadState

from openfold.utils.feats import atom14_to_atom37
from openfold.utils.loss import compute_plddt


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
			raise ValueError( "Incorrect mode: " +
							f"{self.model_config.mode.sm} for structure module..." )


		if self.model_config.mode.plddt == "train":
			print( "Using plddt head in train mode" )
			self.plddt.train()
		elif self.model_config.mode.plddt == "eval":
			print( "Using plddt head in eval mode" )
			self.plddt.eval()
		else:
			raise ValueError( "Incorrect mode: " +
							f"{self.model_config.mode.lddt} for structure module..." )


	def predict( self,
			 evo_output: Dict[str, torch.Tensor],
			 gt_features: Dict[str, Any],
			 batch: Dict[str, Any]
				) -> Tuple[Dict[str, Any], Dict[str, Any]]:
		"""
		Run the structure module and auxillary heads module.
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


	def params( self )  -> List:
		"""
		Return a list of models for the optimizer.
		"""
		return [self.structure_module]
