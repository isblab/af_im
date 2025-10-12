"""
Contains modules for modifying the Evoformer single and pair
	representations at inference time.
"""
from typing import List, Tuple, Dict, Any
import ml_collections as mlc

import torch

from models.base_model import Model
from models.loader import LoadState

from openfold.utils.feats import atom14_to_atom37
from openfold.utils.loss import compute_plddt

from models.adapter import get_adapter


class PairPerturbation( LoadState, Model ):
	"""
	Compose the AF2 structure module and pLDDT head.
	Modify the Pair representation. The StructureModule can be kept frozen.
	This class wraps multiple methods to modify the pair representation (z):
		1. Linear perturbation
		2. Gating
		3. LoRA
		4. FiLM
	Further, we use a binary mask to ignore intra-chain elements in z.
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
		# Feature dim for pair representation.
		self.c_z = 128

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
		if self.model_config.freeze_sm:
			self.no_grad_for_sm()

		if self.model_config.mode.plddt == "train":
			print( "Using plddt head in train mode" )
			self.plddt.train()
		elif self.model_config.mode.plddt == "eval":
			print( "Using plddt head in eval mode" )
			self.plddt.eval()
		else:
			raise ValueError( "Incorrect mode: " +
							f"{self.model_config.mode.lddt} for lddt head..." )

		self.adapter = get_adapter(
			c_z = self.c_z,
			config = self.model_config,
			device = self.device )


	def no_grad_for_sm( self ):
		"""
		Turn OFF gradient computation for the StructureModule.
		"""
		print( "Switching OFF gradient computation for StructureModule." )
		for p in self.structure_module.parameters():
			p.requires_grad_( False )


	def get_inter_chain_mask( self, asym_id: torch.Tensor ):
		"""
		Create a binary mask to ignore intra-chain interactions.
		"""
		if self.model_config.adapter.inter_mask:
			if asym_id is not None:
				# inter_chain_mask -> [N, N, 1]
				inter_chain_mask = ( 
					asym_id[..., None] != asym_id[..., None, :]
					).to( self.device ).squeeze( 0 ). unsqueeze( -1 )
		else:
			inter_chain_mask = None
		return inter_chain_mask


	def predict( self, evo_output, gt_features, batch ):
		"""
		Run the structure module and auxillary heads (lddt) module.
		If specified, mask out the intra-chain elements in pair_rep (z).
		Apply the adapter to perturb the z.
		"""
		outputs = {}
		# z -> [N, N, 128]
		z = evo_output.get( "pair" )
		inter_chain_mask = self.get_inter_chain_mask( asym_id = batch.get( "asym_id" ) )
		evo_output["pair"] = self.adapter( z = z, inter_chain_mask = inter_chain_mask )

		# Don't need the full Evoformer dict, just the Pair and Single representation.
		outputs["sm"] = self.structure_module.forward(
			evoformer_output_dict = evo_output,
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
			# The AuxillaryHeads module requires pair, Single representations in the output dict.
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
		return [self.adapter]


################################################################################
################################################################################
class SinglePerturbation( LoadState, Model ):
	"""
	Compose the AF2 structure module and pLDDT head.
	Modify the Single representation. The StructureModule can be kept frozen.
	This class wraps multiple methods to modify the pair representation (z):
		1. Linear perturbation
		2. Gating
		3. LoRA
		4. FiLM
	Further, we use a binary mask to ignore intra-chain elements in z.
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
		# Feature dim for pair representation.
		self.c_z = 384

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
		if self.model_config.freeze_sm:
			self.no_grad_for_sm()

		if self.model_config.mode.plddt == "train":
			print( "Using plddt head in train mode" )
			self.plddt.train()
		elif self.model_config.mode.plddt == "eval":
			print( "Using plddt head in eval mode" )
			self.plddt.eval()
		else:
			raise ValueError( "Incorrect mode: " +
							f"{self.model_config.mode.lddt} for lddt head..." )

		self.adapter = get_adapter(
			c_z = self.c_z,
			config = self.model_config,
			device = self.device )


	def no_grad_for_sm( self ):
		"""
		Turn OFF gradient computation for the StructureModule.
		"""
		print( "Switching OFF gradient computation for StructureModule." )
		for p in self.structure_module.parameters():
			p.requires_grad_( False )


	def get_inter_chain_mask( self, asym_id: torch.Tensor ):
		"""
		Create a binary mask to ignore intra-chain interactions.
		"""
		if self.model_config.adapter.inter_mask:
			if asym_id is not None:
				# inter_chain_mask -> [N, N, 1]
				inter_chain_mask = ( 
					asym_id[..., None] != asym_id[..., None, :]
					).to( self.device ).squeeze( 0 ). unsqueeze( -1 )
		else:
			inter_chain_mask = None
		return inter_chain_mask


	def predict( self, evo_output, gt_features, batch ):
		"""
		Run the structure module and auxillary heads (lddt) module.
		If specified, mask out the intra-chain elements in pair_rep (z).
		Apply the adapter to perturb the z.
		"""
		outputs = {}
		# z -> [N, N, 128]
		s = evo_output.get( "single" )
		evo_output["single"] = self.adapter( rep = s, inter_chain_mask = None )

		# Don't need the full Evoformer dict, just the Pair and Single representation.
		outputs["sm"] = self.structure_module.forward(
			evoformer_output_dict = evo_output,
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
			# The AuxillaryHeads module requires pair, Single representations in the output dict.
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
		return [self.adapter]
