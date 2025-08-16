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
		print( "Using StructureModuleFineTuning" )
		model = StructureModuleFineTuning( system_features, ofold_config, model_config,
											mode, is_multimer, device )
	elif model_config.name == "pair_perturbation":
		print( f"Using PairPerturbation with adapter = {model_config.adapter.name}" )
		model = PairPerturbation( system_features, ofold_config, model_config,
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
		evo_output --> dict containing Pair, Single representtaions.
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


################################################################################
################################################################################
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


	def predict( self, evo_output, gt_features, batch ):
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


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.structure_module]

################################################################################
################################################################################
class LinearPerturbation( nn.Module ):
	"""
	Modify the pair representation using a linear perturbation to it.
	"""
	def __init__( self, c_z: int, alpha: float, device: str ):
		super().__init__()
		self.linear = nn.Linear( in_features = c_z, out_features = c_z, device = device )
		self.alpha = alpha
		self.lnorm = nn.LayerNorm( c_z, device = device )

		# Initialize weights to a small value.
		nn.init.normal_( self.linear.weight, mean = 0.0, std = 1e-4 )
		# Initialize biases to 0.
		nn.init.zeros_( self.linear.bias )


	def forward( self, z: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		z_mod = self.linear( self.lnorm( z ) )*self.alpha
		if inter_chain_mask is not None:
			z_mod = z_mod*inter_chain_mask
		return z + z_mod


class GatedLinearUnit( nn.Module ):
	"""
	Perturbing the pair representation using a gated linear unit.
	Can use a sigmoid or tanh gate.
	"""
	def __init__( self, c_z: int, gate: str, alpha: float, device: str ):
		super().__init__()
		self.linear1 = nn.Linear( in_features = c_z, out_features = c_z, device = device )
		self.linear2 = nn.Linear( in_features = c_z, out_features = c_z, device = device )

		if gate == "sigmoid":
			self.gate = nn.Sigmoid()
		elif gate == "tanh":
			self.gate = nn.Tanh()
		else:
			raise ValueError( f"Unsupported gate {gate}.." )

		# Initialize weights to a small value.
		nn.init.normal_( self.linear1.weight, mean = 0.0, std = 1e-4 )
		nn.init.normal_( self.linear2.weight, mean = 0.0, std = 1e-4 )
		# Initialize biases to 0.
		nn.init.zeros_( self.linear1.bias )
		nn.init.zeros_( self.linear2.bias )

		self.alpha = alpha


	def forward( self, z: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		z_mod = self.linear1( z ) * self.gate( self.linear2( z ) )*self.alpha
		if inter_chain_mask is not None:
			z_mod = z_mod*inter_chain_mask
		return z + self.alpha*z_mod


class LoRA( nn.Module ):
	"""
	Low-Rank Adaptation (LoRA).
	"""
	def __init__( self, c_z: int, lora_k: int, alpha: float, device: str ):
		super().__init__()
		# Using standard Linear weight initialization.
		self.U = nn.Linear( in_features = c_z, out_features = lora_k, bias = False, device = device )
		self.V = nn.Linear( in_features = lora_k, out_features = c_z, bias = False, device = device )
		self.alpha = alpha
		self.lnorm = nn.LayerNorm( c_z, device = device )


	def forward( self, z: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		z_mod = self.V( self.U( self.lnorm( z ) ) )*self.alpha
		if inter_chain_mask is not None:
			z_mod = z_mod*inter_chain_mask
		return z + z_mod


class FiLM( nn.Module ):
	"""
	Feature-wise Linear Modulation (FiLM)
	z_mod = gamma*z + beta
	Here, gamma and beta represent the scale and shift.

	A custom implementation inspired from the original
		FiLM paper (https://doi.org/10.48550/arXiv.1709.07871).
	The current implementation is differnt from the above:
		I am using FiLM(z) as a residual connection.
		I am using LNorm which as per the paper is not needed.
		I am learning a constant gamma and beta for each feaure.
	"""
	def __init__( self, c_z: int, device: str ):
		super().__init__()
		self.gamma = nn.Parameter( torch.ones( c_z, device = device ), requires_grad = True )
		self.beta = nn.Parameter( torch.ones( c_z, device = device ), requires_grad = True )
		self.lnorm = nn.LayerNorm( c_z, device = device )


	def forward( self, z: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		gamma = self.gamma.view( 1, 1, 1, -1 )
		beta = self.beta.view( 1, 1, 1, -1 )

		y = self.lnorm( z )
		z_mod = y*gamma + beta
		if inter_chain_mask is not None:
			z_mod = z_mod*inter_chain_mask
		return z + z_mod


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

		self.adapter = self.get_adapter()


	def no_grad_for_sm( self ):
		"""
		Turn OFF gradient computation for the StructureModule.
		"""
		print( "Switching OFF gradient computation for StructureModule." )
		for p in self.structure_module.parameters():
			p.requires_grad_( False )


	def get_adapter( self ):
		"""
		Return the required adapter module.
		Initialize the weights and biases to 0.
		"""
		adapter_name = self.model_config.adapter.name
		if adapter_name == "linear_perturb":
			adapter = LinearPerturbation(
				c_z = self.c_z,
				alpha = self.model_config.adapter.alpha,
				device = self.device )
		elif "gating" in adapter_name:
			gate = adapter_name.split( "_" )[0]
			adapter = GatedLinearUnit(
				c_z = self.c_z,
				gate = gate,
				alpha = self.model_config.adapter.alpha,
				device = self.device )
		elif adapter_name == "lora":
			adapter = LoRA(
				c_z = self.c_z,
				lora_k = self.model_config.adapter.lora_k,
				alpha = self.model_config.adapter.alpha,
				device = self.device )
		elif adapter_name == "film":
			adapter = FiLM(
				c_z = self.c_z,
				device = self.device )
		else:
			raise ValueError( f"Incorrect adapter specified: {adapter_name}..." )

		return adapter


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
