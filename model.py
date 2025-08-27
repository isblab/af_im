"""
This script contains model classes.
"""
import os
from typing import List, Tuple, Dict, Any
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
	elif model_config.name == "single_perturbation":
		print( f"Using SinglePerturbation with adapter = {model_config.adapter.name}" )
		model = SinglePerturbation( system_features, ofold_config, model_config,
											mode, is_multimer, device )
	elif model_config.name == "rigid_transform":
		print( f"Using RigidTransformation = {model_config.name}" )
		model = PoseSampling( model_config, device )
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


	def forward( self, rep: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		rep_mod = self.linear( self.lnorm( rep ) )*self.alpha
		if inter_chain_mask is not None:
			rep_mod = rep_mod*inter_chain_mask
		return rep + rep_mod


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


	def forward( self, rep: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		rep_mod = self.linear1( rep ) * self.gate( self.linear2( rep ) )*self.alpha
		if inter_chain_mask is not None:
			rep_mod = z_mod*inter_chain_mask
		return rep + self.alpha*rep_mod


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


	def forward( self, rep: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		rep_mod = self.V( self.U( self.lnorm( rep ) ) )*self.alpha
		if inter_chain_mask is not None:
			rep_mod = rep_mod*inter_chain_mask
		return rep + rep_mod


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


	def forward( self, rep: torch.Tensor, inter_chain_mask: torch.Tensor = None ):
		gamma = self.gamma.view( 1, 1, 1, -1 )
		beta = self.beta.view( 1, 1, 1, -1 )

		y = self.lnorm( rep )
		rep_mod = y*gamma + beta
		if inter_chain_mask is not None:
			rep_mod = rep_mod*inter_chain_mask
		return rep + rep_mod


def get_adapter( c_z: int, config: mlc.ConfigDict, device: str ):
	"""
	Return the required adapter module.
	Initialize the weights and biases to 0.
	"""
	adapter_name = config.adapter.name
	if adapter_name == "linear_perturb":
		adapter = LinearPerturbation(
			c_z = c_z,
			alpha = config.adapter.alpha,
			device = device )
	elif "gating" in adapter_name:
		gate = adapter_name.split( "_" )[0]
		adapter = GatedLinearUnit(
			c_z = c_z,
			gate = gate,
			alpha = config.adapter.alpha,
			device = device )
	elif adapter_name == "lora":
		adapter = LoRA(
			c_z = c_z,
			lora_k = config.adapter.lora_k,
			alpha = config.adapter.alpha,
			device = device )
	elif adapter_name == "film":
		adapter = FiLM(
			c_z = c_z,
			device = device )
	else:
		raise ValueError( f"Incorrect adapter specified: {adapter_name}..." )

	return adapter

################################################################################
################################################################################
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

################################################################################
################################################################################
class RigidTransformation( nn.Module ):
    """
    Learn a rigid transformation comprising a quaternion and a translation vector.
    """
    def __init__( self, n_coords: int, c_hidden: int, device: str ):
        super().__init__()
        self.rigid_transform = nn.Sequential(
            nn.Linear( in_features = n_coords, out_features = c_hidden ),
            nn.ReLU(),
            nn.Linear( in_features = c_hidden, out_features = 16 ),
            nn.ReLU()
            ).to( device )
        self.quaternion = nn.Linear( in_features = 16, out_features = 4, bias = False, device = device )
        self.translation = nn.Linear( in_features = 16, out_features = 3, bias = False, device = device )

    def forward( self, x: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the coordinates for each atom, predict a quaternion
            and translation vector.
        coords -> [R, 3]; R -> no. of rigid bodies
        """
        x = self.rigid_transform( x )
        quat = self.quaternion( x )
        trans = self.translation( x )
        return quat, trans


class PoseSampling( Model ):
	"""
	Predict and apply a rigid transformation to sample conformations of the
		system that satisfy the data.
	"""
	def __init__( self, 
				model_config: mlc.ConfigDict,
				device: str ):
		Model.__init__( self )
		self.model_config = model_config

		self.rigid = RigidTransformation( n_coords = 3, c_hidden = 32, device = device )


	def predict( self, outputs: Dict[str, Any],
				gt_features: Dict[str, Any],
				batch: Dict[str, Any] ):
		"""
		Update the current positions by applying a Rigid Transformation.
		"""
		# [B, N, 37, 3]; B -> batch size = 1; N -> no. of residues.
		final_atom_positions = outputs.pop( "final_atom_positions" )

		rigid_bodies, init_mean_coords = self.split_into_rigid_bodies(
			final_atom_positions = final_atom_positions,
			asym_id = outputs["asym_id"]
			)

		# [B, 4] and [B, 3]
		quat, trans = self.rigid( init_mean_coords )

		transformed_positions = []
		for rb, q, t in zip( rigid_bodies, quat, trans ):
			R = self.quat_to_rotmat( quat = q )

			Rt_rb = self.apply_transform(
				coords = rb,
				R = R,
				trans = t )
			transformed_positions.append( Rt_rb )
		transformed_positions = torch.cat( transformed_positions, dim = 1 )
		outputs["final_atom_positions"] = transformed_positions*gt_features["atom37_atom_exists"].unsqueeze( -1 )
		print( outputs["final_atom_positions"].shape )
		# for k in outputs:
		# 	if isinstance( outputs[k], dict ):
		# 		for m in outputs[k]:
		# 			print( k, "  ", m, "  ", outputs[k][m].shape )
		# 	else:
		# 		print( k, "  ", outputs[k].shape )

		return outputs, batch


	def split_into_rigid_bodies( self,
			final_atom_positions: torch.Tensor,
			asym_id: torch.Tensor ):
		"""
		OTG implemntation for now.
		Split the final_atom_position into rigid bodies.
		Currently keeping each chain as a separate rigid body.
		Also, obtain mean coordinates for all rigid bodies to be
			used as init coords for the model.
		"""
		# with torch.no_grad():
		rigid_bodies = []
		unique_asym_ids = torch.unique( asym_id )

		# asym_id -> [B, N]
		for a_id in unique_asym_ids:
			idx = torch.where( a_id == asym_id )[1]
			rb = final_atom_positions[:, idx, :, :]
			rigid_bodies.append( rb )

		init_mean_coords = []
		for rb in rigid_bodies:
			# [B, N, 37, 3] -> [B, 3]
			init_mean_coords.append(
				torch.mean( rb, dim = ( 1, 2 ) )
				)
		init_mean_coords  = torch.cat( init_mean_coords, dim = 0 )
		return rigid_bodies, init_mean_coords


	def quat_to_rotmat( self, quat: torch.Tensor ):
		"""
		Taken from openfold/utils/geometry/rotation_matrix.from_quaternion
		"""
		w, x, y, z = quat.unbind( -1 )

		inv_norm = torch.rsqrt( torch.clamp( w**2 + x**2 + y**2 + z**2, min = 1e-8 ) )
		w = w * inv_norm
		x = x * inv_norm
		y = y * inv_norm
		z = z * inv_norm

		xx = 1.0 - 2.0 * ( y ** 2 + z ** 2 )
		xy = 2.0 * ( x * y - w * z )
		xz = 2.0 * ( x * z + w * y )
		yx = 2.0 * ( x * y + w * z )
		yy = 1.0 - 2.0 * ( x ** 2 + z ** 2 )
		yz = 2.0 * ( y * z - w * x )
		zx = 2.0 * ( x * z - w * y )
		zy = 2.0 * ( y * z + w * x )
		zz = 1.0 - 2.0 * ( x ** 2 + y ** 2 )

		R = torch.stack( [
			torch.stack( [xx, xy, xz], dim = -1 ),
			torch.stack( [yx, yy, yz], dim = -1 ),
			torch.stack( [zx, zy, zz], dim = -1 )
			], dim = -2 )
		return R


	def apply_transform( self,
			coords: torch.Tensor,
			R: torch.Tensor,
			trans: torch.Tensor ) -> torch.Tensor:
		"""
		"""
		# coords_rot = torch.matmul( coords, R )
		coords_rot = torch.einsum( "bnac,cj->bnaj", coords, R )
		coords_new = coords_rot + trans

		return coords_new


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.rigid]
