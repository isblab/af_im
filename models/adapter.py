"""
Contains several adapter for inference time fine-tuning.
"""
from typing import List, Tuple, Dict, Any
import ml_collections as mlc

import torch
from torch import nn


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


	def forward( self,
			 rep: torch.Tensor,
			 inter_chain_mask: torch.Tensor = None
			 ) -> torch.Tensor:
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


	def forward( self,
			 rep: torch.Tensor,
			 inter_chain_mask: torch.Tensor = None
			 ) -> torch.Tensor:
		rep_mod = self.linear1( rep ) * self.gate( self.linear2( rep ) )*self.alpha
		if inter_chain_mask is not None:
			rep_mod = rep_mod*inter_chain_mask
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


	def forward( self,
			 rep: torch.Tensor,
			 inter_chain_mask: torch.Tensor = None
			 ) -> torch.Tensor:
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


	def forward( self,
			 rep: torch.Tensor,
			 inter_chain_mask: torch.Tensor = None
			 ) -> torch.Tensor:
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