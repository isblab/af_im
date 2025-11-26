"""
Using OpenFold for recycling a prediction.
"""
from typing import Tuple, Dict, Any
import copy
import ml_collections as mlc

import torch
from torch import nn

from openfold.model.model import AlphaFold
from openfold.utils.import_weights import (
    import_jax_weights_ )
from openfold.utils.script_utils import get_model_basename

from openfold.utils.tensor_utils import (
    tensor_tree_map )


class Recycler():
	"""
	Use the predicted configuration (template) to bias the AF2 prediction.
		To let AF2 feel the effect of the new template, use AF sampling techniques.
	"""
	def __init__( self,
			ofold_config: mlc.ConfigDict,
			jax_param_path: str,
			num_iters: int,
			inference_mode: str,
			activate_dropouts: str,
			device: str ):
		self.ofold_config = ofold_config
		self.num_iters = num_iters
		self.device = device

		# When using train mode, tune_chunk_size is expected to be False.
		if inference_mode == "train" or activate_dropouts in ["evoformer", "structure_module"]:
			self.ofold_config.model.template.template_pair_stack.tune_chunk_size = False
		for path in jax_param_path.split( "," ):
			model_basename = get_model_basename(path)
			model_version = "_".join(model_basename.split("_")[1:])
			self.alphafold = AlphaFold( self.ofold_config )
			if inference_mode == "eval":
				print( f"OpenFold inference in eval mode..." )
				self.alphafold = self.alphafold.eval()

				if activate_dropouts == "none":
					pass
				elif activate_dropouts == "evoformer":
					print( "Switched ON Evoformer dropouts..." )
					self.alphafold.evoformer.train()
				elif activate_dropouts == "structure_module":
					print( "Switched ON StructureModule dropouts..." )
					self.alphafold.structure_module.train()
				else:
					raise ValueError(
						f"Incorrect vaue specified for activate_dropouts - {activate_dropouts}." +
						" use evoformer/structure_module/none..." )
			elif inference_mode == "train":
				print( f"OpenFold inference in train mode..." )
				self.alphafold = self.alphafold.train()
			else:
				raise ValueError( f"Incorrect inference mode - {inference_mode} - specified. Use train/eval..." )
			import_jax_weights_(
				self.alphafold, path, version = model_version )


	def forward( self,
			out: Dict[str, Any],
			batch: Dict[str, Any] ):
		"""
		"""
		if None in [out["msa"], out["pair"], out["final_atom_positions"]]:
			print( "None detected in MSA/Pair rep or final_atom_positions." +
				" Recycling embedder will ignore whatever final_atom_positions is provided and initialize to 0..." )
		# Initialize recycling embeddings
		if out["msa"] is None:
			m_1_prev = out["msa"]
		else:
			m_1_prev = out["msa"][..., 0, :, :]
		z_prev, x_prev = out["pair"], out["final_atom_positions"]
		prevs = [m_1_prev, z_prev, x_prev]
		del out

		is_grad_enabled = torch.is_grad_enabled()

		# Main recycling loop
		# num_iters = 1 # batch["aatype"].shape[-1]
		early_stop = False
		num_recycles = 0
		#for cycle_no in range(num_iters):
		# Select the features for the current recycling cycle.
		# fetch_cur_batch = lambda t: t[..., 0]
		# feats = tensor_tree_map(fetch_cur_batch, batch)
		feats = batch
		del batch

		self.alphafold = self.alphafold.to( self.device )

		for cycle_no in range( self.num_iters ):
			# Enable grad iff we're training and it's the final recycling layer
			is_final_iter = cycle_no == ( self.num_iters - 1 ) or early_stop
			with torch.set_grad_enabled( is_grad_enabled and is_final_iter ):
				if is_final_iter:
					# Sidestep AMP bug (PyTorch issue #65766)
					if torch.is_autocast_enabled():
						torch.clear_autocast_cache()

			outputs, m_1_prev, z_prev, x_prev, early_stop = self.alphafold.iteration(
				feats,
				prevs )

			num_recycles += 1

			if not is_final_iter:
				del outputs
				prevs = [m_1_prev, z_prev, x_prev]
				del m_1_prev, z_prev, x_prev
			else:
				break

		outputs["num_recycles"] = torch.tensor( [num_recycles] )
		print( f"Num recycles = {num_recycles}..." )

		outputs["asym_id"] = feats["asym_id"]

		# Run auxiliary heads
		outputs.update( self.alphafold.aux_heads( outputs ) )

		self.alphafold = self.alphafold.to( "cpu" )

		return outputs


		# # Enable grad iff we're training and it's the final recycling layer
		# is_final_iter = True
		# with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
		# 	if is_final_iter:
		# 		# Sidestep AMP bug (PyTorch issue #65766)
		# 		if torch.is_autocast_enabled():
		# 			torch.clear_autocast_cache()

		# 	self.alphafold = self.alphafold.to( self.device )
		# 	# Run the next iteration of the model
		# 	outputs, _, _, _, _ = self.alphafold.iteration(
		# 		feats,
		# 		prevs )

		# outputs["num_recycles"] = torch.tensor( [1] )
		# #outputs["num_recycles"] = torch.tensor(num_recycles, device=feats["aatype"].device)

		# outputs["asym_id"] = feats["asym_id"]
		# #if "asym_id" in batch:
		# #	outputs["asym_id"] = feats["asym_id"]

		# # Run auxiliary heads
		# outputs.update( self.alphafold.aux_heads( outputs ) )

		# self.alphafold = self.alphafold.to( "cpu" )

		# return outputs

