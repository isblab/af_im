"""
Contains a wrapper around the AlphaFold model for recycling a prediction and integrating eith
	MSA subsampling
	Column masking
	MSA/Pair representation corruption.
"""
from typing import Tuple, Dict, Any
import copy
import ml_collections as mlc

import torch
from torch import nn

from openfold.data import data_transforms_multimer
import openfold.np.residue_constants as residue_constants
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat
)
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat
)
from openfold.model.template import (
    embed_templates_average,
    embed_templates_offload,
)

from openfold.utils.tensor_utils import (
    add
)

from loader import LoadState
from base_model import Model


class Recycler( LoadState, Model, nn.Module ):
	"""
	Taken from openfold/model/model.py -> AlphaFold.iterations()
	Recycling a prediction (C-beta xyz) complemented with AF-sampling methods:
		MSA subsampling, column masking, and MSA/Pair representation corruption.
	m -> MSA embedding. [*, S, N_res, C_m]
	z -> [*, N_res, N_res, C_z] pair embedding
	x -> [*, N_res, 3] predicted C_beta coordinates
	"""
	def __init__( self, feats: mlc.ConfigDict,
						ofold_config: mlc.ConfigDict,
						model_config: mlc.ConfigDict,
						mode: str, is_multimer: bool, device: str ):
		LoadState.__init__( self, ofold_config, mode, is_multimer, device )
		Model.__init__( self )
		self.ofold_config = ofold_config
		self.model_config = model_config
		self.feats = feats

		layers = ["input_embedder", "template_embedder", "recycling_embedder",
			"extra_msa", "evoformer", "structure_module", "aux-head"]

		# All models in eval mode.
		self.load_pretrained_models( layers )
		self.input_embedder.eval()
		self.recycling_embedder.eval()
		self.template_embedder.eval()
		self.extra_msa_embedder.eval()
		self.extra_msa_stack.eval()
		self.evoformer.eval()
		self.structure_module.eval()
		self.aux_heads.eval()


	def init_global_vars( self ):
		# This needs to be done manually for DeepSpeed's sake
		dtype = next( self.parameters() ).dtype
		for k in self.feats:
			if self.feats[k].dtype == torch.float32:
				self.feats[k] = self.feats[k].to( dtype = dtype )

		# Grab some data about the input
		self.batch_dims = self.feats["target_feat"].shape[:-2]
		self.no_batch_dims = len( self.batch_dims )
		self.n = self.feats["target_feat"].shape[-2]
		self.n_seq = self.feats["msa_feat"].shape[-3]
		self.device = self.feats["target_feat"].device

		# Controls whether the model uses in-place operations throughout
		# The dual condition accounts for activation checkpoints
		self.inplace_safe = not ( self.training or torch.is_grad_enabled() )

	################################################################################
	################################################################################
	def run_input_embedder(
			self,
			m_1_prev: torch.Tensor,
			z_prev: torch.Tensor,
			final_atom_position: torch.Tensor
			) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Create the initial MSA and Pair representation from input features.
		Recycle the prediction to bias the pair representation.
			In the OpenFold implementation, the recycled pair representation is
				added prior to the template embedding.
				Here, I add the recycled pair rep post the template embedding.
		Use AF smapling techniques either the input or representation level.
			1. Input
				MSA subsampling, column masking
			2. Representation
				Dropouts on MSA representation along the S, N_res, C_m dimensions.
		"""
		feats = copy.deepcopy( self.feats )
		# Get initial MSA and Pair representtaion.
		self.m_init, self.z_init = self.get_inital_msa_pair_representations( feats = feats )
		if self.config.extra_msa.enabled:
			self.a = self.embed_extra_msa_feats(
				feats = feats, dtype = z.dtype )

		# As per the OpenFold implementation.
		# Add the recycled Pair representation
		m, z = self.recycle_prediction(
					m_1_prev = m_1_prev,
					z_prev = z_prev,
					z = z,
					feats = feats,
					x_prev = final_atom_position )

		# Prep some features
		seq_mask = self.feats["seq_mask"]
		pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
		msa_mask = self.feats["msa_mask"]

		if self.config.template.enabled:
			# Get template embedding.
			self.template_embeds = self.get_template_embedding(
				z = z,
				feats = feats )
			msa_mask = self.merge_template_emb_with_msa(
				m = m,
				msa_mask = msa_mask,
				template_embeds = self.template_embeds )

		z = self.update_z_with_templates( z = z )

		if self.config.extra_msa.enabled:
			z = self.update_pair_rep_with_extra_msa(
				m = m, z = z,
				feats = feats, pair_mask = pair_mask )

		return m, z, msa_mask, pair_mask


	def get_inital_msa_pair_representations(
			self,
			feats: Dict[str, Any]
			) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Create the initial MSA and Pair representation from input features.
		This can be reused without modification if not perturbing the
			input MSA and templates. Hope to save some time and memory.
		"""
		if self.is_multimer:
			# Initialize the MSA and pair representations
			# m: [*, S_c, N, C_m]
			# z: [*, N, N, C_z]
			m, z = self.input_embedder( feats )
		else:
			# Initialize the MSA and pair representations
			# m: [*, S_c, N, C_m]
			# z: [*, N, N, C_z]
			m, z = self.input_embedder(
				feats["target_feat"],
				feats["residue_index"],
				feats["msa_feat"],
				inplace_safe = self.inplace_safe,
			)
		return m, z


	def embed_extra_msa_feats( self,
			feats: Dict[str, Any],
			 dtype ):
		if self.is_multimer:
			extra_msa_fn = data_transforms_multimer.build_extra_msa_feat
		else:
			extra_msa_fn = build_extra_msa_feat

		# [*, S_e, N, C_e]
		extra_msa_feat = extra_msa_fn( feats ).to( dtype = dtype )
		a = self.extra_msa_embedder( extra_msa_feat )
		return a


	def embed_templates( self,
			batch,
			feats,
			z,
			pair_mask,
			templ_dim ):
		if self.globals.is_multimer:
			asym_id = feats["asym_id"]
			multichain_mask_2d = (
				asym_id[..., None] == asym_id[..., None, :]
			)
			template_embeds = self.template_embedder(
				batch,
				z,
				pair_mask.to( dtype = z.dtype ),
				templ_dim,
				chunk_size = self.globals.chunk_size,
				multichain_mask_2d = multichain_mask_2d,
				use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
				use_lma = self.globals.use_lma,
				inplace_safe = self.inplace_safe,
				_mask_trans = self.config._mask_trans
			)
			feats["template_torsion_angles_mask"] = (
				template_embeds["template_mask"]
			)
		else:
			if self.template_config.offload_templates:
				return embed_templates_offload( self,
												batch, z, pair_mask, templ_dim, inplace_safe = self.inplace_safe,
												)
			elif self.template_config.average_templates:
				return embed_templates_average( self,
												batch, z, pair_mask, templ_dim, inplace_safe = self.inplace_safe,
												)

			template_embeds = self.template_embedder(
				batch,
				z,
				pair_mask.to( dtype = z.dtype ),
				templ_dim,
				chunk_size = self.globals.chunk_size,
				use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
				use_lma = self.globals.use_lma,
				inplace_safe = self.inplace_safe,
				_mask_trans = self.config._mask_trans
			)

		return template_embeds


	def get_template_embedding( self,
			z: torch.Tensor,
			pair_mask: torch.Tensor,
			feats: Dict[str, Any] ) -> torch.Tensor:
		"""
		Use template features to get the template embeddings.
		"""
		# Embed the templates.
		#if self.config.template.enabled:
		template_feats = {
			k: v for k, v in feats.items() if k.startswith("template_")
		}

		template_embeds = self.embed_templates(
			template_feats,
			feats,
			z,
			pair_mask.to( dtype = z.dtype ),
			self.no_batch_dims
		)
		return template_embeds

	################################################################################
	def recycle_prediction( self,
			m_1_prev: torch.Tensor,
			z_prev: torch.Tensor,
			z: torch.Tensor,
			feats: Dict[str, Any],
			x_prev: torch.Tensor ):
		"""
		Rcycling a predicted structure with the full recyling embedder.
		"""
		# Initialize the recycling embeddings, if needs be 
		if None in [m_1_prev, z_prev, x_prev]:
			# [*, N, C_m]
			m_1_prev = m.new_zeros(
				( *self.batch_dims, self.n, self.config.input_embedder.c_m ),
				requires_grad = False,
			)

			# [*, N, N, C_z]
			z_prev = z.new_zeros(
				( *self.batch_dims, self.n, self.n, self.config.input_embedder.c_z ),
				requires_grad = False,
			)

			# [*, N, 3]
			x_prev = z.new_zeros(
				( *self.batch_dims, self.n, residue_constants.atom_type_num, 3 ),
				requires_grad = False,
			)

		pseudo_beta_x_prev = pseudo_beta_fn(
			feats["aatype"], x_prev, None
		).to( dtype = z.dtype )

		# The recycling embedder is memory-intensive, so we offload first
		if self.offload_inference and self.inplace_safe:
			m = m.cpu()
			z = z.cpu()

		# m_1_prev_emb: [*, N, C_m]
		# z_prev_emb: [*, N, N, C_z]
		m_1_prev_emb, z_prev_emb = self.recycling_embedder(
			m_1_prev,
			z_prev,
			pseudo_beta_x_prev,
			inplace_safe = self.inplace_safe,
		)

		del pseudo_beta_x_prev

		if self.offload_inference and self.inplace_safe:
			m = m.to( m_1_prev_emb.device )
			z = z.to (z_prev.device )

		# [*, S_c, N, C_m]
		m[..., 0, :, :] += m_1_prev_emb

		# [*, N, N, C_z]
		z = add( z, z_prev_emb, inplace = self.inplace_safe )

		# Deletions like these become significant for inference with large N,
		# where they free unused tensors and remove references to others such
		# that they can be offloaded later
		del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

		return m, z


	################################################################################
	#def inject_prediction_into_pair_rep(
	#		self,
	#		z: torch.Tensor,
	#		feats: Dict[str, Any],
	#		final_atom_position: torch.Tensor ):
	#	"""
	#	Rcycling a predicted structure.
	#	Don't need the full RecyclingEmbedder. Just need to update
	#		the pair rep with the prediction.
	#	"""
	#	x_prev = final_atom_position.clone()
	#	del final_atom_position

	#	pseudo_beta_x_prev = pseudo_beta_fn(
	#		feats["aatype"], x_prev, None
	#	).to( dtype = z.dtype )

	#	if self.offload_inference and self.inplace_safe:
	#		x_prev = x_prev.cpu()
	#		z = z.cpu()

	#	del pseudo_beta_x_prev

	#	z_prev_emb = self.pair_update( x = x_prev, z = z )
	#	# [*, N, N, C_z]
	#	z = add( z, z_prev_emb, inplace = self.inplace_safe )
	#	return z


	#def pair_update( self, x: torch.Tensor, z: torch.Tensor ) -> torch.Tensor:
	#	"""
	#	Using the prediction to bias the pair representation.
	#	"""
	#	# [*, N, N, C_z]
	#	z_update = self.recycle_embedder.layer_norm_z( z )
	#	if ( self.inplace_safe ):
	#		z.copy_( z_update )
	#		z_update = z

	#	# This squared method might become problematic in FP16 mode.
	#	bins = torch.linspace(
	#		self.min_bin,
	#		self.max_bin,
	#		self.no_bins,
	#		dtype = x.dtype,
	#		device = x.device,
	#		requires_grad = False,
	#	)
	#	squared_bins = bins ** 2
	#	upper = torch.cat(
	#		[squared_bins[1:], squared_bins.new_tensor( [self.inf] )], dim = -1
	#	)
	#	d = torch.sum(
	#		(x[..., None, :] - x[..., None, :, :]) ** 2, dim = -1, keepdims = True
	#	)

	#	# [*, N, N, no_bins]
	#	d = ( ( d > squared_bins ) * ( d < upper ) ).type( x.dtype )

	#	# [*, N, N, C_z]
	#	d = self.recycle_embedder.linear( d )
	#	z_update = add( z_update, d, self.inplace_safe )
	#	return z_update

	################################################################################
	def merge_template_emb_with_msa( self,
			m: torch.Tensor,
			torsion_angles_mask: torch.Tensor,
			msa_mask: torch.Tensor,
			template_embeds: Dict[str, torch.Tensor]
			) -> torch.Tensor:
		"""
		Merge the template featsures with the MSA features.
		"""
		if (
			"template_single_embedding" in template_embeds
		):
			# [*, S = S_c + S_t, N, C_m]
			m = torch.cat(
				[m, template_embeds["template_single_embedding"]],
				dim = -3
			)

			# [*, S, N]
			if not self.is_multimer:
				msa_mask = torch.cat(
					[msa_mask, torsion_angles_mask[..., 2]],
					dim = -2
				)
			else:
				msa_mask = torch.cat(
					[msa_mask, template_embeds["template_mask"]],
					dim = -2,
				)
		return msa_mask


	def update_pair_rep_with_templates(
			self, z: torch.Tensor ) -> torch.Tensor:
		"""
		Add the template embedding to pair representation.
		As per the OpenFold implementation, th pair representation (z)
			here is informed by the target_feat and the rcycled template.
		"""
		# [*, N, N, C_z]
		z = add( z,
				self.template_embeds.pop( "template_pair_embedding" ),
				self.inplace_safe )
		return z


	def update_pair_rep_with_extra_msa( self,
			m: torch.Tensor,
			z: torch.Tensor,
			feats: Dict[str, Any],
			pair_mask: torch.Tensor ):
		"""
		Update the pair representation with extra MSA features embedding.
		"""
		# Embed extra MSA features + merge with pairwise embeddings
		if self.config.extra_msa.enabled:
			if self.is_multimer:
				extra_msa_fn = data_transforms_multimer.build_extra_msa_feat
			else:
				extra_msa_fn = build_extra_msa_feat

			# [*, S_e, N, C_e]
			extra_msa_feat = extra_msa_fn( feats ).to( dtype = z.dtype )
			a = self.extra_msa_embedder( extra_msa_feat )

			if self.offload_inference:
				# To allow the extra MSA stack (and later the evoformer) to
				# offload its inputs, we remove all references to them here
				input_tensors = [a, z]
				del a, z

				# [*, N, N, C_z]
				z = self.extra_msa_stack._forward_offload(
					input_tensors,
					msa_mask = feats["extra_msa_mask"].to( dtype = m.dtype ),
					chunk_size = self.globals.chunk_size,
					use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
					use_lma = self.globals.use_lma,
					pair_mask = pair_mask.to( dtype = m.dtype ),
					_mask_trans = self.config._mask_trans,
				)

				del input_tensors
			else:
				# [*, N, N, C_z]
				z = self.extra_msa_stack(
					a, z,
					msa_mask = feats["extra_msa_mask"].to(dtype=m.dtype),
					chunk_size = self.globals.chunk_size,
					use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
					use_lma = self.globals.use_lma,
					pair_mask = pair_mask.to(dtype=m.dtype),
					inplace_safe = self.inplace_safe,
					_mask_trans = self.config._mask_trans,
				)
			return z

	################################################################################
	################################################################################
	def run_evoformer( self,
			m: torch.Tensor,
			z: torch.Tensor,
			msa_mask: torch.Tensor,
			pair_mask: torch.Tensor ):
		"""
		Evoformer updates the intial MSA and Pair representtaion.
		"""
		outputs = {}
		# Run MSA + pair embeddings through the trunk of the network
		# m: [*, S, N, C_m]
		# z: [*, N, N, C_z]
		# s: [*, N, C_s]          
		if self.offload_inference:
			input_tensors = [m, z]
			del m, z
			m, z, s = self.evoformer._forward_offload(
				input_tensors,
				msa_mask = msa_mask.to( dtype = input_tensors[0].dtype ),
				pair_mask = pair_mask.to( dtype = input_tensors[1].dtype ),
				chunk_size = self.globals.chunk_size,
				use_deepspeed_evo_attention = self.globals.use_deepspeed_evo_attention,
				use_lma = self.globals.use_lma,
				_mask_trans = self.config._mask_trans,
			)

			del input_tensors
		else:
			m, z, s = self.evoformer(
				m,
				z,
				msa_mask = msa_mask.to( dtype = m.dtype ),
				pair_mask = pair_mask.to( dtype = z.dtype ),
				chunk_size = self.globals.chunk_size,
				use_deepspeed_evo_attention=self.use_deepspeed_evo_attention,
				use_lma = self.globals.use_lma,
				use_flash = self.globals.use_flash,
				inplace_safe = self.inplace_safe,
				_mask_trans = self.config._mask_trans,
			)

		outputs["msa"] = m[..., :self.n_seq, :, :]
		outputs["pair"] = z
		outputs["single"] = s

		del m, z, s
		return outputs


class TheForge( LoadState, Model ):
	"""
	Given a set of rigid bodies, do the following:
		1. Predict rigid transformations to sample new configurations.
		2. Use the predicted configuration (template) to bias the AF2 prediction.
			> To let AF2 feel the effect of the new template, use AF sampling techniques.
	"""
	def __init__( self, system_features: mlc.ConfigDict,
						ofold_config: mlc.ConfigDict,
						model_config: mlc.ConfigDict,
						mode: str, is_multimer: bool, device: str ):
		LoadState.__init__( self, ofold_config, mode, is_multimer, device )
		Model.__init__( self )

