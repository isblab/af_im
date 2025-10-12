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

from loader import LoadState
from base_model import Model
from models.rigid_sampler import PoseSampling

class TheForge( LoadState, Model ):
	"""
	Given a set of rigid bodies, do the following:
		1. Predict rigid transformations to sample new configurations.
		2. Use the predicted configuration (template) to bias the AF2 prediction.
			> To let AF2 feel the effect of the new template, use AF sampling techniques.
	"""
	def __init__( self, feats: mlc.ConfigDict,
						ofold_config: mlc.ConfigDict,
						model_config: mlc.ConfigDict,
						jax_param_path: str,
						mode: str, is_multimer: bool, device: str ):
		LoadState.__init__( self, ofold_config, mode, is_multimer, device )
		Model.__init__( self )
		self.ofold_config = ofold_config
		self.model_config = model_config
		self.feats = feats


		for path in jax_param_path.split( "," ):
			model_basename = get_model_basename(path)
			model_version = "_".join(model_basename.split("_")[1:])
			self.alphafold = AlphaFold( self.ofold_config )
			self.alphafold = self.alphafold.eval()
			import_jax_weights_(
				self.alphafold, path, version = model_version )
			self.alphafold = self.alphafold.to( self.device )



	def forward( self ):
		"""
		"""

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

		return outputs, batch


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.rigid]

