"""
Contains modules for modifying the Evoformer single and pair
	representations at inference time.
"""
from typing import List, Tuple, Dict, Any
import ml_collections as mlc

import torch
from torch import nn

from models.base_model import Model


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


def quat_to_rotmat( quat: torch.Tensor ):
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


def apply_transform(
		coords: torch.Tensor,
		R: torch.Tensor,
		trans: torch.Tensor ) -> torch.Tensor:
	"""
	"""
	# coords_rot = torch.matmul( coords, R )
	coords_rot = torch.einsum( "bnac,cj->bnaj", coords, R )
	coords_new = coords_rot + trans

	return coords_new


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
	KM-TODO: Pull out the rigid body function.
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

		rigid_bodies, init_mean_coords = split_into_rigid_bodies(
			final_atom_positions = final_atom_positions,
			asym_id = outputs["asym_id"]
			)

		# [B, 4] and [B, 3]
		quat, trans = self.rigid( init_mean_coords )

		transformed_positions = []
		for rb, q, t in zip( rigid_bodies, quat, trans ):
			R = quat_to_rotmat( quat = q )

			Rt_rb = apply_transform(
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
