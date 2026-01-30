"""
Contains modules for modifying the Evoformer single and pair
	representations at inference time.
"""
from typing import List, Tuple, Dict, Any
import ml_collections as mlc

import torch
from torch import nn

from models.base_model import Model
from models.rigid_body import get_rigid_body

def quat_to_rotmat( quat: torch.Tensor ):
	"""
	Taken from openfold/utils/geometry/rotation_matrix.from_quaternion
	"""
	w, x, y, z = quat.unbind( -1 )

	#inv_norm = torch.rsqrt( torch.clamp( w**2 + x**2 + y**2 + z**2, min = 1e-8 ) )
	#print("inv norm:", inv_norm )
	#w = w * inv_norm
	#x = x * inv_norm
	#y = y * inv_norm
	#z = z * inv_norm

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
	#coords_rot = torch.matmul( coords, R )
	coords_rot = torch.einsum( "bnac,bcj->bnaj", coords, R )
	coords_new = coords_rot + trans.unsqueeze( 0 ).unsqueeze( 0 )

	return coords_new


class RigidTransformation( nn.Module ):
	"""
	Learn a rigid transformation comprising a quaternion and a translation vector.
	Adapted from openFold/model/.
	"""
	def __init__( self,
		in_feats: int,
		c_hidden: int,
		clamp_translation: bool,
		device: str ):
		super().__init__()
		self.clamp_translation = clamp_translation
		self.rigid_transform = nn.Sequential(
			nn.Linear( in_features = in_feats, out_features = c_hidden ),
			nn.ReLU(),
			nn.Linear( in_features = c_hidden, out_features = c_hidden ),
			nn.ReLU()
			).to( device )
		self.quaternion = nn.Linear(
			in_features = c_hidden,
			out_features = 4,
			bias = False, device = device )
		self.translation = nn.Linear(
			in_features = c_hidden,
			out_features = 3,
			bias = False, device = device )

	def forward( self, x: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Given the coordinates for each atom, predict a quaternion
			and translation vector.
		coords -> [R, 3]; R -> no. of rigid bodies
		"""
		x = self.rigid_transform( x )
		quat = self.quaternion( x )
		#print( quat, " <--" )
		if torch.isnan( quat ).any() or torch.isinf( quat ).any():
			print( "Warning: quat contains NaN or Inf before normalization" )
		# Normalize.
		quat_norm = torch.norm( quat, dim = -1, keepdim = True ).clamp( 1e-8 )
		quat = quat/ quat_norm
		trans = self.translation( x )
		if self.clamp_translation:
			trans = torch.tanh( trans )
		return quat, trans

################################################################################
################################################################################
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
		self.device = device

		if self.model_config.input_feats == "com":
			in_feats = 3
		elif self.model_config.input_feats == "uvd":
			in_feats = 4
		else:
			raise ValueError( f"Unsupported input_feats specified. Use com/uvd..." )
		self.rigid = RigidTransformation(
			in_feats = in_feats,
			c_hidden = self.model_config.c_hidden,
			clamp_translation = self.model_config.clamp_translation,
			device = device )

		self.transformations = {k:[] for k in ["rotation", "translation"]}


	def create_input_feats( self, rb_fixed: torch.Tensor, rb_moving: torch.Tensor ):
		"""
		Create input features given the coordinates for the fixed and moving rigid bodies.
		rb_fixed -> [B, N, 37, 3]; rb_moving -> [B, N, 37, 3]
		"""
		if self.model_config.input_feats == "com":
			# [B, 3]
			feats = torch.mean( rb_moving, dim = ( 1, 2 ) )
		elif self.model_config.input_feats == "uvd":
			# Unit Vector-Distance -> [B, 3]
			com_fixed = torch.mean( rb_fixed, dim = ( 1, 2 ) )
			com_moving = torch.mean( rb_moving, dim = ( 1, 2 ) )

			com_vec = com_fixed - com_moving
			unit_vec = com_vec/torch.linalg.norm( com_vec )

			com_dist = torch.sqrt(
				torch.sum( ( com_fixed - com_moving )**2 )
			)
			feats = torch.cat( [unit_vec.reshape( -1 ), com_dist.reshape( -1 )] )
			# [B, 4]
			feats = feats.reshape( 1, -1 )
			# print( com_fixed.shape, "  ", com_moving.shape, "  ", unit_vec.shape, "  ", feats.shape )
			# exit()

		return feats


	def predict( self, out: Dict[str, torch.Tensor] ):
		"""
		Given the coordinates for the complex,
			Split the complex into rigid bodies.
			Keep one of the rigid bodies fixed (by default the 1st rigid body).
			Obtain features for each rigid body.
			Using the rigid body features as input to predict the rigid transformations.
			Apply the rigid transformation to obtain the new conformation.
		"""
		final_atom_positions = out.pop( "final_atom_positions" )

		final_atom_positions = final_atom_positions.to( self.device )
		rigid_bodies = get_rigid_body(
			final_atom_positions = final_atom_positions,
			asym_id = out["asym_id"],
			rigid_type = self.model_config.rigid_type
		)

		tmp_q, tmp_t = [], []
		rb_fixed = rigid_bodies[0]
		transformed_positions = [rb_fixed]
		for rb_moving in rigid_bodies[1:]:
			feats = self.create_input_feats( rb_fixed = rb_fixed, rb_moving = rb_moving )

			# [B, 4] and [B, 3]
			quat, trans = self.rigid( feats )
			R = quat_to_rotmat( quat = quat )

			Rt_rb = apply_transform(
				coords = rb_moving,
				R = R,
				trans = trans )
			transformed_positions.append( Rt_rb )

			# Keep track of rigid transformations.
			with torch.no_grad():
				tmp_q.append( quat.detach().cpu() )
				tmp_t.append( trans.detach().cpu() )
		transformed_positions = torch.cat( transformed_positions, dim = 1 )

		out["final_atom_positions"] = transformed_positions*out["final_atom_mask"].unsqueeze( -1 ).to( self.device )

		self.transformations["rotation"].append( torch.cat( tmp_q, dim = 0 ) )
		self.transformations["translation"].append( torch.cat( tmp_t, dim = 0 ) )

		return out


	def params( self ):
		"""
		Return a list of models for the optimizer.
		"""
		return [self.rigid]

################################################################################
################################################################################
class RandomPoseSampling():
	"""
	Randomly sample a rigid transformation - quaternion and translation.
	Apply a rigid transformation to sample conformations of the
		system that satisfy the data.
	"""
	def __init__( self, 
				model_config: mlc.ConfigDict,
				device: str ):
		# Model.__init__( self )
		self.model_config = model_config
		self.device = device

		self.transformations = {k:[] for k in ["rotation", "translation"]}


	def random_quaternion( self, batch_size: int ):
		"""
		Taken from:
			https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
		This yields a normalized quaternion uniformly distributed over the 3-sphere.
		"""
		u1 = torch.rand( batch_size, device = self.device )
		u2 = torch.rand( batch_size, device = self.device )
		u3 = torch.rand( batch_size, device = self.device )

		quat = torch.stack(
			[
			torch.sqrt( 1 - u1 )*torch.sin( 2*torch.pi*u2 ),
			torch.sqrt( 1 - u1 )*torch.cos( 2*torch.pi*u2 ),
			torch.sqrt( u1 )*torch.sin( 2*torch.pi*u3 ),
			torch.sqrt( u1 )*torch.cos( 2*torch.pi*u3 ),
			], dim = -1 )
		return quat


	def random_translation( self, batch_size: int ):
		"""
		Randomly sample a translation vector from a uniform distribution U(0, 1).
		"""
		trans = torch.rand( batch_size, 3, device = self.device )
		return trans


	def predict( self, out: Dict[str, torch.Tensor] ):
		"""
		Update the current positions by applying a Rigid Transformation.
		"""
		final_atom_positions = out.pop( "final_atom_positions" )
		final_atom_positions = final_atom_positions.to( self.device )

		rigid_bodies = get_rigid_body(
			final_atom_positions = final_atom_positions,
			asym_id = out["asym_id"],
			rigid_type = self.model_config.rigid_type
		)

		# [B, 4] and [B, 3]
		quat = self.random_quaternion( batch_size = len( rigid_bodies ) )
		trans = self.random_translation( batch_size = len( rigid_bodies ) )

		transformed_positions = []
		for rb, q, t in zip( rigid_bodies, quat, trans ):
			R = quat_to_rotmat( quat = q )

			Rt_rb = apply_transform(
				coords = rb.to( R.device ),
				R = R,
				trans = t )
			transformed_positions.append( Rt_rb )
		transformed_positions = torch.cat( transformed_positions, dim = 1 )

		out["final_atom_positions"] = transformed_positions*out["final_atom_mask"].unsqueeze( -1 ).to( self.device )

		# Keep track of rigid transformations.
		with torch.no_grad():
			self.transformations["rotation"].append( quat.detach().cpu() )
			self.transformations["translation"].append( trans.detach().cpu() )

		return out

