"""
Contains modules for modifying the Evoformer single and pair
	representations at inference time.
"""
from typing import List, Tuple, Dict, Any
import ml_collections as mlc

import torch
from torch import nn

import openfold.np.residue_constants as rc

from models.base_model import Model
from models.rigid_body import get_rigid_body


def axis_angle_to_matrix(axis_angle: torch.Tensor ) -> torch.Tensor:#, fast: bool = False) -> torch.Tensor:
	"""
	Taken from pytorch3d.
	Convert rotations given as axis/angle to rotation matrices.

	Args:
		axis_angle: Rotations given as a vector in axis angle form,
			as a tensor of shape (..., 3), where the magnitude is
			the angle turned anticlockwise in radians around the
			vector's direction.
		fast: Whether to use the new faster implementation (based on the
			Rodrigues formula) instead of the original implementation (which
			first converted to a quaternion and then back to a rotation matrix).

	Returns:
		Rotation matrices as tensor of shape (..., 3, 3).
	"""
	# if not fast:
	# 	return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

	shape = axis_angle.shape
	device, dtype = axis_angle.device, axis_angle.dtype

	angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

	rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
	zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
	cross_product_matrix = torch.stack(
		[zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
	).view(shape + (3,))
	cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

	identity = torch.eye(3, dtype=dtype, device=device)
	angles_sqrd = angles * angles
	angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
	return (
		identity.expand(cross_product_matrix.shape)
		+ torch.sinc(angles / torch.pi) * cross_product_matrix
		+ ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
	)


def quat_to_rotmat( quat: torch.Tensor ):
	"""
	Taken from openfold/utils/geometry/rotation_matrix.from_quaternion
	quat -> [B, 4]
	"""
	w, x, y, z = quat.unbind( -1 )

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
	Apply a rigid transformation (rotation + translation) to
		a batch of atomic coordinates.
	The input coordinates are temporarily centered to allow
		local rotation and avoid rotation around the global axis.
	Raises RuntimeError for an invalid rotation
		matrix (determinant approx != 1.0).

	Inputs
	----------
	coords: [B, N, A, 3]; input coordinates where
		B -> batch size.
		N -> no. of residues.
		A -> atoms per residue.
		3 -> xyz coordinates.
	R: [B, 3, 3]; rotation matrix.
	trans: [B, 3]; translation vector.

	Returns
	----------
	Transformed coordinates of shape [B, N, A, 3], after:
		1. Centering the coordinates,
		2. Applying the rotation,
		3. Adding back the original center and translation vector.
	"""
	with torch.no_grad():
		det_R = torch.det( R[0] )
		if ( 1.0 - det_R ).abs() > 1e-3:
			raise RuntimeError( f"Rotation matrix seterminant ({det_R}) != 1.0..." )
		# del det_R
	# [N, N, A, 3] -> [B, 1, 1, 3]
	com = coords.mean( dim = ( 1,2 ), keepdim = True )
	coords_centered = coords - com
	coords_rot = torch.einsum( "bnac,bjc->bnaj", coords_centered, R )
	# [B, 3] -> [B, 1, 1, 3]
	trans = trans[:, None, None, :]
	# [B, N, A, 3]
	coords_new = coords_rot + com + trans

	return coords_new


class RigidTransformation( nn.Module ):
	"""
	Predict a rigid transformation ( rotation + translation ).
	Rotation can be parameterized:
		Axis-angle (3D vector)
		Quaternion (4D vector)
	Scalar features are aggregated using feature-wise attention across 
		interacting bodies.
	Vector features are aggregated using a PointNet-style max pooling.

	Inputs
	----------
	trans_step_size: scaling factor for predicted translation.
	use_axis_angle: if True, predict axis-angle else predict quaternion.
	device: device string for model placement.
	"""
	def __init__( self,
		rot_step_size: float,
		trans_step_size: float,
		use_axis_angle: bool,
		device: str,
		):
		super().__init__()
		self.h_s1, self.h_s2 = 32, 16
		self.h_v1, self.h_v2 = 18, 9
		self.h_sv1, self.h_sv2 = 32, 16
		self.rot_step_size = rot_step_size
		self.trans_step_size = trans_step_size
		self.use_axis_angle = use_axis_angle

		self.scalar_embedder = nn.Sequential(
			nn.Linear( in_features = 16, out_features = self.h_s1, bias = True ),
			nn.ReLU(),
			nn.Linear( in_features = self.h_s1, out_features = self.h_s2, bias = True ),
			nn.LayerNorm( self.h_s2 ),
			nn.ReLU()
			)
		self.vector_embedder = nn.Sequential(
			nn.Linear( in_features = 9, out_features = self.h_v1, bias = True ),
			nn.ReLU(),
			nn.Linear( in_features = self.h_v1, out_features = self.h_v2, bias = True ),
			nn.LayerNorm( self.h_v2 ),
			nn.ReLU()
			)
		self.global_embedder = nn.Sequential(
			nn.Linear( in_features = self.h_s2+self.h_v2, out_features = self.h_sv1, bias = True ),
			nn.LayerNorm( self.h_sv1 ),
			nn.ReLU(),
			nn.Linear( in_features = self.h_sv1, out_features = self.h_sv2, bias = True ),
			nn.LayerNorm( self.h_sv2 ),
			nn.ReLU()
			)
		self.transition_layer = nn.Sequential(
			nn.Linear( in_features = self.h_sv2, out_features = self.h_sv1, bias = True ),
			nn.LayerNorm( self.h_sv1 ),
			nn.ReLU()
			)
		# Use an axis-angle representation for rotations.
		if self.use_axis_angle:
			self.omega = nn.Sequential(
				nn.Linear( in_features = self.h_sv1, out_features = 3, bias = True ),
				nn.Tanh()
			)
		else:
			# Predict a quaternion.
			self.quaternion = nn.Sequential(
				nn.Linear( in_features = self.h_sv1, out_features = 4, bias = True )
			)
		self.translation = nn.Sequential(
			nn.Linear( in_features = self.h_sv1, out_features = 3, bias = True ),
			nn.Tanh()
		)


	def forward( self,
		scalar_feats: torch.Tensor,
		vector_feats: torch.Tensor
		) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Predict a rigid transformation (rotation + translation) given
			the scalar and vector features

		Inputs
		----------
		scalar_feats: [B, I, C_h]; tensor containing  scalar features
			for each interacting body.
			C_h -> bin size used for histogram creation.
		vector_feats:> [B, I, 9]; pairwise vectors between
			surface residues.

		Returns
		----------
		Rigid transformation comprising:
			Rotation:
				- [B, 3] axis-angle if use_axis_angle = True.
				- [B, 4] unit quaternion if use_axis_angle = False.
			Translation: [B, 3]; translation vector.
		"""
		# [B, I, H_s2]
		s_emb = self.scalar_embedder( scalar_feats )
		# [B, I, H_v2]
		v_emb = self.vector_embedder( vector_feats )

		# [B, I, H_s2+H_v2]
		sv = torch.cat( [s_emb, v_emb], dim = -1 )
		# [B, H_sv2]
		global_feats = self.global_embedder( sv )

		# Aggregate features across the interacting bodies.
		# [B, I, H_sv2] -> [B, I]
		# score = torch.mean( global_feats, dim = -1 )
		# [B, I, 1]
		w = nn.functional.softmax( global_feats, dim = 1 ) #.unsqueeze( -1 )
		# [B, I, H_sv2] -> [B, H_sv2]
		global_feats = ( w*global_feats ).sum( dim = 1 )

		# [B, H_sv1]
		global_feats = self.transition_layer( global_feats )

		# [B, 3]
		trans_raw = self.translation( global_feats )
		trans = self.trans_step_size*trans_raw

		if self.use_axis_angle:
			# [B, 3]
			omega = self.rot_step_size*self.omega( global_feats )
			return omega, trans
		else:
			# Quaternion -> [B, 4]
			quat_raw = self.quaternion( global_feats )
			if torch.isnan( quat_raw ).any() or torch.isinf( quat_raw ).any():
				print( "Warning: quat contains NaN or Inf before normalization" )
			# Normalize the quaternion.
			quat_norm = torch.linalg.norm( quat_raw, dim = -1, keepdim = True ).clamp( min = 1e-8 )
			# [B, 4]
			quat = quat_raw/quat_norm
			return quat, trans

################################################################################
################################################################################
class PoseSampling( Model ):
	"""
    Module for sampling new conformations of a multi-body system by predicting
    	rigid transformations for each rigid body.
    For each rigid body, interaction features with the remaining bodies are
    	constructed using only surface-exposed Cα atoms.
	These features are used to predict a rotation and translation.
    The predicted transformation can be parameterized either as:
        - Axis-angle rotation + translation
        - Quaternion rotation + translation
    The module stores the predicted transformations for later inspection.
	"""
	def __init__( self, 
		model_config: mlc.ConfigDict,
		device: str ):
		super().__init__()
		self.model_config = model_config
		self.sequential_update = model_config.sequential_update
		self.device = device

		self.rigid = RigidTransformation(
			# clamp_translation = self.model_config.clamp_translation,
			rot_step_size = self.model_config.rot_step_size,
			trans_step_size = self.model_config.trans_step_size,
			device = device,
			use_axis_angle = self.model_config.use_axis_angle )

		self.transformations = {k:[] for k in ["rotation", "translation"]}


	def create_input_feats( self,
		rigid_bodies: Dict[int, torch.Tensor],
		surface_residue_mask: Dict[int, torch.Tensor],
		xl_res_mask: torch.Tensor,
		asym_id: torch.Tensor,
		mb_id: int,
		hist_bins: int,
		hist_range: List[float],
		) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Construct fixed size input features for a selected rigid body.
		Coordinates for all rigid bodies are used as input.
		All coordinates are centered using the COM of the entire complex to
			remove global translation effects.
		For the selected rigid body "mb_id", interaction features are computed
			with respect to every other rigid body.
		Two types of features are produced:
			1. Scalar features
				Select the surface-exposed Cα atoms for each rigid body.
				Compute the distance map (D_ij) for the i-th and j-th rigid bodies.
					Convert to a histogram with specified no. of bins.
			2. Vector features
				COM vector from the i-th to the j-th rigid body.
					Using the centered coordinates for the entire chain here.
				Mean displacement vector for cross-linked residue pairs,
					pointing towards the j-th rigid body.
					Using the centered coordinates for the entire chain here.
				Mean clash vector for clashing/near-clashing residue pairs,
					pointing towards the i-th rigid body.
					Using the surface exposed residue coords.

		Inputs
		----------
		rigid_bodies: list of rigid body coordinate tensors.
			Each tensor has shape: [B, N_i, A, 3] where
				B: batch size
				N_i: no. of residues in body i
				A: no. of atoms per residue
		surface_residue_mask: [B, N_i]; boolean masks identifying
			surface-exposed residues for each rigid body.
		xl_res_mask: [N, N]; binary mask for cross-linked (XL'd) residues.
		asym_id: numeric chain ID as created in OpenFold.
		mb_id: rigid body ID for which the features are computed.
		hist_bins: no. of bins for creating the histogram of interface distances.
		hist_range: [min, max] distance for creating the histogram of interface distances.
		B = 1; N -> no. of residues in the system; N_i/N_j -> no. of residues in i/j-th rigid body.

		Returns
		----------
		scalar_feats: histogram ofinterface distances; [B, I, hist_bins]
			where I is the number of interacting rigid bodies.
		vector_feats: COM vector, mean Xl displacement vector and weighted clash vector stacked;
			[B, I, 9].
		"""
		ca_idx = rc.atom_order["CA"]
		# Get COM of the complex.
		stacked_coords = torch.cat( list( rigid_bodies.values() ), dim = 1 )
		# [B, 1, 3]
		com = torch.mean( stacked_coords[: , :, ca_idx, :], dim = 1, keepdim = True )

		centered_coords, surface_coords = {}, {}
		# for i in range( len( rigid_bodies ) ):
		for k in rigid_bodies:
			rb = rigid_bodies[k]
			# [B, N, 37, 3] -> [B, N, 3]
			ca_coord = rb[:, :, ca_idx, :]
			centered_coords[k] = ca_coord - com
			mask = surface_residue_mask[k]
			# Select only the surface exposed residues.
			surface_coords[k] = centered_coords[k][:, mask[0, :].bool(), :]

		# Surface coords for the i-th rigid body.
		coords_i = surface_coords[mb_id]
		# [B, 3]; COM for i=th rigid body
		com_i = torch.mean( centered_coords[mb_id], dim = 1 )

		# Map system-level (global) indices to rigid body level (local) indices.
		global_to_local_maps = {}
		for k in rigid_bodies:
			# [N_k]
			idx_k = torch.where( asym_id[0] ==  k )[0]
			# The '-' sign is a placeholder to exclude indices not part of the chain.
			map_k = -torch.ones( asym_id.shape[1], dtype = torch.long, device = asym_id.device )
			# Now we fill in the indices for i-th rigid body.
			map_k[idx_k] = torch.arange( len( idx_k ), device = asym_id.device )
			global_to_local_maps[k] = map_k

		scalar_feats, vector_feats = [], []
		for j in surface_coords:
			if j == mb_id:
				continue
			coords_j = surface_coords[j]

			# Scalar features ----------
			# [B, N_i, N_j]; distance map for all surface residues.
			D_ij = torch.cdist( coords_i, coords_j )
			# This is brittle for B>1; but for us B=1 always.
			# [hist_bins]
			hist = torch.histc(
				D_ij.flatten( start_dim = -2 ), bins = hist_bins,
				min = hist_range[0], max = hist_range[1] )
			# [B, hist_bins]
			hist = ( hist/hist.sum().clamp_min( 1e-8 ) ).unsqueeze( 0 )
			scalar_feats.append( hist )

			# Vector features ----------
			# COM vector
			## --------------------
			# [B, 3]
			com_j = torch.mean( centered_coords[j], dim = 1 )
			# [B, 3]
			com_ij = com_j - com_i

			# Mean displacement vector for XL'd residue pairs; pointing towards the j-th rigid body.
			## --------------------
			# [B, N_i, N_j]; mask to get XL'd residues for i,j-rigid body pair.
			ij_mask = (
				(asym_id == mb_id)[:, :, None] &
				(asym_id == j)[:, None, :]
			)
			# print( torch.sum( ij_mask ), "  ", centered_coords[mb_id].shape[1]*centered_coords[j].shape[1] )
			# [B, N_i, N_j]
			ij_xls = ij_mask&xl_res_mask.bool()

			if ij_xls.any():
				# Get the system-level indices for cross-linked residues.
				idx_xl_i, idx_xl_j = torch.where( ij_xls[0] )

				# Rigid body level indices for XL'd residues.
				idx_xl_i_local = global_to_local_maps[mb_id][idx_xl_i]
				idx_xl_j_local = global_to_local_maps[j][idx_xl_j]

				# Now select the coords for the XL'd residues.
				xl_i = centered_coords[mb_id][:, idx_xl_i_local, :]
				xl_j = centered_coords[j][:, idx_xl_j_local, :]

				# [B, K, 3]; K -> no. of XL'd pairs.
				xl_ij = torch.mean( xl_j - xl_i, dim = 1 )

			else:
				# Create a 0-tensor if no XL'd residues exist for the i-j pair.
				xl_ij = torch.zeros_like( com_ij )

			# Weighted displacement vector for clashing residues.
			## --------------------
			# [B, N_i, N_j, 3];  Pairwise vectors pointing from the j-th to i-th rigid body.
			V_ij = coords_i[:, :, None] - coords_j[:, None, :]
			# [B, N_i*N_j, 3]
			flat_V_ij = V_ij.flatten( start_dim = -3, end_dim = -2 )
			# [B, 1]; weigh the closer residue pairs higher.
			w = nn.functional.softmin( D_ij.flatten( start_dim = -2 ), dim = -1 ).unsqueeze( -1 )
			# [B, 3]
			clash_ij = torch.sum( w*flat_V_ij, dim = -2 )
			vector_feats.append(
				torch.cat( [com_ij, xl_ij, clash_ij], dim = -1 )
			)
		# [B, I, hist_bins]; I -> no. of interacting rigid bodies.
		scalar_feats = torch.stack( scalar_feats, dim = 1 )
		# [B, I, 9]; I -> no. of interacting rigid bodies.
		vector_feats = torch.stack( vector_feats, dim = 1 )
		# print( scalar_feats.shape, "  ", vector_feats.shape )

		return scalar_feats, vector_feats


	def predict( self,
		out: Dict[str, torch.Tensor],
		batch: Dict[str, Any],
		) -> Dict[str, torch.Tensor]:
		"""
		Predict and apply rigid transformations to all rigid bodies in the system.
		Given the coordinates for the complex,
			Split the complex into rigid bodies.
			Obtain input features for the i-th rigid body.
			Using the rigid body features as input to predict the rigid transformations.
			Apply the rigid transformation to obtain the new conformation.

		Inputs
		----------
		out: dict containing OpenFold outputs, including:
				final_atom_positions: [B, N, A, 3] -> current atomic coordinates.
				final_atom_mask: [B, N, A] -> mask indicating valid atoms.
				asym_id: [N] -> chain or rigid-body identifier for each residue.
		surface_residue_mask: surface residue masks for each rigid body.
				Each tensor has shape: [N_i]

		Returns
		----------
		out: updated dictionary where `final_atom_positions` contains the
			transformed coordinates after applying rigid-body motions.
		"""
		final_atom_positions = out.pop( "final_atom_positions" )

		rigid_bodies = get_rigid_body(
			final_atom_positions = final_atom_positions,
			asym_id = out["asym_id"],
			rigid_type = self.model_config.rigid_type
		)

		tmp_q, tmp_t = [], []
		transformed_positions = []
		for i, coords_i in rigid_bodies.items():
			with torch.no_grad():
				scalar_feats, vector_feats = self.create_input_feats(
					rigid_bodies = rigid_bodies,
					surface_residue_mask = batch["surface_residue_mask"],
					xl_res_mask = batch["xl_restraint"]["xl_res_mask"],
					asym_id = batch["asym_id"],
					mb_id = i,
					hist_bins = self.model_config.hist_bins,
					hist_range = self.model_config.hist_range
				)

			# Predict axis_angle.
			if self.model_config.use_axis_angle:
				# [B, 3] and [B, 3]
				omega, trans = self.rigid(
					scalar_feats = scalar_feats,
					vector_feats = vector_feats )
				R = axis_angle_to_matrix( axis_angle = omega )
			# Predict quaternion.
			else:
				# [B, 4] and [B, 3]
				quat, trans = self.rigid(
					scalar_feats = scalar_feats,
					vector_feats = vector_feats )
				R = quat_to_rotmat( quat = quat )

			Rt_rb = apply_transform(
				coords = coords_i,
				R = R,
				trans = trans )
			transformed_positions.append( Rt_rb )

			# Update the rigid body post-move.
			if self.sequential_update:
				rigid_bodies[i] = Rt_rb

			# Keep track of rigid transformations.
			with torch.no_grad():
				if self.model_config.use_axis_angle:
					tmp_q.append( omega.detach().cpu() )
				else:
					tmp_q.append( quat.detach().cpu() )
				tmp_t.append( trans.detach().cpu() )
		transformed_positions = torch.cat( transformed_positions, dim = 1 )

		out["final_atom_positions"] = transformed_positions*out["final_atom_mask"].unsqueeze( -1 ).to( self.device )

		self.transformations["rotation"].append( torch.cat( tmp_q, dim = 0 ) )
		self.transformations["translation"].append( torch.cat( tmp_t, dim = 0 ) )

		return out


	def params( self ):
		"""
		Return the list of trainable model modules.
		This method returns the neural modules whose parameters should be
			optimized.
		Parameter extraction is handled externally.
		"""
		return [self.rigid]

################################################################################
################################################################################
class RandomPoseSampling():
	"""
	Randomly sample a rigid transformation - quaternion and translation.
	Apply a rigid transformation to sample conformations of the
		system that satisfies the data.
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

