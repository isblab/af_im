import numpy as np
import torch
from torch import nn

from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.loss import softmax_cross_entropy

from typing import Dict, Optional



class XlRestraint():
	def __init__( self, config ):
		self.name = "xlr"
		self.config = config
		self.length_scale = 10.0
		self.eps = config.eps


	def get( self, out, restraint_feature ):
		if self.config.type == "fape_xlr":
			return lambda: self.fape_xl_restraint( out, restraint_feature["xl_res_mask"],
													restraint_feature["xl_max_bound"] )
		elif self.config.type == "simple_xlr":
			return lambda: self.simple_xl_restraint( out, restraint_feature["xl_res_mask"],
													restraint_feature["xl_max_bound"] )
		# Same as simple_xlr, just modular.
		elif self.config.type == "mse_xlr":
			return lambda: self.mse_xl_restraint( out, restraint_feature["xl_res_mask"],
													restraint_feature["xl_max_bound"] )
		# Same as simple_xlr, just modular.
		elif self.config.type == "rmse_xlr":
			return lambda: self.rmse_xl_restraint( out, restraint_feature["xl_res_mask"],
													restraint_feature["xl_max_bound"] )
		elif self.config.type == "disto_xlr":
			return lambda: self.disto_xl_restraint( out, **restraint_feature )
		else:
			raise Exception( "At least one of the XL restraint types must be enabled..." )


	def frames_to_dist_map( self, out: Dict[str, torch.Tensor] ):
		"""
		Given the predicted rigif frames for the system, compute a distance map.
		"""
		# Get the predicted affine matrices.
		# traj --> [R,N,N,A,F] e.g. For 2ayo, [8,1,480,4,4]
		traj = out["sm"]["frames"]

		# Legacy: not required but keeping OpenFold implementation.
		# Need to check if the traj belongs to 4*4 matrix or a tensor_7
		if traj.shape[-1] == 7:
			pred_frames = Rigid.from_tensor_7( traj )
		elif traj.shape[-1] == 4:
			# OpenFold uses a rotation matrix.
			# pred_frames here is an object of class Rigid() which wraps a rotation and translation.
			pred_aff = Rigid.from_tensor_4x4( traj )

		# This step is not needed as pred_aff is already an object of Rigid(). - Kartik -
		# pred_aff = Rigid(
		# 	Rotation( 
		# 		rot_mats = pred_aff.get_rots().get_rot_mats(), 
		# 		quats = None ),
		# 		pred_aff.get_trans(),
		# 	)

		# The translation vector of the affine matrix is used as positions.
		pred_positions = pred_aff.get_trans()

		# Apply the affine transformation to obtain the predicted positions in local frame.
	    # This directly gives us the pairwise distances for each residue along xyz.
	    # local_pred_pos --> [R,B,N,N,X] or [8,1,480,480,3]
		local_pred_pos = pred_aff.invert()[..., None].apply(
			pred_positions[..., None, :, :],
		)
		
		# Calculate the pairwise Euclidean distance matrix.
		# 	eps: Krde karam ke dil ye chain paayega
		pred_dist_map = torch.sqrt( torch.sum( local_pred_pos**2, dim = -1 ) + self.eps )

		return pred_dist_map


	def fape_xl_restraint( self, out: Dict[str, torch.Tensor],
							xl_res_mask: torch.tensor,
							xl_max_bound: float ):
		"""
		Calculate the cross-linking restraint loss as the mean squared deviation for the 
		predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
		This is the FAPE implementation for the restraint.
		This restraint is implemented as a max bound restraint since XLMS 
			povides a max bound for the distance between XL'd residue pairs.

		XL_restraint = ( D - max_bound )*82 if D > max_bound else 0
		
		Note: R - SM recycling dim; B - batch dim (1); N - no. of residues;
				A - no. of atoms (14 or 37); X - coords dim (3); F - frame dim (4).
		Taking example of 2ayo; N = 480.
		"""
		# # Get the predicted affine matrices.
		# # traj --> [R,N,N,A,F] e.g. For 2ayo, [8,1,480,4,4]
		# traj = out["sm"]["frames"]

		# # Legacy: not required but keeping OpenFold implementation.
		# # Need to check if the traj belongs to 4*4 matrix or a tensor_7
		# if traj.shape[-1] == 7:
		# 	pred_frames = Rigid.from_tensor_7( traj )
		# elif traj.shape[-1] == 4:
		# 	# OpenFold uses a rotation matrix.
		# 	# pred_frames here is an object of class Rigid() which wraps a rotation and translation.
		# 	pred_aff = Rigid.from_tensor_4x4( traj )

		# # This step is not needed as pred_aff is already an object of Rigid(). - Kartik -
		# # pred_aff = Rigid(
		# # 	Rotation( 
		# # 		rot_mats = pred_aff.get_rots().get_rot_mats(), 
		# # 		quats = None ),
		# # 		pred_aff.get_trans(),
		# # 	)

		# # The translation vector of the affine matrix is used as positions.
		# pred_positions = pred_aff.get_trans()

		# # Apply the affine transformation to obtain the predicted positions in local frame.
	    # # This directly gives us the pairwise distances for each residue along xyz.
	    # # local_pred_pos --> [R,B,N,N,X] or [8,1,480,480,3]
		# local_pred_pos = pred_aff.invert()[..., None].apply(
		# 	pred_positions[..., None, :, :],
		# )
		
		# # Calculate the pairwise Euclidean distance matrix.
		# # 	eps: Krde karam ke dil ye chain paayega
		# pred_dist_map = torch.sqrt( torch.sum( local_pred_pos**2, dim = -1 ) + self.eps )

		pred_dist_map = self.frames_to_dist_map( out )

		# Adjust the length scales.
		pred_dist_map = pred_dist_map / self.length_scale
		xl_max_bound = xl_max_bound / self.length_scale

		# Apply cross-linked residue mask.
		pred_dist_map = pred_dist_map * xl_res_mask

		# For all XL violations, calculate the squared difference from the max_bound XL distance.
		viols_mask = pred_dist_map > xl_max_bound

		if viols_mask.any():
			loss = ( pred_dist_map[viols_mask] - xl_max_bound )**2
		else:
			loss = pred_dist_map*0 

		# Normalizing by the total no. of cross-linked residue pairs.
		loss = torch.sum( loss )/ (self.eps + torch.sum( xl_res_mask ) )
		return loss


	def final_pred_to_dist_map( self, final_atom_pos: torch.Tensor
								) -> torch.Tensor:
		"""
		Compute a distance map from the final_atom_positions.
		Using only Ca-coordinates for distance map calculation.

		Input:
		----------
		final_atom_pos --> coordinates in atom37 representtaion.
						[B,N,37,3] --> For 2ayo: [1,480,37,3]

		Returns:
		----------
		D --> Ca-distance map [B,N,N].
		"""
		# Extracting Ca-coordinates - index 1.
		# [B,N,3] --> For 2ayo: [1,480,3]
		ca_pos = final_atom_pos[..., 1, :]
		diff = ca_pos.unsqueeze( 2 ) - ca_pos.unsqueeze( 1 )
		D = torch.sqrt( 
						torch.sum( ( diff )**2, dim = -1 ) + self.eps
						)

		# Adjust the length scales.
		scaled_D = D / self.length_scale
		return scaled_D


	def get_violations_mask( self, D: Dict[str, torch.Tensor],
							xl_res_mask: torch.tensor,
							scaled_xl_max_bound: float ) -> torch.Tensor:
		"""
		Adjust the length sacales for the distance map and xl_max_bound.
		Identify XL violations.

		Input:
		----------
		D --> Ca-distance map [B,N,N].
		xl_res_mask --> binary mask indicating cross-linked residue pairs.
		xl_max_bound --> max distance between the cross-licked residues 
						scaled by self.length_scale.

		Returns:
		----------
		viols_mask --> bool mask indicating violated XLs.
		"""
		# Consider only the cross-linked residues.
		D = D*xl_res_mask

		# Identify Xl violations.
		viols_mask = D > xl_max_bound
		return viols_mask


	def mse_xl_restraint( self, out: Dict[str, torch.Tensor],
							xl_res_mask: torch.tensor,
							xl_max_bound: float ) -> torch.Tensor:
		"""
		Calculate the cross-linking restraint loss as the mean squared error (MSE) for the 
		predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
		Here, I am using the "final_atom_positions" for the restraint.

		Input:
		----------
		out --> output dict from the model.
		xl_res_mask --> binary mask indicating cross-linked residue pairs.
		xl_max_bound --> max distance between the cross-licked residues.

		Returns:
		----------
		loss --> mean squared loss.
		"""
		D = self.final_pred_to_dist_map( out["final_atom_positions"] )

		# Adjust the length scales.
		scaled_xl_max_bound = xl_max_bound / self.length_scale
		
		viols_mask = self.get_violations_mask( D, xl_res_mask, scaled_xl_max_bound )

		# Loss is computed only for the violated XLs.
		if viols_mask.any():
			squared_diff = ( D[viols_mask] - scaled_xl_max_bound )**2
		else:
			squared_diff = D*0 
		
		# Normalizing by the total no. of cross-linked residue pairs.
		denom = self.eps + torch.sum( xl_res_mask )
		mse = torch.sum( squared_diff )/ denom
		return mse


	def rmse_xl_restraint( self, out: Dict[str, torch.Tensor],
							xl_res_mask: torch.tensor,
							xl_max_bound: float ) -> torch.Tensor:
		"""
		Using root mean squared error (RMSE) as the XL restraint.
			Read doc string for mse_xl_restraint.

		Input:
		----------
		out --> output dict from the model.
		xl_res_mask --> binary mask indicating cross-linked residue pairs.
		xl_max_bound --> max distance between the cross-licked residues.

		Returns:
		----------
		loss --> root mean squared loss.
		"""
		mse = self.mse_xl_restraint( out, xl_res_mask, xl_max_bound )

		rmse = torch.sqrt( mse )

		return rmse


	def simple_xl_restraint( self, out: Dict[str, torch.Tensor],
							xl_res_mask: torch.tensor,
							xl_max_bound: float ) -> torch.Tensor:
		"""
		Calculate the cross-linking restraint loss as the mean squared error (MSE) for the 
		predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
		Here, I am using the "final_atom_positions" for the restraint.

		Input:
		----------
		out --> output dict from the model.
		xl_res_mask --> binary mask indicating cross-linked residue pairs.
		xl_max_bound --> max distance between the cross-licked residues.

		Returns:
		----------
		loss --> mean squared loss.
		"""
		# [B,N,37,3] --> For 2ayo: [1,480,37,3]
		pred_positions = out["final_atom_positions"]

		# Extracting Ca-coordinates.
		# [B,N,3] --> For 2ayo: [1,480,3]
		ca_pos = pred_positions[..., 1, :]
		diff = ca_pos.unsqueeze( 2 ) - ca_pos.unsqueeze( 1 )
		D = torch.sqrt( 
						torch.sum( ( diff )**2, dim = -1 ) + self.eps
						)
		
		# Adjust the length scales.
		D = D / self.length_scale
		xl_max_bound = xl_max_bound / self.length_scale
		
		# Consider only the cross-linked residues.
		D = D*xl_res_mask

		# Identify Xl violations.
		viols_mask = D > xl_max_bound
		if viols_mask.any():
			squared_diff = ( D[viols_mask] - xl_max_bound )**2
		else:
			# loss = torch.tensor( [0.0], device = D.device, requires_grad = True )
			squared_diff = D*0 
		
		# Normalizing by the total no. of cross-linked residue pairs.
		denom = self.eps + torch.sum( xl_res_mask )
		mse = torch.sum( squared_diff )/ denom
		return mse



	def disto_xl_restraint( self, out: Dict[str, torch.Tensor],
								xl_res_mask: torch.Tensor,
								xl_max_bound: float,
								gt_distogram: torch.Tensor,
								cb_corr: float = 3.0 ):
		"""
		Calculate the cross-linking restraint as the softmax cross entropy loss between
		the predicted and ground truth distogram.
		Loss implementation is adapted from the OpenFold distogram_loss().
		"""
		assert torch.all( ( gt_distogram == 0 ) | ( gt_distogram == 1 ) ), "gt_distogram should be binary"
		# Distogram is based on Ca distances. Correcting for Cb.
		# 	Ca-Cb bond length = 1.54A so subtracting ~2*3.54.
		xl_max_bound = xl_max_bound - cb_corr
		
		# Get predicted distogram logits.
		logits = out["distogram_logits"]
		
		errors = softmax_cross_entropy(
		    logits, gt_distogram )

		# square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

		# FP16-friendly sum.
		# Here xl_res_mask is equivalent to square_mask in OpenFold implementation.
		denom = self.eps + torch.sum( xl_res_mask, dim = ( -1, -2 ) )
		mean = errors * xl_res_mask
		# Penalizing only the violated cross-links.
		viols = errors > xl_max_bound
		mean = mean[viols]

		mean = torch.sum( mean, dim = -1 )
		mean = mean / denom[..., None]
		mean = torch.sum( mean, dim = -1 )

		# Average over the batch dimensions
		mean = torch.mean(mean)

		return mean