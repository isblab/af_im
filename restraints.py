import numpy as np
import torch
from torch import nn

from openfold.utils.rigid_utils import Rotation, Rigid

from typing import Dict, Optional



class XlRestraint():
	def __init__( self, config ):
		self.name = "xlr"
		self.config = config
		self.length_scale = 10.0
		self.eps = config.eps


	def get( out, batch ):
		if self.config.fape_xlr:
			return self.fape_xl_restraint( out, batch )
		elif self.config.simple_xlr:
			return self.simple_xl_restraint( out, batch )
		else:
			raise Exception( "At least one of the XL restraint types must be enabled..." )


	def fape_xl_restraint( out: Dict[str, torch.Tensor],
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

		# This step is not needed as pred_aff is already an object of Rigid().
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
		pred_dist_map = torch.sqrt( torch.sum( local_pred_pos**2, dim = -1 ) + eps )

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

		loss = torch.mean( loss )

		return loss


	def simple_xl_restraint( out: Dict[str, torch.Tensor],
							xl_res_mask: torch.tensor,
							xl_max_bound: float ):
		"""
		Calculate the cross-linking restraint loss as the mean squared deviation for the 
		predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
		Here, I am using the "final_atom_positions" for the restraint.
		"""
		# [B,N,37,3] --> For 2ayo: [1,480,37,3]
		pred_positions = out["final_atom_positions"]

		# Extracting Ca-coordinates.
		# [B,N,3] --> For 2ayo: [1,480,3]
		ca_pos = pred_positions[..., 1, :]
		diff = ca_pos.unsqueeze( 2 ) - ca_pos.unsqueeze( 1 )
		D = torch.sqrt( 
						torch.sum( ( diff )**2, dim = -1 ) + eps
						)
		# Adjust the length scales.
		D = D / self.length_scale
		xl_max_bound = xl_max_bound / self.length_scale
		
		# Consider only the cross-linked residues.
		D = D*xl_res_mask

		# Identify Xl violations.
		viols_mask = D > xl_max_bound
		if viols_mask.any():
			loss = ( D[viols_mask] - xl_max_bound )**2
		else:
			# loss = torch.tensor( [0.0], device = D.device, requires_grad = True )
			loss = D*0 
		# Aggregate the loss (sum or mean).
		loss = torch.mean( loss )
		return loss
