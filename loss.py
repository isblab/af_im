import numpy as np
import torch
from torch import nn

from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.loss import ( find_structural_violations, 
								compute_renamed_ground_truth,
								fape_loss,
								supervised_chi_loss,
								violation_loss,
								chain_center_of_mass_loss
								)

from typing import Dict, Optional


def xl_restraint( out: Dict[str, torch.Tensor],
				xl_tgt_mask: torch.tensor, 
				xl_res_mask: torch.tensor, 
				length_scale: Optional[float] = 10.0,
				max_bound_dist: Optional[float] = 35.0,
				lambda_: Optional[float] = 0.5,
				eps: Optional[float] = 1e-8
				):
	"""
	Calculate the cross-linking restraint loss as the mean squared deviation for the 
	predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
	This is the FAPE implementation for the restraint.
	This restraint is implemented as a max bound restraint since XLMS 
		povides a max bound for the distance between XL'd residue pairs.

	XL_restraint = ( D - max_bound )*82 if D > max_bound else 0
	
	Note: R - SM recycling dim; B - batch dim (1); N - no. of residues;
			A - no. of atoms (14, 37); X - coords dim (3); F - frame dim (4).
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

	# [B,N,N] --> [1,480,480]
	xl_tgt_mask = xl_tgt_mask.unsqueeze( 0 )
	xl_res_mask = xl_res_mask.unsqueeze( 0 )

	# Adjust the length scales.
	pred_dist_map = pred_dist_map / length_scale
	xl_tgt_mask = xl_tgt_mask / length_scale

	# Apply cross-linked residue mask.
	pred_dist_map = pred_dist_map * xl_res_mask

	# For all XL violations, calculate the squared difference from the max_bound XL distance.
	viols_mask = pred_dist_map > max_bound_dist/length_scale
	xl_viols = ( pred_dist_map[viols_mask] - max_bound_dist )**2

	loss = torch.mean( lambda_ * xl_viols + eps )

	return loss


# def xl_restraint( out: Dict[str, torch.Tensor],
# 				xl_tgt_mask: torch.tensor, 
# 				xl_res_mask: torch.tensor, 
# 				length_scale: Optional[float] = 10.0,
# 				max_bound_dist: Optional[float] = 35.0,
# 				lambda_: Optional[float] = 0.5,
# 				eps: Optional[float] = 1e-8
# 				):
# 	"""
# 	Calculate the cross-linking restraint loss as the mean squared deviation for the 
# 	predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
# 	Here, I am using the "final_atom_positions" for the restraints.
# 	"""
# 	pred_positions = out["final_atom_positions"]
	
# 	xl_tgt_mask = xl_tgt_mask.unsqueeze( 0 ) / length_scale
# 	xl_res_mask = xl_res_mask.unsqueeze( 0 )

# 	ca_pos = pred_positions[..., 1, :]
# 	diff = ca_pos.unsqueeze( 2 ) - ca_pos.unsqueeze( 1 )
# 	print( diff.shape )
# 	D = torch.sqrt( 
# 					torch.sum( ( diff )**2, dim = -1 ) + eps
# 					)
# 	D = D / length_scale
# 	D = D*xl_res_mask

# 	# Compute the loss only for entries where the mask is True
# 	loss = lambda_ * ( D - xl_tgt_mask )**2

# 	# Aggregate the loss (sum or mean).
# 	loss = torch.mean( loss )
# 	return loss



class LossFunction( nn.Module ):
	def __init__( self, config ):
		print( config.keys() )
		self.config = config

	def forward( self, out, batch ):
		if "violation" not in out.keys():
			out["violation"] = find_structural_violations(
				batch,
				out["sm"]["positions"][-1],
				**self.config.violation,
			)

		if "renamed_atom14_gt_positions" not in out.keys():
			batch.update(
				compute_renamed_ground_truth(
					batch,
					out["sm"]["positions"][-1],
				)
			)

		loss_fns = {
			"fape": lambda: fape_loss(
				out,
				batch,
				self.config.fape,
			),
			"supervised_chi": lambda: supervised_chi_loss(
								out["sm"]["angles"],
								out["sm"]["unnormalized_angles"],
								**{**batch, **self.config.supervised_chi},
			),
			"violation": lambda: violation_loss(
						out["violation"],
						**{**batch, **self.config.violation},
			),
		}
		if self.config.chain_center_of_mass.enabled:
			loss_fns["chain_center_of_mass"] = lambda: chain_center_of_mass_loss(
								all_atom_pred_pos = out["final_atom_positions"],
								**{**batch, **self.config.chain_center_of_mass},
			)

		loss_fns["xlr"] = lambda: xl_restraint( 
							out = out, 
							**batch["xl_restraint"]
							) 

		cum_loss = 0.
		losses = {}
		for loss_name, loss_fn in loss_fns.items():
			weight = self.config[loss_name].weight
			loss = loss_fn()
			print( loss_name, "  ", loss, "  ", weight )
			if torch.isnan( loss ) or torch.isinf( loss ):
				print( f"{loss_name} loss is NaN. Skipping..." )
				loss = loss.new_tensor( 0., requires_grad = True )
			cum_loss = cum_loss + weight * loss
			losses[loss_name] = loss.detach().clone()
		losses["unscaled_loss"] = cum_loss.detach().clone()

		return cum_loss, losses

		# Scale the loss by the square root of the minimum of the crop size and
		# the (average) sequence length. See subsection 1.9.
		# seq_len = torch.mean(batch["seq_length"].float())

