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
				max_bound_dist: Optional[float] = 35.0,
				lambda_: Optional[float] = 0.5
				):
	"""
	Calculate the cross-linking restraint loss as the mean squared deviation for the 
	predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
	"""
	print( out["sm"]["frames"].shape )
	print( out["sm"]["positions"].shape )
	# traj = out["sm"]["frames"]
	pred_positions = out["final_atom_positions"]
	# ### need to check if the traj belongs to 4*4 matrix or a tensor_7
	# print( traj.shape )
	# if traj.shape[-1] == 7:
	# 	pred_frames = Rigid.from_tensor_7( traj )
	# elif traj.shape[-1] == 4:
	# 	pred_frames = Rigid.from_tensor_4x4( traj )
	# print( pred_frames.shape )

	# pred_frames = Rigid(
	# 	Rotation( 
	# 		rot_mats = pred_frames.get_rots().get_rot_mats(), 
	# 		quats = None ),
	# 		pred_frames.get_trans(),
	# 	)
	# print( pred_frames.shape )
	# print( "\n---------------------------------------------\n" )

	# # [*, N_frames, N_pts, 3]
	# # Get the predicted positions in the predicted frames.
	# local_pred_pos = pred_frames.invert()[..., None].apply(
	# 	pred_positions[..., None, :, :]
	# )
	# ca_pos = local_pred_pos[..., 1, :]
	
	xl_tgt_mask = xl_tgt_mask.unsqueeze( 0 )
	xl_res_mask = xl_res_mask.unsqueeze( 0 )

	ca_pos = pred_positions[..., 1, :]
	ca_pos = ca_pos
	diff = ca_pos.unsqueeze( 2 ) - ca_pos.unsqueeze( 1 )
	print( diff.shape )
	D = torch.sqrt( 
					torch.sum( ( diff )**2, dim = -1 )
					)
	D = D*xl_res_mask
	mask = D > max_bound_dist
	print( D[mask] )

	# Compute the loss only for entries where the mask is True
	loss = torch.zeros_like( D )
	loss[mask] = lambda_ * ( D[mask] - xl_tgt_mask[mask] )**2

	# Aggregate the loss (sum or mean).
	loss = torch.mean( loss )
	return loss



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
							# xl_tgt_mask = batch["xl_restraint"]["xl_tgt_mask"], 
							# xl_res_mask = batch["xl_restraint"]["xl_res_mask"]
							) 

		cum_loss = 0.
		losses = {}
		print( self.config.keys() )
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

