import numpy as np
import torch
from torch import nn

from openfold.utils.loss import ( find_structural_violations, 
								compute_renamed_ground_truth,
								fape_loss,
								supervised_chi_loss,
								violation_loss,
								chain_center_of_mass_loss
								)


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

		cum_loss = 0.
		losses = {}
		for loss_name, loss_fn in loss_fns.items():
			weight = self.config[loss_name].weight
			loss = loss_fn()
			if torch.isnan(loss) or torch.isinf(loss):
				print( f"{loss_name} loss is NaN. Skipping..." )
				loss = loss.new_tensor(0., requires_grad=True)
			cum_loss = cum_loss + weight * loss
			losses[loss_name] = loss.detach().clone()
		losses["unscaled_loss"] = cum_loss.detach().clone()

		return cum_loss, losses

		# Scale the loss by the square root of the minimum of the crop size and
		# the (average) sequence length. See subsection 1.9.
		# seq_len = torch.mean(batch["seq_length"].float())
