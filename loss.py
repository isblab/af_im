import numpy as np
import torch
from torch import nn

from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.loss import ( find_structural_violations, 
								compute_renamed_ground_truth,
								# fape_loss,
								supervised_chi_loss,
								violation_loss,
								chain_center_of_mass_loss, distogram_loss
								)

from mod_openfold import fape_loss
from restraints import XlRestraint

from typing import Dict, Optional


class FapeLoss():
	# Just a wrapper for the OpenFold Fape loss.
	def __init__(  self, config ):
		self.name = "fape"
		self.config = config

	def get( self, out, batch ):
		return lambda: fape_loss(
								out,
								batch,
								self.config
								)

class SupervisedChiLoss():
	# Just a wrapper for the OpenFold Supervised Chi loss.
	def __init__(  self, config ):
		self.name = "supervised_chi"
		self.config = config

	def get( self, out, batch ):
		return lambda: supervised_chi_loss(
								out["sm"]["angles"],
								out["sm"]["unnormalized_angles"],
								**{**batch, **self.config},
								)

class ViolationLoss():
	# Just a wrapper for the OpenFold Violation loss.
	def __init__(  self, config ):
		self.name = "violation"
		self.config = config

	def get( self, out, batch ):
		# Violation loss does not need the argument batch, just kept here for uniformity.
		return lambda: violation_loss(
								out["violation"],
								**{**batch, **self.config}
								)

class ChainCenterOfMassLoss():
	# Just a wrapper for the OpenFold Chain center of mass loss.
	def __init__(  self, config ):
		self.name = "chain_center_of_mass"
		self.config = config

	def get( self, out, batch ):
		return lambda: chain_center_of_mass_loss(
								all_atom_pred_pos = out["final_atom_positions"],
								**{**batch, **self.config},
								)


class DistogramLoss():
	# Just a wrapper for the OpenFold Chain center of mass loss.
	def __init__(  self, config ):
		self.name = "distogram"
		self.config = config

	def get( self, out, batch ):
		return lambda: distogram_loss(
								logits = out["distogram_logits"],
								**{**batch, **self.config},
								)


class LossFunction( nn.Module ):
	def __init__( self, config, device: str ):
		self.config = config
		self.device = device

		self.loss_fns_included  =self.loss_included()


	def forward( self, out: Dict, batch: Dict, restraint_features: Dict ):
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

		# Iteratively calculate the loss for all included terms.
		loss_fns = {}
		for obj in self.loss_fns_included:
			loss_name = obj.name
			if loss_name == "xlr":
				loss_fns[loss_name] = obj.get( out, restraint_features["xl_restraint"] )
			else:
				loss_fns[loss_name] = obj.get( out, batch )


		cum_loss = torch.tensor( [0] ).to( self.device )
		losses = {}

		# adaptive_weight = {}
		# total_weight = torch.tensor( 0.0, device = self.device )
		# # calculating adaptive weights.
		# for loss_name, loss_fn in loss_fns.items():
		# 	loss = loss_fn()
		# 	if self.config[loss_name]["add_penalty"]:
		# 		if loss.item() == 0:
		# 			w = 0
		# 		else:
		# 			w = torch.tensor( 1/loss.item() + 1e-8, device = self.device )
		# 		adaptive_weight[loss_name] = w
		# 		total_weight = total_weight + w

		# print( adaptive_weight )
		# weights_tensor = torch.stack(
		#     [adaptive_weight[name] for name in adaptive_weight.keys()]
		# ).to( self.device )

		# Normalize weights using Softmax.
		# adaptive_weight_softmax = nn.functional.softmax( weights_tensor, dim = 0 )
		# print( adaptive_weight_softmax )

		# exit()

		# Normalize all weights.
		# for loss_name in adaptive_weight:
		# 	adaptive_weight[loss_name] = adaptive_weight[loss_name]/ total_weight
		# print( adaptive_weight )
		# exit()
		for loss_name, loss_fn in loss_fns.items():
			weight = torch.tensor( self.config[loss_name].weight, device = self.device )
			loss = loss_fn()

			# print_str += f"{loss_name}: {loss.item()} --> {weight}\t"

			if torch.isnan( loss ) or torch.isinf( loss ):
				print( f"{loss_name} loss is NaN. Skipping..." )
				loss = loss.new_tensor( 0., requires_grad = True )
			# If add_penalty is False, the loss will not be included for backprop.
			if self.config[loss_name]["add_penalty"]:

				# cum_loss = cum_loss + adaptive_weight[loss_name] * weight * loss
				cum_loss = cum_loss + weight * loss
			losses[loss_name] = loss.detach().clone()
		
		# print( print_str )

		losses["unscaled_loss"] = cum_loss.detach().clone()

		return cum_loss, losses

		# Scale the loss by the square root of the minimum of the crop size and
		# the (average) sequence length. See subsection 1.9.
		# seq_len = torch.mean(batch["seq_length"].float())


	def loss_included( self ):
		"""
		Loss terms to be included in the full loss function.
		"""
		loss_fns = []
		if self.config.fape.enabled:
			loss_fns.append( FapeLoss( self.config.fape ) )
		
		if self.config.supervised_chi.enabled:
			loss_fns.append( SupervisedChiLoss( self.config.supervised_chi ) )

		if self.config.violation.enabled:
			loss_fns.append( ViolationLoss( self.config.violation ) )

		if self.config.chain_center_of_mass.enabled:
			loss_fns.append( ChainCenterOfMassLoss( self.config.chain_center_of_mass ) )

		if self.config.distogram.enabled:
			loss_fns.append( DistogramLoss( self.config.distogram ) )

		if self.config.xlr.enabled:
			loss_fns.append( XlRestraint( self.config.xlr ) )

		return loss_fns

