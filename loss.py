from typing import Tuple, Dict, Optional
import numpy as np
import torch
from torch import nn

from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.geometry.vector import Vec3Array, euclidean_distance
from openfold.utils.loss import ( find_structural_violations, 
								compute_renamed_ground_truth,
								supervised_chi_loss,
								violation_loss,
								chain_center_of_mass_loss, distogram_loss
								)

from mod_openfold import fape_loss
from restraints import XlRestraint, final_pred_to_dist_map




def compute_complex_com( pos: torch.Tensor,
						chain_pos_mask: torch.Tensor,
						eps: float
						) -> Vec3Array:
	"""
	Compute COM of the entire complex.
	"""
	# [B,C,N_pts,1] * [B,1,N_pts,3] -> [B,C,N_pts,3] -> [B,3]
	center_sum = (chain_pos_mask[..., None] * pos[..., None, :, :]).sum(dim=[-3,-2])
	# [B,3]/[B] -> [B,3]
	complex_center = center_sum / (torch.sum(chain_pos_mask, dim=[-2,-1], keepdim=True) + eps)
	return Vec3Array.from_array( complex_center )


def compute_per_chain_com( pos: torch.Tensor,
							chain_pos_mask: torch.Tensor,
							eps: float
							) -> Vec3Array:
	"""
	Compute the COM for each chain.
	"""
	# [B,C,N_pts,1] * [B,1,N_pts,3] -> [B,C,N_pts,3] -> [B,C,3]
	center_sum = (chain_pos_mask[..., None] * pos[..., None, :, :]).sum(dim=-2)
	# [B,C,3]/[B,c] -> [B,C,3]
	chain_centers = center_sum / (torch.sum(chain_pos_mask, dim=-1, keepdim=True) + eps)
	return Vec3Array.from_array( chain_centers )


def get_center_of_mass(
	all_atom_positions: torch.Tensor,
	asym_id: torch.Tensor,
	all_atom_mask: torch.Tensor,
	eps: float = 1e-10 ) -> Tuple[Vec3Array, Vec3Array]:
	"""
	Compute the center of mass for:
		the complex
		each chain in the complex

	Taken from openfold/utils/loss.py -> chain_center_of_mass_loss.
	"""
	ca_pos = residue_constants.atom_order["CA"]
	# [B,N_pts,37,3] -> [B,N_pts,3]
	all_atom_positions = all_atom_positions[..., ca_pos, :]
	# [B,N_pts,37] -> [B,N_pts,1]
	all_atom_mask = all_atom_mask[..., ca_pos: (ca_pos + 1)]  # keep dim

	# [B,N_pts] -> [B,N_pts,C]; C --> no. of chains
	one_hot = torch.nn.functional.one_hot(asym_id.long()).to(dtype=all_atom_mask.dtype)
	# Mask out non-existant atoms
	one_hot = one_hot * all_atom_mask
	# [B,N_pts,C] -> [B,C,N_pts]
	chain_pos_mask = one_hot.transpose(-2, -1)

	complex_center = compute_complex_com( pos = all_atom_positions,
										chain_pos_mask = chain_pos_mask,
										eps = eps )
	chain_centers = compute_per_chain_com( pos = pos,
										chain_pos_mask = chain_pos_mask,
										eps = eps )
	return complex_center, chain_centers


def get_chain_distances( 
    all_atom_pred_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    eps: float = 1e-10, **kwargs
    ) -> torch.Tensor:
	"""
	Compute the distance of the COM of each chain
		from the COM of the complex.
	"""
	# [B,3] and [B,C,3]
	complex_center, chain_centers = get_center_of_mass(
		all_atom_positions = all_atom_positions,
		asym_id = asym_id,
		all_atom_mask = all_atom_mask,
		eps = eps )

	# [B,C]
	chain_distance = euclidean_distance(
		vec1 = pred_chain_centers,
		vec2 = pred_complex_center,
		eps = eps )
	return chain_distance


def com_loss( 
    all_atom_pred_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    gt_chain_distance: torch.Tensor,
    clamp_distance: float = 10.0,
    eps: float = 1e-10, **kwargs
    ) -> torch.Tensor:
	"""
	A variant of the AF2 chain_center_of_mass_loss.
		It preserves distance between the COM of all chains
		as in the ground truth structure.

	We wanna prevent the complex from blowing apart.
		So we anchor the COM of all chains to the COM of the complex.
	Compute the distance of the COM of all chains
		from the COM of the complex in the predicted structure.
	Penalize deviations in the distance of each chain
		from the ground truth distance + some clamp_distance.
	"""
	chain_distance = get_chain_distances( 
		all_atom_positions = all_atom_pred_positions,
		all_atom_mask = all_atom_mask,
		asym_id = asym_id,
		eps = eps
	)

	diff = pred_chain_distance - gt_chain_distance - clamp_distance
	squared_diff = torch.sum(
		torch.clamp( diff, min = 0 )**2, dim = -1 )
	com_loss = torch.mean( squared_diff )
	return com_loss

###############################################################################
###############################################################################
def rigid_chain_loss(
	final_atom_position: torch.Tensor,
	gt_distance_map: torch.Tensor,
	asym_id: torch.Tensor,
	length_scale: float,
	eps: float ):
	"""
	Penalize intrachain distance deviations from the ground truth structure.
	Thi smay help keep each chain as a rigid unit and prevent squishing.
	"""
	pred_dist_map = final_pred_to_dist_map(
		final_atom_pos = final_atom_position,
		length_scale = length_scale,
		eps = eps
		)
	gt_distance_map = gt_distance_map/length_scale

	intrachain_mask = asym_id == asym_id

	squared_diff = ( pred_dist_map*intrachain_mask - gt_distance_map*intrachain_mask )**2
	loss = torch.mean( squared_diff )

	return loss

###############################################################################
###############################################################################
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

class CenterOfMassLoss():
	# Just a wrapper for the custom center of mass loss.
	def __init__(  self, config ):
		self.name = "center_of_mass"
		self.config = config

	def get( self, out, batch ):
		return lambda: com_loss(
					all_atom_pred_pos = out["all_atom_positions"],
					**{**batch, **self.config},
					)


class RigidChainLoss():
	# Custom loss to preserve intrachain distances.
	def __init__(  self, config ):
		self.name = "rigid_chain"
		self.config = config

	def get( self, out, batch ):
		return lambda: rigid_chain_loss(
						final_atom_position = out["final_atom_positions"],
						gt_distance_map = batch["distance_map"],
						asym_id = batch["asym_id"],
						length_scale = self.config.length_scale,
						eps = self.config.eps
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

		if self.config.rigid_chain.enabled:
			loss_fns.append( RigidChainLoss( self.config.rigid_chain ) )

		if self.config.xlr.enabled:
			loss_fns.append( XlRestraint( self.config.xlr ) )

		return loss_fns

