from typing import Tuple, Dict, Optional, Any
import numpy as np
import ml_collections as mlc
import torch
from torch import nn

# from openfold.utils.rigid_utils import Rotation, Rigid
# from openfold.utils.geometry.vector import Vec3Array, euclidean_distance
# from openfold.utils.loss import ( find_structural_violations, 
# 								compute_renamed_ground_truth,
# 								supervised_chi_loss,
# 								violation_loss,
# 								chain_center_of_mass_loss, distogram_loss
# 								)
from openfold.utils.tensor_utils import batched_gather

# from mod_openfold import fape_loss
from restraints import XlRestraint, final_pred_to_dist_map


###############################################################################
###############################################################################
def get_excluded_volume(
	dist: torch.Tensor,
	intra_enabled: bool,
	inter_enabled: bool,
	intra_ev_mask: torch.Tensor,
	inter_ev_mask: torch.Tensor,
	allowed_res_dist: torch.Tensor,
	clash_tolerance: float,
	beta: float,
	eps: float ) -> torch.Tensor:
	"""
	Loss to penalize steric clashes between residues,
		both intra and inter-chain (excluded volume).
	Residue pairs within allowed_res_dist+clash_tolerance are
		considered for EV loss to account for near-clashes.

	dist -> [1, N, N]
	intra_ev_mask -> [1, N, N] mask for intrachain residues.
	inter_ev_mask -> [1, N, N] mask for interchain residues.
	allowed_res_dist -> [1, N, N] allowed distance between the Ca-atoms of two residues.
	"""
	violated_dist_mask = dist <= allowed_res_dist+clash_tolerance
	# violated_dist_mask = dist <= 8.0+clash_tolerance
	violated_dist = allowed_res_dist - dist
	# Intrachain clashes.
	if intra_enabled:
		violated_intra_pairs = violated_dist_mask & intra_ev_mask.bool()
		# if violated_intra_pairs.any():
		intra_viols = violated_dist[violated_intra_pairs]

		diff_intra = torch.nn.functional.softplus(
			intra_viols,
			beta = beta )
		denom = intra_ev_mask.sum() + eps
		intra_ev = diff_intra.sum()/denom
		# else:
		# 	intra_ev = dist.new_tensor( 0.0 )
	else:
		intra_ev = dist.new_tensor( 0.0 )

	# Interchain clashes.
	if inter_enabled:
		violated_inter_pairs = violated_dist_mask & inter_ev_mask.bool()
		print( violated_inter_pairs.sum() )
		# if violated_inter_pairs.any():
		inter_viols = violated_dist[violated_inter_pairs]
		diff_inter = torch.nn.functional.softplus(
			inter_viols,
			beta = beta )
		denom = inter_ev_mask.sum() + eps
		inter_ev = diff_inter.sum()/denom
		# else:
		# 	inter_ev = dist.new_tensor( 0.0 )
	else:
		inter_ev = dist.new_tensor( 0.0 )

	loss = intra_ev + inter_ev

	return loss

###############################################################################
###############################################################################
def get_sequence_connectivity(
	dist: torch.Tensor,
	inter_res_dist: float,
	tolerance_sigma: float,
	connectivity_mask: torch.Tensor,
	eps: float ) -> torch.Tensor:
	"""
	Loss to penalize violation fo sequence connectivity.
	Distance between Ca-stoms must be within tolerance,
		modeled as a gaussian.
	
	inter_res_dist -> expected distance between adjacent Ca-atoms to consider connectivity.
	tolerance_sigma -> standard deviation around the expected distance.
	"""
	connectivity_mask = connectivity_mask.to( dist.device )

	# Interchain clashes.
	masked = dist*connectivity_mask
	connectivity_viol = masked > inter_res_dist
	denom = torch.sum( connectivity_viol )
	loss = torch.sum(
		( masked[connectivity_viol] - inter_res_dist )**2
		)/ ( denom + eps )

	return loss

###############################################################################
###############################################################################
def get_violation_loss(
		final_atom_positions: torch.Tensor,
		batch: Dict[str, torch.Tensor],
		config: mlc.ConfigDict ) -> torch.Tensor:
	"""
	Compute a violation loss accounting for excluded volume and sequence connectivity.
	"""
	dist = final_pred_to_dist_map(
		final_atom_pos = final_atom_positions,
		eps = config.eps )

	ev = get_excluded_volume(
				dist = dist,
				intra_enabled = config.ev.intra_enabled,
				inter_enabled = config.ev.inter_enabled,
				intra_ev_mask = batch["intra_ev_mask"],
				inter_ev_mask = batch["inter_ev_mask"],
				allowed_res_dist = batch["allowed_res_dist"],
				clash_tolerance = config.ev.clash_tolerance,
				beta = config.ev.beta,
				eps = config.eps )

	if config.sc.enabled:
		sc = get_sequence_connectivity(
					dist = dist,
					connectivity_mask = batch["connectivity_mask"],
					inter_res_dist = config.sc.inter_res_dist,
					tolerance_sigma = config.sc.tolerance_sigma,
					eps = config.eps )
	else:
		sc = dist.new_tensor( 0.0, requires_grad = True )

	print( f"ev = {ev}\tsc = {sc}" )
	loss = config.ev.weight*ev + config.sc.weight*sc
	return loss

###############################################################################	
###############################################################################
def get_com_loss(
	final_atom_positions: torch.Tensor,
	atom_mask: torch.Tensor,
	batch: Dict[str, Any],
	config: mlc.ConfigDict ) -> torch.Tensor:
	"""
	Restrain the COM fo the prediction to the COM of the ground truth COM.
		Here, ground truth COM is the COM of the initial predicted complex.
	"""
	gt_com = batch["com"]
	# [B, N, 37, 3] -> [B, 1]
	masked_pos = final_atom_positions*atom_mask.unsqueeze( -1 )
	pred_com = masked_pos.sum( dim = (1, 2) )/ atom_mask.sum( dim = ( 1, 2 ) )

	diff = pred_com - gt_com
	loss = ( diff**2 ).sum( dim = -1 )

	return loss

###############################################################################
###############################################################################
class CenterOfMassLoss():
	# Just a wrapper for the custom center of mass loss.
	def __init__(  self, config ):
		self.name = "com"
		self.config = config

	def get( self, out, batch ):
		return lambda: get_com_loss(
					final_atom_positions = out["final_atom_positions"],
					atom_mask = out["final_atom_mask"],
					batch = batch,
					config = self.config
					)

# class ViolationLoss():
# 	# Just a wrapper for the OpenFold violation loss.
# 	def __init__(  self, config ):
# 		self.name = "violation"
# 		self.config = config

# 	def get( self, out, gt_feature_dict ):
# 		return lambda: violation_loss(
#                 out["violation"],
#                 **{**gt_feature_dict, **self.config},
#             )

class ViolationLoss():
	# Just a wrapper for the custom violation loss.
	def __init__(  self, config ):
		self.name = "violation"
		self.config = config

	def get( self, out, batch ):
		return lambda: get_violation_loss(
				final_atom_positions = out["final_atom_positions"],
				batch = batch,
                config = self.config
            )


def atom37_to_atom14(
		atom37: torch.Tensor,
		gt_feature_dict: Dict[str, torch.Tensor] ):
	"""
	Convert atom37 representation to atom14.
	Adapted from openfold/data/feats/atom14_to_atom37().
	"""
	atom14_data = batched_gather(
		atom37,
		gt_feature_dict["residx_atom14_to_atom37"],
		dim=-2,
		no_batch_dims=len(atom37.shape[:-2]),
	)

	atom14_data = atom14_data * gt_feature_dict["atom14_atom_exists"][..., None]

	return atom14_data

###############################################################################
###############################################################################
class LossFunction( nn.Module ):
	def __init__( self, config, device: str ):
		self.config = config
		self.device = device

		self.loss_running = {}
		self.loss_fns_included  = self.loss_included()


	def forward( self,
		out: Dict[str, Any],
		batch: Dict[str, Any] ):
		device = out["final_atom_positions"].device
		# Iteratively calculate the loss for all included terms.
		loss_fns = {}
		for obj in self.loss_fns_included:
			loss_name = obj.name
			if loss_name == "xlr":
				loss_fns[loss_name] = obj.get( out, batch["xl_restraint"] )
			else:
				loss_fns[loss_name] = obj.get( out, batch )


		cum_loss = torch.tensor( [0.0] ).to( device )
		losses = {}

		for loss_name, loss_fn in loss_fns.items():
			weight = torch.tensor( self.config[loss_name].weight, device = device )
			loss = loss_fn()

			if torch.isnan( loss ) or torch.isinf( loss ):
				print( f"{loss_name} loss is NaN. Skipping..." )
				loss = loss.new_tensor( 0.0, requires_grad = True )

			# if loss_name not in self.loss_running:
			# 	self.loss_running[loss_name] = loss.detach()
			# else:
			# 	self.loss_running[loss_name] = 0.9*self.loss_running[loss_name] + 0.1*loss.detach()

			# If add_penalty is False, the loss will not be included for backprop.
			if self.config[loss_name]["add_penalty"]:
				# loss = ( loss/( self.loss_running[loss_name] + 1e-8 ) )
				cum_loss = cum_loss + weight * loss
			losses[loss_name] = loss.detach().clone()

		losses["unscaled_loss"] = cum_loss.detach().clone()

		return cum_loss, losses


	def loss_included( self ):
		"""
		Loss terms to be included in the full loss function.
		"""
		loss_fns = []

		#if self.config.violation.enabled:
		#	loss_fns.append( ViolationLoss( self.config.violation ) )

		if self.config.violation.enabled:
			loss_fns.append( ViolationLoss( self.config.violation ) )

		if self.config.com.enabled:
			loss_fns.append( CenterOfMassLoss( self.config.com ) )

		if self.config.xlr.enabled:
			loss_fns.append( XlRestraint( self.config.xlr ) )

		return loss_fns

