from typing import Tuple, Dict, Optional, Any
import numpy as np
import ml_collections as mlc
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
from openfold.utils.tensor_utils import batched_gather

from mod_openfold import fape_loss
from restraints import XlRestraint, final_pred_to_dist_map


###############################################################################
###############################################################################
def get_excluded_volume(
	dist: torch.Tensor,
	intra_enabled: bool,
	inter_enabled: bool,
	intra_ev_mask: torch.Tensor,
	inter_ev_mask: torch.Tensor,
	intra_chain_dist: float,
	inter_chain_dist: float,
	eps: float ) -> torch.Tensor:
	"""
	Loss to penalize steric clashes between residues,
		both intra and inter-chain (excluded volume).
	
	intra_chain_dist -> distance between intrachain Ca-atoms to consider a clash.
	inter_chain_dist -> distance between interchain Ca-atoms to consider a clash.
	"""
	intra_ev_mask = intra_ev_mask.to( dist.device )
	inter_ev_mask = inter_ev_mask.to( dist.device )

	# Intrachain clashes.
	if intra_enabled:
		diff_intra = torch.clamp( intra_chain_dist - dist, min = 0.0 )
		intra_ev = torch.sum(
			( diff_intra[intra_ev_mask.bool()] )**2
			)/ ( torch.sum( intra_ev_mask ) + eps )
	else:
		intra_ev = torch.tensor( [0.0] ).to( dist.device )

	# Interchain clashes.
	if inter_enabled:
		diff_inter = torch.clamp( inter_chain_dist - dist, min = 0.0 )
		inter_ev = torch.sum(
			( diff_inter[inter_ev_mask.bool()] )**2
			)/ ( torch.sum( inter_ev_mask ) + eps )
	else:
		inter_ev = torch.tensor( [0.0] ).to( dist.device )

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

	#dist = final_pred_to_dist_map(
	#	final_atom_pos = out["final_atom_positions"],
	#	length_scale = length_scale,
	#	eps = eps )

	#diff = torch.clamp( inter_res_dist - dist, min = 0.0 )
	#loss = torch.sum(
	#	( diff[connectivity_mask.bool()] )**2
	#	)/ ( torch.sum( connectivity_mask ) + eps )

	#connectivity_viol = ( dist - inter_res_dist )**2/ tolerance_sigma
	#loss = torch.sum(
	#	connectivity_viol[connectivity_mask.bool()] )/ ( torch.sum( connectivity_mask ) + eps )
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
		gt_feature_dict: Dict[str, torch.Tensor],
		config: mlc.ConfigDict ) -> torch.Tensor:
	"""
	Compute a violation loss accounting for excluded volume and sequence connectivity.
	"""
	dist = final_pred_to_dist_map(
		final_atom_pos = final_atom_positions,
		length_scale = config.length_scale,
		eps = config.eps )

	ev = get_excluded_volume(
				dist = dist,
				intra_enabled = config.ev.intra_enabled,
				inter_enabled = config.ev.inter_enabled,
				intra_ev_mask = gt_feature_dict["intra_ev_mask"],
				inter_ev_mask = gt_feature_dict["inter_ev_mask"],
				intra_chain_dist = config.ev.intra_chain_dist,
				inter_chain_dist = config.ev.inter_chain_dist,
				eps = config.eps )

	if config.sc.enabled:
		sc = get_sequence_connectivity(
					dist = dist,
					connectivity_mask = gt_feature_dict["connectivity_mask"],
					inter_res_dist = config.sc.inter_res_dist,
					tolerance_sigma = config.sc.tolerance_sigma,
					eps = config.eps )
	else:
		sc = torch.tensor( [0.0] ).to( dist.device )

	print( f"ev = {ev}\tsc = {sc}" )
	loss = config.ev.weight*ev + config.sc.weight*sc
	return loss

###############################################################################	
###############################################################################
def get_com_loss(
	final_atom_positions: torch.Tensor,
	atom_mask: torch.Tensor,
	gt_feature_dict: Dict[str, Any],
	config: mlc.ConfigDict ) -> torch.Tensor:
	"""
	Restrain the COM fo the prediction to the COM of the ground truth COM.
		Here, ground truth COM is the COM of the initial predicted complex.
	"""
	gt_com = gt_feature_dict["com"]
	# [B, N, 37, 3] -> [B, 1]
	masked_pos = final_atom_positions*atom_mask.unsqueeze( -1 )
	pred_com = masked_pos.sum( dim = (1, 2) )/ atom_mask.sum( dim = ( 1, 2 ) )

	diff = pred_com - gt_com
	loss = ( diff**2 ).sum( dim = -1 )

	return loss

###############################################################################
###############################################################################
#def compute_complex_com( pos: torch.Tensor,
#						chain_pos_mask: torch.Tensor,
#						eps: float
#						) -> Vec3Array:
#	"""
#	Compute COM of the entire complex.
#	"""
#	# [B,C,N_pts,1] * [B,1,N_pts,3] -> [B,C,N_pts,3] -> [B,3]
#	center_sum = (chain_pos_mask[..., None] * pos[..., None, :, :]).sum(dim=[-3,-2])
#	# [B,3]/[B] -> [B,3]
#	complex_center = center_sum / (torch.sum(chain_pos_mask, dim=[-2,-1], keepdim=True) + eps)
#	return Vec3Array.from_array( complex_center )


#def compute_per_chain_com( pos: torch.Tensor,
#							chain_pos_mask: torch.Tensor,
#							eps: float
#							) -> Vec3Array:
#	"""
#	Compute the COM for each chain.
#	"""
#	# [B,C,N_pts,1] * [B,1,N_pts,3] -> [B,C,N_pts,3] -> [B,C,3]
#	center_sum = (chain_pos_mask[..., None] * pos[..., None, :, :]).sum(dim=-2)
#	# [B,C,3]/[B,c] -> [B,C,3]
#	chain_centers = center_sum / (torch.sum(chain_pos_mask, dim=-1, keepdim=True) + eps)
#	return Vec3Array.from_array( chain_centers )


#def get_center_of_mass(
#	all_atom_positions: torch.Tensor,
#	asym_id: torch.Tensor,
#	all_atom_mask: torch.Tensor,
#	eps: float = 1e-10 ) -> Tuple[Vec3Array, Vec3Array]:
#	"""
#	Compute the center of mass for:
#		the complex
#		each chain in the complex

#	Taken from openfold/utils/loss.py -> chain_center_of_mass_loss.
#	"""
#	ca_pos = rc.atom_order["CA"]
#	# [B,N_pts,37,3] -> [B,N_pts,3]
#	all_atom_positions = all_atom_positions[..., ca_pos, :]
#	# [B,N_pts,37] -> [B,N_pts,1]
#	all_atom_mask = all_atom_mask[..., ca_pos: (ca_pos + 1)]  # keep dim

#	# [B,N_pts] -> [B,N_pts,C]; C --> no. of chains
#	one_hot = torch.nn.functional.one_hot(asym_id.long()).to(dtype=all_atom_mask.dtype)
#	# Mask out non-existant atoms
#	one_hot = one_hot * all_atom_mask
#	# [B,N_pts,C] -> [B,C,N_pts]
#	chain_pos_mask = one_hot.transpose(-2, -1)

#	complex_center = compute_complex_com( pos = all_atom_positions,
#										chain_pos_mask = chain_pos_mask,
#										eps = eps )
#	chain_centers = compute_per_chain_com( pos = pos,
#										chain_pos_mask = chain_pos_mask,
#										eps = eps )
#	return complex_center, chain_centers


#def get_chain_distances( 
#    all_atom_pred_pos: torch.Tensor,
#    all_atom_mask: torch.Tensor,
#    asym_id: torch.Tensor,
#    eps: float = 1e-10, **kwargs
#    ) -> torch.Tensor:
#	"""
#	Compute the distance of the COM of each chain
#		from the COM of the complex.
#	"""
#	# [B,3] and [B,C,3]
#	complex_center, chain_centers = get_center_of_mass(
#		all_atom_positions = all_atom_positions,
#		asym_id = asym_id,
#		all_atom_mask = all_atom_mask,
#		eps = eps )

#	# [B,C]
#	chain_distance = euclidean_distance(
#		vec1 = pred_chain_centers,
#		vec2 = pred_complex_center,
#		eps = eps )
#	return chain_distance


#def com_loss( 
#    all_atom_pred_pos: torch.Tensor,
#    all_atom_mask: torch.Tensor,
#    asym_id: torch.Tensor,
#    gt_chain_distance: torch.Tensor,
#    clamp_distance: float = 10.0,
#    eps: float = 1e-10, **kwargs
#    ) -> torch.Tensor:
#	"""
#	A variant of the AF2 chain_center_of_mass_loss.
#		It preserves distance between the COM of all chains
#		as in the ground truth structure.

#	We wanna prevent the complex from blowing apart.
#		So we anchor the COM of all chains to the COM of the complex.
#	Compute the distance of the COM of all chains
#		from the COM of the complex in the predicted structure.
#	Penalize deviations in the distance of each chain
#		from the ground truth distance + some clamp_distance.
#	"""
#	chain_distance = get_chain_distances( 
#		all_atom_positions = all_atom_pred_positions,
#		all_atom_mask = all_atom_mask,
#		asym_id = asym_id,
#		eps = eps
#	)

#	diff = pred_chain_distance - gt_chain_distance - clamp_distance
#	squared_diff = torch.sum(
#		torch.clamp( diff, min = 0 )**2, dim = -1 )
#	com_loss = torch.mean( squared_diff )
#	return com_loss


# class ViolationLoss():
# 	# Just a wrapper for the OpenFold Violation loss.
# 	def __init__(  self, config ):
# 		self.name = "violation"
# 		self.config = config

# 	def get( self, out, batch ):
# 		# Violation loss does not need the argument batch, just kept here for uniformity.
# 		return lambda: violation_loss(
# 								out["violation"],
# 								**{**batch, **self.config}
# 								)

#class ChainCenterOfMassLoss():
#	# Just a wrapper for the OpenFold Chain center of mass loss.
#	def __init__(  self, config ):
#		self.name = "chain_center_of_mass"
#		self.config = config

#	def get( self, out, batch ):
#		return lambda: chain_center_of_mass_loss(
#								all_atom_pred_pos = out["final_atom_positions"],
#								**{**batch, **self.config},
#								)

class CenterOfMassLoss():
	# Just a wrapper for the custom center of mass loss.
	def __init__(  self, config ):
		self.name = "com"
		self.config = config

	def get( self, out, gt_feature_dict ):
		return lambda: get_com_loss(
					final_atom_positions = out["final_atom_positions"],
					atom_mask = out["final_atom_mask"],
					gt_feature_dict = gt_feature_dict,
					config = self.config
					)

#class DistogramLoss():
#	# Just a wrapper for the OpenFold Chain center of mass loss.
#	def __init__(  self, config ):
#		self.name = "distogram"
#		self.config = config

#	def get( self, out, batch ):
#		return lambda: distogram_loss(
#								logits = out["distogram_logits"],
#								**{**batch, **self.config},
#								)

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

	def get( self, out, gt_feature_dict ):
		return lambda: get_violation_loss(
				final_atom_positions = out["final_atom_positions"],
				gt_feature_dict = gt_feature_dict,
                config = self.config
            )


#class ExcludedVolumeLoss():
#	# Just a wrapper for the custom excluded volume loss.
#	def __init__(  self, config ):
#		self.name = "ev"
#		self.config = config

#	def get( self, out, batch ):
#		return lambda: get_excluded_volume(
#				out = out,
#				intra_ev_mask = batch["intra_ev_mask"],
#				inter_ev_mask = batch["inter_ev_mask"],
#				intra_chain_dist = self.config.intra_chain_dist,
#				inter_chain_dist = self.config.inter_chain_dist,
#				length_scale = self.config.length_scale,
#				eps = self.config.eps
#			)


#class SequenceConnectivityLoss():
#	# Just a wrapper for the custom sequence connectivity loss.
#	def __init__(  self, config ):
#		self.name = "sc"
#		self.config = config

#	def get( self, out, batch ):
#		return lambda: get_sequence_connectivity(
#				out = out,
#				connectivity_mask = batch["connectivity_mask"],
#				inter_res_tolerance = self.config.inter_res_tolerance,
#				length_scale = self.config.length_scale,
#				eps = self.config.eps
#			)


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

		self.loss_fns_included  =self.loss_included()


	def forward( self, out: Dict[str, Any],
		gt_feature_dict: Dict[str, torch.Tensor],
		restraint_features: Dict[str, Any] ):
		# AF2 losses require the atom14 representation.
		# atom37 = out["final_atom_positions"]
		# atom14 = atom37_to_atom14( atom37 = atom37, gt_feature_dict = gt_feature_dict )

		# if "violation" not in out.keys():
		# 	out["violation"] = find_structural_violations(
		# 		gt_feature_dict,
		# 		atom14,
		# 		# out["sm"]["positions"][-1],
		# 		**self.config.violation,
		# 	)

		# if "renamed_atom14_gt_positions" not in out.keys():
		# 	gt_feature_dict.update(
		# 		compute_renamed_ground_truth(
		# 			gt_feature_dict,
		# 			atom14
		# 			# out["sm"]["positions"][-1],
		# 		)
		# 	)

		device = out["final_atom_positions"].device
		# Iteratively calculate the loss for all included terms.
		loss_fns = {}
		for obj in self.loss_fns_included:
			loss_name = obj.name
			if loss_name == "xlr":
				loss_fns[loss_name] = obj.get( out, restraint_features["xl_restraint"] )
			else:
				loss_fns[loss_name] = obj.get( out, gt_feature_dict )


		cum_loss = torch.tensor( [0] ).to( device )
		losses = {}

		for loss_name, loss_fn in loss_fns.items():
			weight = torch.tensor( self.config[loss_name].weight, device = device )
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
		#if self.config.fape.enabled:
		#	loss_fns.append( FapeLoss( self.config.fape ) )
		
		#if self.config.supervised_chi.enabled:
		#	loss_fns.append( SupervisedChiLoss( self.config.supervised_chi ) )

		#if self.config.violation.enabled:
		#	loss_fns.append( ViolationLoss( self.config.violation ) )

		#if self.config.chain_center_of_mass.enabled:
		#	loss_fns.append( ChainCenterOfMassLoss( self.config.chain_center_of_mass ) )

		#if self.config.distogram.enabled:
		#	loss_fns.append( DistogramLoss( self.config.distogram ) )

		#if self.config.rigid_chain.enabled:
		#	loss_fns.append( RigidChainLoss( self.config.rigid_chain ) )

		#if self.config.gdr.enabled:
		#	loss_fns.append( GaussianDistanceRestraint( self.config.gdr ) )

		if self.config.violation.enabled:
			loss_fns.append( ViolationLoss( self.config.violation ) )

		if self.config.com.enabled:
			loss_fns.append( CenterOfMassLoss( self.config.com ) )
		#if self.config.ev.enabled:
		#	loss_fns.append( ExcludedVolumeLoss( self.config.ev ) )

		#if self.config.sc.enabled:
		#	loss_fns.append( SequenceConnectivityLoss( self.config.sc ) )

		if self.config.xlr.enabled:
			loss_fns.append( XlRestraint( self.config.xlr ) )

		return loss_fns

