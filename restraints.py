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
		if self.config.type == "ub_harmonic":
			return lambda: self.upper_bound_harmonic( out, restraint_feature["xl_res_dict"],
													restraint_feature["xl_max_bound"],
													restraint_feature["total_xls"] )
		elif self.config.type == "disto_xlr":
			return lambda: self.disto_xl_restraint( out, **restraint_feature )
		else:
			raise Exception( "At least one of the XL restraint types must be enabled..." )


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
		viols_mask = D > scaled_xl_max_bound
		return viols_mask


	def compute_distance_violation( self, D: torch.Tensor,
									viols_mask: torch.Tensor,
									scaled_xl_max_bound: float
									) -> torch.Tensor:
		"""
		Calculate the difference between the predicted distances and 
			the XL max bound for XL'd residue airs.
		"""
		if viols_mask.any():
			diff = ( D[viols_mask] - scaled_xl_max_bound )**2
		else:
			diff = D*0
		return diff


	def upper_bound_harmonic( self, out: Dict[str, torch.Tensor],
							xl_res_dict: torch.tensor,
							xl_max_bound: float,
							total_xls: int ) -> torch.Tensor:
		"""
		Calculate the cross-linking restraint loss as the mean squared error (MSE) for the 
		predicted Ca distances between the ceoss-linked residues from the max cross-link bound.
		Here, I am using the "final_atom_positions" for the restraint.
		Here we compute the loss per XL pair rather than all together in a single tensor.
		The loss for each XL residue pair is the minimum over all ambiguous XL pairs.
		Final loss is the mean over all XL pairs.

		Input:
		----------
		out --> output dict from the model.
		xl_res_dict --> dict with an index as key and the value corresponding to 
						all ambiguous XL pairs for a residue pair.
		xl_max_bound --> max distance between the cross-licked residues.

		Returns:
		----------
		loss --> xl restraint loss.
		"""
		D = self.final_pred_to_dist_map( out["final_atom_positions"] )

		# Adjust the length scales.
		scaled_xl_max_bound = xl_max_bound / self.length_scale

		# agg_loss = torch.tensor( 0.0 ).to( D.device )
		agg_loss = torch.zeros( 1 ).to( D.device )
		for xl_pair in xl_res_dict:
			res_idx1 = torch.tensor( xl_res_dict[xl_pair]["res1"] ).to( D.device )
			res_idx2 = torch.tensor( xl_res_dict[xl_pair]["res2"] ).to( D.device )
			# Indices for all ambiguous XLs for a cross-linked residue pair.
			xl_indices = ( 0, res_idx1, res_idx2 )

			viols_mask = D[xl_indices] > scaled_xl_max_bound
			# If all ambiguous pairs are violated.
			if viols_mask.all():
				# Get the minimum distance over all ambiguous pairs.
				min_D = torch.min( D[xl_indices] )
				diff = min_D - scaled_xl_max_bound
				squared_diff = diff**2
			# If any ambiguous pair is satisfied, the restraint is satisfied.
			else:
				squared_diff = ( D[xl_indices]*0 ).sum()
				# squared_diff = torch.tensor( 0.0, device = D.device )

			agg_loss += squared_diff
		
		# Normalizing by the total no. of cross-linked residue pairs.
		total_xls = torch.tensor( total_xls ).to( D.device )
		denom = self.eps + total_xls

		mse = agg_loss/ denom

		if self.config.func_form == "mse":
			loss = mse
		elif self.config.func_form == "rmse":
			rmse = torch.sqrt( mse + self.eps )
			loss = rmse
		return loss


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