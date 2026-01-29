from typing import Dict
import torch

from openfold.utils.loss import softmax_cross_entropy

from utils.metric_utils import final_pred_to_dist_map


class XlRestraint():
	def __init__( self, config ):
		self.name = "xlr"
		self.config = config
		self.allow_xl_tolerance = config.allow_xl_tolerance
		self.length_scale = config.length_scale
		self.eps = config.eps


	def get( self, out, restraint_feature ):
		if self.allow_xl_tolerance:
			xl_sat_tolerance = restraint_feature["xl_sat_tolerance"]
			xl_max_bound = restraint_feature["xl_max_bound"] + xl_sat_tolerance
		else:
			xl_max_bound = restraint_feature["xl_max_bound"]
		if self.config.type == "ub_harmonic":
			print( "Using an upper bound harmonic as XL restraint." )
			return lambda: self.upper_bound_harmonic( out, restraint_feature["xl_res_dict"],
													xl_max_bound,
													restraint_feature["total_xls"] )
		elif self.config.type == "pseudo_huber":
			print( "Using pseudo huber as XL restraint." )
			return lambda: self.pseudo_huber( out, restraint_feature["xl_res_dict"],
													xl_max_bound,
													restraint_feature["total_xls"] )
		elif self.config.type == "softplus":
			print( "Using softplus loss as XL restraint." )
			return lambda: self.softplus_loss( out, restraint_feature["xl_res_dict"],
													xl_max_bound,
													restraint_feature["total_xls"] )
		elif self.config.type == "gated_harmonic":
			print( "Using gated_harmonic loss as XL restraint." )
			return lambda: self.gated_harmonic_loss( out, restraint_feature["xl_res_dict"],
													xl_max_bound,
													restraint_feature["total_xls"] )

		else:
			raise Exception( "At least one of the XL restraint types must be enabled..." )


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
		D = final_pred_to_dist_map(
			final_atom_pos = out["final_atom_positions"],
			length_scale = self.length_scale,
			eps = self.eps )

		if torch.isnan(out["final_atom_positions"]).any() or torch.isinf(out["final_atom_positions"]).any():
			print("NaN or Inf detected in final_atom_positions!")

		# Adjust the length scales.
		scaled_xl_max_bound = xl_max_bound / self.length_scale

		# Aggregate loss across all XLs.
		agg_loss = torch.zeros( 1 ).to( D.device )
		for xl_pair in xl_res_dict:
			res_idx1 = torch.tensor( xl_res_dict[xl_pair]["res1"] ).to( D.device )
			res_idx2 = torch.tensor( xl_res_dict[xl_pair]["res2"] ).to( D.device )
			# Indices for all ambiguous XLs for a cross-linked residue pair.
			xl_indices = ( 0, res_idx1, res_idx2 )

			viols_mask = D[xl_indices] > scaled_xl_max_bound
			# If any ambiguous pairs are violated.
			if viols_mask.all():
				# Get the minimum distance over all ambiguous pairs.
				min_D = torch.min( D[xl_indices] )
				diff = min_D - scaled_xl_max_bound
				squared_diff = diff**2
			else:
				# If any ambiguous pair is satisfied, the restraint is satisfied.
				squared_diff = ( D[xl_indices]*0 ).sum()

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


	def pseudo_huber( self, out: Dict[str, torch.Tensor],
							xl_res_dict: torch.tensor,
							xl_max_bound: float,
							total_xls: int ) -> torch.Tensor:
		"""
		Quick and dirty implementation for now. Just wanna see if it works.
		Using pseudo huber loss for the XL restraint.
			L2-penalty for small violations and L1-penalty for large violations.
		Psudo huber = delta**2( sqrt( 1 + ( ( y_hat - y )/delta )**2 ) ) - 1
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
		delta = self.config.huber_delta/ self.length_scale
		D = final_pred_to_dist_map(
			final_atom_pos = out["final_atom_positions"],
			length_scale = self.length_scale,
			eps = self.eps )

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
				a = torch.sqrt( 1 + ( diff/delta )**2 )
				huber = delta**2*a - 1
			# If any ambiguous pair is satisfied, the restraint is satisfied.
			else:
				huber = ( D[xl_indices]*0 ).sum()
				# squared_diff = torch.tensor( 0.0, device = D.device )

			agg_loss += huber
		
		# Normalizing by the total no. of cross-linked residue pairs.
		total_xls = torch.tensor( total_xls ).to( D.device )
		denom = self.eps + total_xls

		loss = agg_loss/ denom

		return loss


	def softplus_loss( self, out: Dict[str, torch.Tensor],
							xl_res_dict: torch.tensor,
							xl_max_bound: float,
							total_xls: int ) -> torch.Tensor:
		"""
		Using the softplus function to model the cross-linking data.
		Softplus is a smooth approximation to ReLU and the steepness.
			The steepness of the loss within max bound can be
				controlled using the beta parameter.
				Small beta -> more steeper
				Large beta -> more flat.
			L = 1/beta * log( 1 + exp( beta*( d - d_max ) ) )
			d -> distance for the cross-linked residues in the predicted structure.
			d_max -> XL max bound.
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
		D = final_pred_to_dist_map(
			final_atom_pos = out["final_atom_positions"],
			length_scale = self.length_scale,
			eps = self.eps )

		if torch.isnan(out["final_atom_positions"]).any() or torch.isinf(out["final_atom_positions"]).any():
			print("NaN or Inf detected in final_atom_positions!")

		# Adjust the length scales.
		scaled_xl_max_bound = xl_max_bound / self.length_scale

		# Aggregate loss across all XLs.
		agg_loss = torch.zeros( 1 ).to( D.device )
		for xl_pair in xl_res_dict:
			res_idx1 = torch.tensor( xl_res_dict[xl_pair]["res1"] ).to( D.device )
			res_idx2 = torch.tensor( xl_res_dict[xl_pair]["res2"] ).to( D.device )
			# Indices for all ambiguous XLs for a cross-linked residue pair.
			xl_indices = ( 0, res_idx1, res_idx2 )

			# Min distance across all ambiguous XL pair.
			min_D = min_D = torch.min( D[xl_indices] )
			softplus = torch.nn.functional.softplus(
				x = min_D - scaled_xl_max_bound,
				beta = self.config.beta,
				threshold = scaled_xl_max_bound
			)

			agg_loss += softplus

		# Normalizing by the total no. of cross-linked residue pairs.
		total_xls = torch.tensor( total_xls ).to( D.device )
		denom = self.eps + total_xls
		loss = agg_loss/ denom

		return loss


	def gated_harmonic_loss( self, out: Dict[str, torch.Tensor],
							xl_res_dict: torch.tensor,
							xl_max_bound: float,
							total_xls: int ) -> torch.Tensor:
		"""
		Using a harmonic penalty weighted by a logistic function
			(gated harmonic) to model the cross-linking restraint.
		w = sigmoid( ( d - d_max )/beta )
			beta controls the softness of the max bound.
				Large beta -> Steeper penalty within the max bound
					pulling it closer to the max bound.
				Small beta -> Flatter penalty within the max bound
					potentially allowing exploration.
		L = w*( d - d_max )**2
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
		D = final_pred_to_dist_map(
			final_atom_pos = out["final_atom_positions"],
			length_scale = self.length_scale,
			eps = self.eps )

		if torch.isnan(out["final_atom_positions"]).any() or torch.isinf(out["final_atom_positions"]).any():
			print("NaN or Inf detected in final_atom_positions!")

		# Adjust the length scales.
		scaled_xl_max_bound = xl_max_bound / self.length_scale

		# Aggregate loss across all XLs.
		agg_loss = torch.zeros( 1 ).to( D.device )
		for xl_pair in xl_res_dict:
			res_idx1 = torch.tensor( xl_res_dict[xl_pair]["res1"] ).to( D.device )
			res_idx2 = torch.tensor( xl_res_dict[xl_pair]["res2"] ).to( D.device )
			# Indices for all ambiguous XLs for a cross-linked residue pair.
			xl_indices = ( 0, res_idx1, res_idx2 )

			min_D = min_D = torch.min( D[xl_indices] )
			# Sigmoid weight
			w = torch.sigmoid( min_D - scaled_xl_max_bound )/self.config.beta
			gated_harmonic = w*( min_D - scaled_xl_max_bound )**2

			agg_loss += gated_harmonic

		# Normalizing by the total no. of cross-linked residue pairs.
		total_xls = torch.tensor( total_xls ).to( D.device )
		denom = self.eps + total_xls
		loss = agg_loss/ denom

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