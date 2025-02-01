import ml_collections as mlc

import torch
from torchmetrics import Accuracy

from typing import Dict, List


def pred_to_dist_map( pred_positions: torch.Tensor, xl_max_bound: float
			) -> torch.Tensor:
	"""
	Using the "final_atom_positions" to create a Ca distance map.
	"""
	# Extracting Ca-coordinates.
	# [B,N,3] --> For 2ayo: [1,480,3]
	ca_pos = pred_positions[..., 1, :]
	diff = ca_pos.unsqueeze( 2 ) - ca_pos.unsqueeze( 1 )
	D = torch.sqrt( 
					torch.sum( ( diff )**2, dim = -1 ) + 1e-8
					)

	return D


class XlMetrics():
	def __init__( self, config: Dict, restraint_features: Dict ):
		"""
		Calculate the percentage of XLs satisfied.
		Check what all XLs are satisfied.
		"""
		self.name = "xlr"
		self.length_scale = 10.0
		self.xl_max_bound = restraint_features["xl_max_bound"]/ self.length_scale
		# A binary mask for selecting only cross-linked residues.
		self.xl_res_mask = restraint_features["xl_res_mask"]

		# Ensure the xl_res_mask is binary.
		assert torch.all( ( self.xl_res_mask == 0 ) | ( self.xl_res_mask == 1 ) ), "xl_res_mask should be binary"

		# Total cross-linked residue pairs. These will be twice the 
		# 	actual no. of crosslinked residue pairs as we consider both ij and ji interactions.
		self.total_xls = torch.count_nonzero( self.xl_res_mask )
		# Used for keeping track of all satisfied XLs across all epochs.
		self.satisfied_xls = torch.zeros( ( self.xl_res_mask.shape ) )


	def forward( self, out: Dict[str, torch.Tensor] ):
		self.get_predicted_distance_map( out )
		self.apply_xl_mask()
		
		xl_satisfaction = self.compute_xl_satisfaction()
		self.track_satisfied_xls()

		return xl_satisfaction


	def get_predicted_distance_map( self, out: Dict[str, torch.Tensor] ) -> None:
		"""
		Get the predicted distance map.
		"""
		self.D = pred_to_dist_map( out["final_atom_positions"], 
									self.xl_max_bound )

		# Adjust the length scales.
		self.D = self.D / self.length_scale


	def apply_xl_mask( self ) -> None:
		"""
		Mask the non cross-linked residues.
		Convert to binary.
		"""
		# Consider only the cross-linked residues.
		self.D = self.D*self.xl_res_mask

		# Convert all satisfied XLs to 1.
		mask_satisfy = ( self.D != 0 ) & ( self.D <= self.xl_max_bound )
		self.D[mask_satisfy] = 1

		# Convert all violated XLs to 0.
		mask_viol = self.D > self.xl_max_bound
		self.D[mask_viol] = 0


	def compute_xl_satisfaction( self ) -> float:

		"""
		Calculate the XL satisfaction as the percentage of XLs satisfied.
		"""
		# Identify Xl violations.
		# m = self.D != 0
		# violated = torch.sum( self.D > self.xl_max_bound )
		satisfied = torch.sum( self.D == 1 )
		# xl_satisfaction = 1 - ( violated/ self.total_xls )
		xl_satisfaction = satisfied/ self.total_xls

		return xl_satisfaction



	def track_satisfied_xls( self ) -> None:
		"""
		Keep track of all satisfied XLs.
		"""
		self.satisfied_xls += self.D



class Metrics():
	def __init__( self, config: mlc.ConfigDict, restraint_features: Dict ):
		self.config = config
		self.restraint_features = restraint_features

		self.included_metrics = self.metrics_included()


	def forward( self, out: torch.Tensor, last_epoch: bool ) -> Dict[str, float]:
		# Store results for all metrics.
		scalar_metric_dict = {}
		other_metric_dict = {}

		for obj in self.included_metrics:
			name = obj.name
			value = obj.forward( out )
			scalar_metric_dict[name] = value

			if last_epoch and name == "xlr":
				other_metric_dict[name] = obj.satisfied_xls

		return scalar_metric_dict, other_metric_dict



	def metrics_included( self ) -> List:
		"""
		Metrics to be calculated.
		"""
		included_metrics = []
		if self.config.xlr.enabled:
			included_metrics.append( XlMetrics( self.config.xlr, self.restraint_features["xl_restraint"] ) )

		return included_metrics


