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
		# A dict containing all ambiguous pairs for each cross-linked residue pair.
		self.xl_res_dict = restraint_features["xl_res_dict"]
		# Total XL pairs.
		self.total_xls = restraint_features["total_xls"]

		# Used for keeping track of all satisfied XLs across all epochs.
		self.satisfied_xls = torch.zeros( self.total_xls )


	def forward( self, out: Dict[str, torch.Tensor] ):
		self.get_predicted_distance_map( out )
		xl_satisfaction = self.get_satisfied_xl_pairs()
		xl_metric = self.compute_xl_metric( xl_satisfaction )
		self.compute_global_xl_satisfaction( xl_satisfaction )

		# self.apply_xl_mask()
		# xl_satisfaction = self.compute_xl_satisfaction()
		# self.track_satisfied_xls()

		return xl_metric


	def get_predicted_distance_map( self, out: Dict[str, torch.Tensor]
									) -> None:
		"""
		Get the predicted distance map.
		"""
		self.D = pred_to_dist_map( out["final_atom_positions"], 
									self.xl_max_bound )

		# Adjust the length scales.
		self.D = self.D / self.length_scale


	def get_satisfied_xl_pairs( self ):
		"""
		For each XL pair, compute if the predicted distance
			is within xl_max_bound.
		"""
		xl_satisfaction = []
		for xl_pair in self.xl_res_dict:
			res_idx1 = self.xl_res_dict[xl_pair]["res1"]
			res_idx2 = self.xl_res_dict[xl_pair]["res2"]
			xl_indices = ( 0, res_idx1, res_idx2 )

			min_D = torch.min( self.D[xl_indices] )

			violated = int( min_D > self.xl_max_bound )
			xl_satisfaction.append( 1 - violated )

		return torch.tensor( xl_satisfaction )


	def compute_xl_metric( self, xl_satisfaction: torch.Tensor ):
		"""
		Compute xl_metric as the fraction of satisfied XLs.
		"""
		xl_metric = torch.sum( xl_satisfaction )/ self.total_xls
		return xl_metric


	def compute_global_xl_satisfaction( self, xl_satisfaction: torch.Tensor ):
		"""
		Keep track of all satisfied/violated XLs.
		"""
		self.satisfied_xls += xl_satisfaction


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
			included_metrics.append(
				XlMetrics( self.config.xlr,
							self.restraint_features["xl_restraint"] )
			)

		return included_metrics
