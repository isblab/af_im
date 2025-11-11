"""
Contains modules to compute data satisfaction metrics.
"""
from typing import List, Dict
import ml_collections as mlc
import numpy as np

import torch

from utils.metric_utils import final_pred_to_dist_map



class XlMetrics():
	def __init__( self, config: Dict, restraint_features: Dict ):
		"""
		Calculate the percentage of XLs satisfied.
		Check what all XLs are satisfied.
		"""
		self.name = "xlr"
		self.length_scale = config.length_scale
		self.eps = config.eps
		self.xl_max_bound = restraint_features["xl_max_bound"]/ self.length_scale
		# A dict containing all ambiguous pairs for each cross-linked residue pair.
		self.xl_res_dict = restraint_features["xl_res_dict"]
		# Total XL pairs.
		self.total_xls = restraint_features["total_xls"]

		# Used for keeping track of all satisfied XLs across all epochs.
		self.xl_satisfaction_array = np.array( [] )


	def forward( self, out: Dict[str, torch.Tensor] ):
		self.get_predicted_distance_map( out )
		xl_satisfaction = self.get_satisfied_xl_pairs()

		# All XL residue pairs must be accounted for.
		if xl_satisfaction.shape[0] != self.total_xls:
			raise ValueError( f"No. of XLs accounted for ({xl_satisfaction.shape[0]}) " +
								f"does not match the total no. of XLs ({self.total_xls})...")

		xl_metric = self.compute_xl_metric( xl_satisfaction )
		self.stack_per_model_xl_satisfaction( xl_satisfaction )

		return xl_metric


	def get_predicted_distance_map( self, out: Dict[str, torch.Tensor]
									) -> None:
		"""
		Get the predicted distance map.
		"""
		self.D = final_pred_to_dist_map(
			final_atom_pos = out["final_atom_positions"],
			length_scale = self.length_scale,
			eps = self.eps
		)


	def get_satisfied_xl_pairs( self ):
		"""
		For each XL pair, compute if the predicted distance
			is within xl_max_bound.
		"""
		xl_satisfaction = []
		for xl_pair in self.xl_res_dict:
			res_idx1 = self.xl_res_dict[xl_pair]["res1"]
			res_idx2 = self.xl_res_dict[xl_pair]["res2"]

			# [B, N, N]
			min_D = torch.min( self.D[0, res_idx1, res_idx2] )

			violated = int( min_D > self.xl_max_bound )
			xl_satisfaction.append( 1 - violated )

		return torch.tensor( xl_satisfaction )


	def compute_xl_metric( self, xl_satisfaction: torch.Tensor ):
		"""
		Compute xl_metric as the fraction of satisfied XLs.
		"""
		xl_metric = torch.sum( xl_satisfaction )/ self.total_xls
		return xl_metric


	def stack_per_model_xl_satisfaction( self, xl_satisfaction: torch.Tensor ):
		"""
		Keep track of all satisfied XLs.
		Stack, the xl_staisfaction array for all epochs.
		"""
		if self.xl_satisfaction_array.shape[0] == 0:
			self.xl_satisfaction_array = xl_satisfaction
		else:
			self.xl_satisfaction_array = np.vstack( [self.xl_satisfaction_array, xl_satisfaction.numpy()] )



	def compute_global_xl_satisfaction( self ):
		"""
		Global XL satisfaction denotes the fraction of XLs satisfied across all models.
		"""
		print( f"XL satisfaction array: {self.xl_satisfaction_array.shape} \t Total XLs = {self.total_xls}" )
		ensemble_satisfaction = np.sum( self.xl_satisfaction_array, axis = 0 )
		total_satisfied = np.count_nonzero( ensemble_satisfaction )
		global_xl_satisfaction = total_satisfied/self.total_xls
		return global_xl_satisfaction


class Metrics():
	def __init__( self, config: mlc.ConfigDict, restraint_features: Dict ):
		self.config = config
		self.restraint_features = restraint_features

		self.included_metrics = self.metrics_included()

		# Dict to store metadata for all metrics.
		self.metric_metadata_dict = {}


	def forward( self, out: torch.Tensor, last_epoch: bool ) -> Dict[str, float]:
		# Store per epoch results for all metrics.
		metrics_dict = {}

		for obj in self.included_metrics:
			metric_name = obj.name
			value = obj.forward( out )
			metrics_dict[metric_name] = value

			if last_epoch and metric_name == "xlr":
				self.store_metadata( metric_name = metric_name,
									metric_instance = obj )

		return metrics_dict


	def store_metadata( self, metric_name: str, metric_instance ):
		"""
		Store metadata for all metrics.
		XL data:
			XL satisfaction array for all models.
			Global XL satisfaction.
		"""
		self.metric_metadata_dict[metric_name] = {
		"global_satisfaction": metric_instance.compute_global_xl_satisfaction(),
		"xl_satisfaction_array": metric_instance.xl_satisfaction_array
		}
		print("Here")



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

