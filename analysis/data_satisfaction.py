"""
Contains module for computing data satisfcation for predicted structures.
	Supports cross-linking (XL) data.
"""
from typing import List, Dict
import numpy as np

from utils.pdb_utils import Parser

def map_chain_residues_to_system_indices():
	"""
	Given a system, 
	"""


class XlSatisfaction():
	"""
	Compute XL satisfaction for the given set
		of predicted structures.
	"""
	def __init__(
		self,
		model_ids: List[int],
		model_files: List[str],
		xl_dict: str,
		xl_max_bound: float
	):
		self.model_ids = model_ids
		self.model_files = model_files
		self.xl_dict = xl_dict
		self.xl_max_bound = xl_max_bound


	def forward( self ):
		"""
		Given a set of model_ids and the corresponding
			structure file,
			- Extract coordinates from the predicted model file.
			- Create distance map.
			- Select the distances for the cross-linked residues.
			- Compute XL metrics.
		"""
		xl_metrics = {}
		struct_dict = self.parse_struct()
		dist_dict = self.compute_distance_matrix( struct_dict = struct_dict )
		xl_dists = self.subset_xl_distances( dist_dict = dist_dict )
		xl_metadata = self.compute_xl_metadata_per_model( xl_dists = xl_dists )
		xl_metrics = self.compute_xl_metrics( xl_metadata = xl_metadata )
		return xl_metrics

	################################################################################
	################################################################################
	def parse_struct( self ) -> Dict[int, Dict[str, np.ndarray]]:
		"""
		Given a set of files corresponding to the predicted structure,
			Parse the file.
			Extract the coordinates and make a np.ndarray.
		Each model file contains a single model.

		Returns:
		----------
		struct_dict: dict containing the coords and plddt for each model_id.
		{
			model_id: {
				coords: np.ndarray,   [N, 3]
				plddt: np.ndarray,    [N 1]
			}
		}
		"""
		struct_dict = {}
		for model_id, file in zip( self.model_ids, self.model_files ):
			coords, plddt = [], []
			p = Parser( pdb_file = file )
			for model in p.get_model_ids():
				for residue, chain_id in self.get_residues_from_model( model ):
					coords.append( self.extract_perresidue_quantity( residue, "coords" ) )
					plddt.append( self.extract_perresidue_quantity( residue, "plddt" ) )
			coords = np.array( coords ).reshape( -1, 3 )
			plddt = np.array( plddt ).reshape( -1, 1 )
			struct_dict[model_id] = {
				"coords": coords,
				"plddt": plddt
			}
		return struct_dict

	################################################################################
	def compute_distance_matrix( self,
		struct_dict: Dict[int, Dict[str, np.ndarray]]
	) -> Dict[str, np.ndarray]:
		"""
		Compute a NxN distance matrix for all models, given
			the corresponding coordinates.
			N -> no. of residues in the system.
		Assumption: self.metadata has been created and contains the
			coordinates.

		Inputs:
		----------
		struct_dict: dict containing the coords and plddt for each model_id.

		Returns:
		----------
		dist_dict: dict containing the distance map for each model_id.
		{
			model_id: {
				coords: np.ndarray,   [N, 3]
				plddt: np.ndarray,    [N 1]
				dist_mat: np.ndarray  [N, N]
			}
		}
		"""
		dist_dict = {}
		for model_id in struct_dict:
			# [N, 3]
			coords = struct_dict[model_id]["coords"]
			# [N, N]
			dist_mat = coords[:, None, :] - coords[None, :, :]
			dist_dict[model_id]["dist_mat"] = dist_mat
		return dist_dict

	################################################################################
	def subset_xl_distances( self,
		dist_dict: Dict[int, np.ndarray]
	):
		"""
		For each distance matrix across all predicted model,
			Select the subset of distances corresponding to the
				XL'd residue pairs.

		Inputs:
		----------
		dist_dict: dict containing the distance map for each model_id.

		Returns:
		----------
		xl_dists: a dict containing Xl distances for each pair identified by an xl_idx.
			For ambiguous XLs, it contains distances for all ambiguous
				residue pairs.
		{
			xl_idx: np.ndarray
		}
		Updates the self.metadata dict.
		"""
		xl_dists = {}
		for model_id in dist_dict:
			dist_mat = dist_dict[model_id]["dist_mat"]
			xl_dists[model_id] = {}
			for xl_idx in self.xl_dict:
				res1 = self.xl_dict[xl_idx]["residue1"]
				res2 = self.xl_dict[xl_idx]["residue2"]
				xl_dists[model_id][xl_idx] = dist_mat[res1, res2]
		return xl_dists

	################################################################################
	def compute_xl_metadata( self,
		xl_dists: Dict[int, np.ndarray]
	):
		"""
		Given the distance map cooresponding to a model_ids,
			compute XL metadata for XL metric.
			- xl_satisfied: Binary label for XLs satisfaction/violation.
				For ambiguous XLs atleast 1 XL pair must be satisfied.
			- xl_min_dist: min distance across all ambiguous pairs.
			- xl_avg_dist: avg distance across all ambiguous pairs.

		Inputs:
		----------
		xl_dists: a dict containing Xl distances for each pair identified by an xl_idx.

		Returns:
		----------
		xl_metadata: dict containing arrays that store the metadata across all models.
		{
			xl_satisfied: np.ndarray,  -> [M, T]
			xl_min_dist: np.ndarray,   -> [M, T]
			xl_avg_dist: np.ndarray    -> [M, T]
		}
			where M -> no. of models and T -> no. of XLs.
		"""
		metadata = {k:[] for k in ["xl_satisfied", "xl_min_dist", "xl_avg_dist"]}
		for model_id in xl_dists:
			xl_satisfied, xl_min_dist, xl_avg_dist = [], [], []
			xl_dists = xl_dists[model_id]["xl_dists"]
			for xl_idx in xl_dists:
				dist = xl_dists[xl_idx]

				min_D = dist.min()
				avg_D = ( dist ).sum()/len( dist )
				xl_satisfied.append( int( min_D <= self.xl_max_bound ) )
				xl_min_dist.append( min_D )
				xl_avg_dist.append( avg_D )
			metadata["xl_satisfied"].append( xl_satisfied )
			metadata["xl_min_dist"].append( xl_min_dist )
			metadata["xl_avg_dist"].append( xl_avg_dist )

		xl_metadata = {
			"xl_satisfied": np.array( metadata["xl_satisfied"] ),
			"xl_min_dist": np.array( metadata["xl_min_dist"] ),
			"xl_avg_dist": np.array( metadata["xl_avg_dist"] )
		}
		return xl_metadata

	################################################################################
	def compute_xl_metrics( self,
		xl_metadata: Dict[int, np.ndarray]
	):
		"""
		Compute the following XL metrics:
			- xl_sat: Fraction of satisfied XLs.
				For ambiguous XLs atleast 1 XL pair must be satisfied.
			- xl_sat_global: XL satisfaction across all models.
			- xl_pair_sat: No. of times an XL pair is satisfied across all models.
			- Distribution of XL distance across all models.
				Min. over all ambiguous XLs.
				Avg. over all ambiguous XLs.

		Creates a dict containing the XL metrics computed from the given metadata.

		Returns:
		----------
		xl_metrics: dict containing the xl metrics computed across all models.
		{
			xl_satisfaction: np.ndarray, --> [M]
			xl_satisfaction_global: float,
			xl_pair_satisfaction: np.ndarray --> [T]
		}
			where M -> no. of models and T -> no. of XLs.
		"""
		total_xls = xl_metadata["xl_satisfied"].shape[1]
		satisfied = xl_metadata["xl_satisfied"]
		xl_sat = satisfied.sum( axis = 1 )/total_xls

		# Total no. of times an XL pair is satisfied across all models.
		xl_pair_sat = satisfied.sum( axis = 0 )

		# Xl satisfaction across all models.
		xl_sat_global = np.count_nonzero( xl_pair_sat )/total_xls

		xl_metrics = {
			"xl_satisfaction": xl_sat,
			"xl_satisfaction_global": xl_sat_global,
			"xl_pair_satisfaction": xl_pair_sat
		}
		return xl_metrics
		
