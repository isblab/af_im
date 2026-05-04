"""
Contains a wrapper module for performing analysis for the benchmark.
"""
from typing import List, Dict, Any
import os, argparse, json, shutil, time
import pickle as pkl
import numpy as np
import pandas as pd

from config import get_config_dict
from model_configs import BOLTZ, GRASP, ALPHALINK
from data_satisfaction import XlSatisfaction
from rmsd import StructuralSimilarity
from dockq import DockQ
from utils.mappings import (
	yield_restraints,
	get_entity_chain_mapping,
	map_residue_positions_to_system_indices
)
from utils.paths import (
	BASE_DIR,
	get_benchmark_csv_file,
	get_xl_file_path,
	get_native_struct_file,
	get_model_output_dir_path,
	return_model_sys_file,
	get_benchmark_analysis_dir_path
)


class Analysis():
	"""
	Perform analysis for the benchmark.
	"""
	def __init__(
		self,
		# model:str,
		# xl_type: str
		):
		self.config_dict = get_config_dict()
		self.base_dir = os.path.join(
			os.path.abspath( self.config_dict.models.base_dir )
			)
		self.benchmark_name = self.config_dict.benchmark.globals.benchmark_name

		self.model = "boltz2"
		self.cpu_cores = 50
		self.model_config = {}

		# Path to store the metadata for all systems.
		self.pred_metadata = {}
		# Contains native to system chain mapping for all systems.
		self.native_sys_chain_map = {}
		# Contains system index and residue position mapping.
		self.sys_index_res_pos_map = {}
		# Contains XLs with system indices for all benchmark systems.
		self.xl_res_dict = {}


	def forward( self ):
		"""
		Create an output dir for storing analysis results.
		Create file_paths for all predicted structures.
		Compute evaluation metrics:
			Data satisfaction.
			DockQ.
			No. of Unique structures.
		"""
		self.create_required_files()
		self.create_required_dir()
		self.init_logs()
		self.load_benchmark()
		self.get_sys_pred_metadata()

		self.get_residues_to_sys_index_mapping()
		self.create_native_sys_chain_mapping()

		self.create_xl_gt_features()

		self.run_analysis_per_config()


	def create_required_files( self ):
		"""
		Create the required file paths.
		"""
		self.analysis_dir = get_benchmark_analysis_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name
		)
		self.struct_sim_tmp_dir_path = os.path.join(
			self.analysis_dir,
			"tmp_struct_sim"
		)
		self.dockq_tmp_dir_path = os.path.join(
			self.analysis_dir,
			"tmp_dockq"
		)

		self.per_config_logs_file = os.path.join(
			self.analysis_dir,
			f"Logs_{self.benchmark_name}_{self.model}.npy"
		)
		self.cross_config_logs_file = os.path.join(
			self.analysis_dir,
			f"Logs_{self.benchmark_name}_{self.model}.npy"
		)


	def create_required_dir( self ):
		"""
		Create the required directories.
		"""
		for dir_ in [self.analysis_dir]:
			os.makedirs( dir_, exist_ok = True )


	def init_logs( self ):
		"""
		Initialize the log dict or load a pre-existing one.
		"""
		if os.path.exists( self.per_config_logs_file ):
			self.per_config_logs = np.load(
				self.per_config_logs_file, allow_pickle = True
			).item()
		else:
			self.per_config_logs = {}

		if os.path.exists( self.cross_config_logs_file ):
			self.cross_config_logs = np.load(
				self.cross_config_logs_file, allow_pickle = True
			).item()
		else:
			self.cross_config_logs = {}


	def return_config_names( self, model: str ) -> List[str]:
		"""
		Return the configs for the specified model.
		See model_configs.py for the existing configs.
		Exclude configs that are undefined.

		Inputs:
		----------
		model: identifier for the model being used: alphalInk2/grasp/boltz2

		Returns:
		----------
		config_names: list of all existing configs for the specified model.
		"""
		if model == "boltz2":
			self.model_config = BOLTZ
		elif model == "grasp":
			self.model_config = GRASP
		elif model == "alphalink2":
			self.model_config = ALPHALINK

		config_names = [c for c in self.model_config if c != {}]
		return config_names

	################################################################################
	################################################################################
	def load_benchmark( self ):
		"""
		Load the benchmark from disk.
		"""
		# Load the benchmark.
		benchmark_file = get_benchmark_csv_file(
			self.base_dir,
			self.benchmark_name,
			raw_file = False
			)
		self.benchmark = pd.read_csv( benchmark_file )


	def get_xl_max_bound( self, config_name: str ) -> float:
		"""
		Set the XL max bound according to the XL type used.

		Inputs:
		----------
		config_name: str identifier for the model configuration
			used for prediction.
			See model_configs.py.

		Returns:
		----------
		xl_max_bound: max-bound for the cross-link type
			specified in the config.
		"""

		if self.model_config[config_name]["xl_type"] is None:
			xl_max_bound = 0.0
		elif self.model_config[config_name]["xl_type"] == "short":
			xl_max_bound = self.config_dict.benchmark.jwalk.short_linker
		elif self.model_config[config_name]["xl_type"] == "long":
			xl_max_bound = self.config_dict.benchmark.jwalk.long_linker
		else:
			raise ValueError( f"Invalid XL type: " +
				f"{self.model_config['xl_type']} specified..."
			)
		return xl_max_bound


	def get_sys_pred_metadata( self ):
		"""
		Across all models, prediction types
			( guided/unguided, shot/long XL )
		For each system we store:
			- model IDs
			- Path to the sructure file.
		model_ids are created from 0 to N_s; where N_s is the no.
			of predicted structures.
		No output files would be returned if the prediction failed.
			e.g. Boltz2 -> 6iww, 7agf

		self.pred_metadata: {
			sys_name: {
				model_ids: List[int],
				model_files: List[str]
			}
		}
		"""
		# for model in ["alphalink2", "grasp", "boltz2"]:
		for config_name in self.return_config_names( model = self.model ):
			# print( self.model_config[config_name].keys() )
			# print( self.model_config.keys() )
			self.xl_type = self.model_config[config_name]["xl_type"]
			model_key = f"{self.model}_{config_name}"
			self.pred_metadata[model_key] = {}

			for sys_name in self.benchmark["PDB ID"]:
				out_files = return_model_sys_file(
					model = self.model,
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					config_name = config_name,
					sys_name = sys_name
				)
				model_ids = np.arange( 0, len( out_files[0] ), 1 )
				self.pred_metadata[model_key][sys_name] = {
					"model_ids": model_ids,
					"model_files": out_files[0]
				}

	################################################################################
	################################################################################
	def get_residues_to_sys_index_mapping( self ):
		"""
		For all systems in the benchmark, obtain the mapping between
			the residue positions and system indices.
		"""
		for sys_name in self.benchmark["PDB ID"]:
			mapping = map_residue_positions_to_system_indices(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name
			)
			self.sys_index_res_pos_map[sys_name] = mapping

	################################################################################
	def create_native_sys_chain_mapping( self ):
		"""
		Map the native chain IDs to the system chain IDs.
		The auth_asym_ids are stored as comma-separated string:
			A:B,C:D
			':' separates multiple instances of an entity.
		
		self.native_sys_chain_map: {
			sys_name: {
				native_chain: sys_chain
			}
		}
		"""
		for i in self.benchmark.index:
			sys_name = self.benchmark.loc[i, "PDB ID"]
			auth_asym_ids = self.benchmark.loc[i, "Auth Asym ID"].split( "," )

			sys_chains = list( self.sys_index_res_pos_map[sys_name].keys() )
			native_chains = []
			# "A:B,C:D" -> ["A:B", "C:D"]
			native_chains.extend( auth_asym_ids )
			# "A:B:C:D" -> [A, B, C, D]
			native_chains = ":".join( native_chains ).split( ":" )

			if sys_name == "8kbh":
				print( auth_asym_ids )
				print( native_chains )
				print( sys_chains )
				# exit()
			self.native_sys_chain_map[sys_name] = dict( zip( native_chains, sys_chains ) )

	################################################################################
	################################################################################
	def create_xl_gt_features( self ):
		"""
		Wrapper for creating XL ground truth features for all systems in
			the benchmark.
		"""

		for sys_name in self.benchmark["PDB ID"]:
			xl_dict = self.create_xl_gt_features_per_sys( sys_name = sys_name )
			self.xl_res_dict[sys_name] = xl_dict


	def create_xl_gt_features_per_sys( self,
		sys_name: str
		):
		"""
		Create a dict containing all the cross-linked residue.
		Each xl_pair is identified by an index (starting from 0).
			Comprises system indices for the given XL'd residues
				for all combinations of ambiguous chains.
		- Get all ambiguous XL pairs.
		- Group all ambiguous pairs for each XL'd residue identified
			by an XL index.
		- Map each residue pair to system index.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.

		Returns:
		----------
		xl_res_dict: {
			xl_pair: {
				[], -> system indices for residues in prot1
				[] -> system indices for residues in prot2
			}
		}
		To account for ambiguity, each xl_pair contains all combinations
			of XL'd residues from ambiguous chain pairs.
		"""
		xl_dict = {}

		xl_flat_amb_dict = self.get_all_ambiguous_xls_per_sys(
			sys_name = sys_name
		)
		# Sanity check: at this stage XLs must exist.
		if any( [len( v ) == 0 for k, v in xl_flat_amb_dict.items()] ):
			raise ValueError( f"No XLs in xl_amb_flat-dict..." )
		xl_amb_dict = self.merge_ambiguous_xls(
			xl_flat_amb_dict = xl_flat_amb_dict
		)
		xl_dict = self.map_residues_to_sys_indices(
			sys_name = sys_name,
			xl_amb_dict = xl_amb_dict
		)
		return xl_dict


	def get_all_ambiguous_xls_per_sys( self,
		sys_name: str
		) -> Dict[str, List]:
		"""
		For the given system,
			Collect all XL'd residue pairs, explicitly expanding ambiguous
				XLs into all possible chain-level combinations.
		yield_restraints() returns all ambiguous residue pairs.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.

		Returns:
		----------
		xl_amb_dict: a flat dict containing XL'd residues (including
			ambiguous pairs).
			{
				entity_id1: [],
				entity_id2: [],
				chain_id1: [],
				chain_id2: [],
				residue1: [],
				residue2: []
			}

		"""
		xl_flat_amb_dict = {k:[] for k in [
			"entity_id1", "chain_id1", "residue1",
			"entity_id2", "chain_id2", "residue2"
			]}
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = self.xl_type
		)
		entity_chain_map = get_entity_chain_mapping(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		for row in yield_restraints(
			sys_name = sys_name,
			xl_file = xl_file,
			entity_chain_map = entity_chain_map,
			numeric_chain_ids = False,
			return_seq_numbering = False ):
			( entity_id1, entity_id2, chain_id1,
				chain_id2, res1, res2 ) = row
			xl_flat_amb_dict["entity_id1"].append( entity_id1 )
			xl_flat_amb_dict["entity_id2"].append( entity_id2 )
			xl_flat_amb_dict["chain_id1"].append( chain_id1 )
			xl_flat_amb_dict["chain_id2"].append( chain_id2 )
			xl_flat_amb_dict["residue1"].append( res1 )
			xl_flat_amb_dict["residue2"].append( res2 )
		return xl_flat_amb_dict


	def merge_ambiguous_xls( self,
		xl_flat_amb_dict: Dict[str, List]
		) -> Dict[int, Dict[str, np.ndarray]]:
		"""
		Group expanded crosslink pairs into ambiguity sets based on residue
			identity.
		Given a flat list of crosslink pairs (as produced by
			`get_all_ambiguous_xls_per_sys`), this function groups entries
			that originate from the same underlying XL but differ due to
			chain ambiguity.
		Grouping is performed by assigning a unique index to consecutive entries
			with identical (residue1, residue2) pairs.
		By construction, entries corresponding to the same ambiguous XL appear
  			consecutively in `xl_flat_amb_dict`.

		Inputs:
		----------
		xl_flat_amb_dict: a flat dict containing XL'd residues (including
			ambiguous pairs).

		Returns:
		----------
		xl_amb_dict: contains ambiguity grouped XLs.
			{
				xl_idx: {
					entity_id1: np.ndarray,
					entity_id2: np.ndarray,
					chain_id1: np.ndarray,
					chain_id2: np.ndarray,
					residue1: np.ndarray,
					residue2: np.ndarray
				}
			}
		"""
		residue1 = xl_flat_amb_dict["residue1"]
		residue2 = xl_flat_amb_dict["residue2"]

		xl_amb_dict = {}
		xl_pairs = []
		prev = ( residue1[0], residue2[0] )
		xl_group_id = 0
		# Add the 0-th pair.
		xl_pairs.append( xl_group_id )

		for r1, r2 in zip( residue1[1:], residue2[1:] ):
			if prev == ( r1, r2 ):
				pass
			else:
				xl_group_id += 1
				prev = ( r1, r2 )
			xl_pairs.append( xl_group_id )
		xl_pairs = np.array( xl_pairs ).reshape( -1 )

		for gid in np.unique( xl_pairs ):
			idx = np.where( xl_pairs == gid )[0]
			xl_amb_dict[gid] = {
				# k: xl_flat_amb_dict[k][idx]
				k: np.array( xl_flat_amb_dict[k] )[idx]
				for k in xl_flat_amb_dict
			}

		return xl_amb_dict


	def map_residues_to_sys_indices( self,
		sys_name: str,
		xl_amb_dict: Dict[int, Dict[str, np.ndarray]]
		):
		"""
		For all XLs, map the residue positions to the
			corresponding system index.
		At this stage, each residue position has a system index mapping.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		xl_amb_dict: contains ambiguity grouped XLs.

		Returns:
		----------
		xl_dict: dict containing ambiguity grouped XLs, where the residue
			positions are mapped to the ssytem index.
		"""
		xl_dict = {}
		mapping = self.sys_index_res_pos_map[sys_name]

		for xl_group_idx in xl_amb_dict:
			xl_dict[xl_group_idx] = {k:[] for k in ["residue1", "residue2"]}
			for i in range( len( xl_amb_dict[xl_group_idx]["entity_id1"] ) ):
				chain_id1 = xl_amb_dict[xl_group_idx]["chain_id1"][i]
				res1 = xl_amb_dict[xl_group_idx]["residue1"][i]
				chain_id2 = xl_amb_dict[xl_group_idx]["chain_id2"][i]
				res2 = xl_amb_dict[xl_group_idx]["residue2"][i]

				sys_ind1 = mapping[chain_id1]["res_to_ind"][res1]
				sys_ind2 = mapping[chain_id2]["res_to_ind"][res2]
				xl_dict[xl_group_idx]["residue1"].append( sys_ind1 )
				xl_dict[xl_group_idx]["residue2"].append( sys_ind2 )
		return xl_dict

	################################################################################
	################################################################################
	def run_analysis_per_config( self ):
		"""
		Measure the following:
			Data satisfaction
			Structural similarity
			Select unique models based on structural similarity
			DockQ wrt native
		"""
		for model_key in self.pred_metadata:
			print( f"\nRunning analysis for {model_key}..." )
			_, config_name = model_key.split( "_" )
			if model_key not in self.per_config_logs:
				self.per_config_logs[model_key] = {}
			xl_max_bound = self.get_xl_max_bound( config_name = config_name )
			remove = False
			for sys_name in self.pred_metadata[model_key]:
				self.per_config_logs[model_key][sys_name] = {}

				print( f"System: {sys_name} " + "-"*20 )
				t_s = time.perf_counter()

				model_ids = self.pred_metadata[model_key][sys_name]["model_ids"]
				model_files = self.pred_metadata[model_key][sys_name]["model_files"]
				native_file = get_native_struct_file(
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					sys_name = sys_name
				)

				native_sys_chain_map = self.native_sys_chain_map[sys_name]

				if len( model_ids ) == 0:
					continue

				if "xl_metrics" not in self.per_config_logs[model_key][sys_name]:
					xl_metrics = self.run_data_satisfaction_calc_per_sys(
						sys_name = sys_name,
						model_ids = model_ids,
						model_files = model_files,
						xl_max_bound = xl_max_bound
					)
					self.per_config_logs[model_key][sys_name]["xl_metrics"] = xl_metrics

				if "struct_similarity" not in self.per_config_logs[model_key][sys_name]:
					similarity_dict = self.run_struct_similarity_calc_per_sys(
						model_ids = model_ids,
						model_files = model_files
					)
					self.per_config_logs[model_key][sys_name]["struct_similarity"] = similarity_dict

				if "unique_models" not in self.per_config_logs[model_key][sys_name]:
					unique_models = self.get_unique_models_per_sys(
						xl_metrics = xl_metrics,
						similarity_dict = similarity_dict,
						model_files = model_files
					)
					self.per_config_logs[model_key][sys_name]["unique_models"] = unique_models

				if "dockq" not in self.per_config_logs[model_key][sys_name]:
					dock_dict = self.run_dockq_calc_per_sys(
						model_ids = model_ids,
						model_files = model_files,
						native_file = native_file,
						native_sys_chain_map = native_sys_chain_map
					)
					self.per_config_logs[model_key][sys_name]["dockq"]  =dock_dict

				t_e = time.perf_counter()
				time_taken = t_e - t_s

				if "time_taken" not in self.per_config_logs[model_key][sys_name]:
					self.per_config_logs[model_key][sys_name]["time_taken"]  =time_taken
				# print( f"Time taken for {sys_name} = {time_taken/60} minutes..." )
				np.save(
					self.per_config_logs_file, self.per_config_logs, allow_pickle = True
					)

	################################################################################
	def run_data_satisfaction_calc_per_sys(
		self,
		sys_name: str,
		model_ids: List[int],
		model_files: List[str],
		xl_max_bound: float
		):
		"""
		Given the predicted models for a system, compute data
			satisfaction for all predicted models.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		model_ids: a list of integer identifiers for a model.
		model_files: a list of file paths for the predicted model.
		xl_max_bound: max-bound for the cross-link type
			specified in the config.

		Returns:
		----------
		xl_metrics: dict containing the XL metrics computed for
			all the given models.
		"""
		xl_sat_obj = XlSatisfaction(
			model_ids = model_ids,
			model_files = model_files,
			xl_dict = self.xl_res_dict[sys_name],
			xl_max_bound = xl_max_bound
		)
		xl_metrics = xl_sat_obj.forward()
		return xl_metrics

	################################################################################
	def run_struct_similarity_calc_per_sys(
		self,
		model_ids: List[int],
		model_files: List[str]
		):
		"""
		For a given system, compute the all-v-all structural similarity.

		Inputs:
		----------
		model_ids: a list of integer identifiers for a model.
		model_files: a list of file paths for the predicted model.

		Returns:
		----------
		"""
		sim_obj = StructuralSimilarity(
			model_ids1 = model_ids,
			model_files1 = model_files,
			model_ids2 = model_ids,
			model_files2 = model_files,
			tmp_dir_path = self.struct_sim_tmp_dir_path,
			cpu_cores = self.cpu_cores
		)
		similarity_dict = sim_obj.forward()

		return similarity_dict

	################################################################################
	def get_unique_models_per_sys(
		self,
		xl_metrics: Dict[str, Any],
		similarity_dict: Dict[int, Dict[int, Dict]],
		model_files: List[str]
	):
		"""
		For a given system, given the xl_metrics and similarity_dict,
			identify unique models (TM-score < 0.7) and obtain
			xl metrics for the subset.
		"""
		unique_models = {
			k: [] for k in ["model_id", "xl_satisfaction", "struct_file"]
		}
		# Fraction of XLs satisfied per model.
		xl_satisfied = xl_metrics["xl_satisfaction"]
		# For all i-th models.
		for k_i in similarity_dict:
			# Select all models except for k_i==k_j.
			sim_ij = np.array(
				[v["tm"] for k_j, v in similarity_dict[k_i].items() if k_i != k_j]
			)
			if np.all( sim_ij < 0.7 ):
				# By construction, model_id and the model index are the same.
				# 	See self.get_sys_pred_metadata()
				unique_models["model_id"].append( k_i )
				unique_models["xl_satisfaction"].append( xl_satisfied[k_i] )
				unique_models["struct_file"].append( model_files[k_i] )
		return unique_models

	################################################################################
	def run_dockq_calc_per_sys(
		self,
		model_ids: List[int],
		model_files: List[str],
		native_file: str,
		native_sys_chain_map: Dict[str, str]
	):
		"""
		For a given system, compute the DockQ for all models wrt the
			native structure.
		For the native model, a random model_id is given - 1000.

		Inputs:
		----------
		model_ids: a list of integer identifiers for a model.
		model_files: a list of file paths for the predicted model.
		native_file: file path for the native structure of the system.
		native_sys_chain_map: dict containing mapping between the native
			and system chain IDs.

		Returns:
		----------
		dock_dict: dict contaiing DockQ of all models wrt the native structure.
		{
			model_id: {
				native_id: dockq
			}
		}
		"""
		dockq_obj = DockQ(
			model_ids1 = model_ids,
			model_files1 = model_files,
			model_ids2 = [1000],
			model_files2 = [native_file],
			native_sys_chain_map = native_sys_chain_map,
			tmp_dir_path = self.dockq_tmp_dir_path,
			cpu_cores = self.cpu_cores
		)
		dockq_dict = dockq_obj.forward()
		return dockq_dict


if __name__ == "__main__":
	Analysis().forward()
	pass

