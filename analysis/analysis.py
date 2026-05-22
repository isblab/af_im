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
from unique_models import UniqueStructures
from molprobity import Molprobity
from variability import EnsembleVariability
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
		model: str,
		# xl_type: str
		):
		self.config_dict = get_config_dict()
		self.base_dir = os.path.join(
			os.path.abspath( self.config_dict.models.base_dir )
			)
		self.benchmark_name = self.config_dict.benchmark.globals.benchmark_name

		self.model = model
		self.cpu_cores = 100
		self.save_every = 1
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
		print( f"Running analysis pipeline for {self.model}...\n" + "-"*50  )
		self.create_required_files()
		self.create_required_dir()
		self.init_logs()
		self.load_benchmark()
		self.get_sys_pred_metadata()

		print( f"Will save results every {self.save_every} iterations..." )

		self.get_residues_to_sys_index_mapping()
		self.create_native_sys_chain_mapping()

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
			f"tmp_struct_sim_{self.model}"
		)
		self.dockq_tmp_dir_path = os.path.join(
			self.analysis_dir,
			f"tmp_dockq_{self.model}"
		)
		self.molprob_tmp_dir_path = os.path.join(
			self.analysis_dir,
			f"tmp_molprob_{self.model}"
		)

		self.per_config_logs_file = os.path.join(
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

		# if os.path.exists( self.cross_config_logs_file ):
		# 	self.cross_config_logs = np.load(
		# 		self.cross_config_logs_file, allow_pickle = True
		# 	).item()
		# else:
		# 	self.cross_config_logs = {}


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

		config_names = [
			c for c in self.model_config if len( self.model_config[c] ) != 0
		]
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
		For config==alpha -> unguided prediction
			We will use the short linker xl_max_bound for evaluation.

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
			xl_max_bound = self.config_dict.benchmark.jwalk.short_linker
		elif self.model_config[config_name]["xl_type"] == "short":
			xl_max_bound = self.config_dict.benchmark.jwalk.short_linker
		elif self.model_config[config_name]["xl_type"] == "long":
			xl_max_bound = self.config_dict.benchmark.jwalk.long_linker
		else:
			raise ValueError( f"Invalid XL type: " +
				f"{self.model_config[config_name]['xl_type']} specified..."
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
		For config==alpha -> unguided prediction
			We will use the short linker xl_max_bound for evaluation.

		self.pred_metadata: {
			sys_name: {
				model_ids: List[int],
				model_files: List[str]
			}
		}
		"""
		for config_name in self.return_config_names( model = self.model ):
			if self.model_config[config_name]["xl_type"] is None:
				xl_type = "short"
			else:
				xl_type = self.model_config[config_name]["xl_type"]

			# self.xl_type = self.model_config[config_name]["xl_type"]
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
					"model_files": out_files[0],
					"xl_type": xl_type,
					"frac_fp": self.model_config[config_name]["frac_fp"]
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
		In the benchmark csv auth_asym_ids, are stored as comma-separated string:
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

			self.native_sys_chain_map[sys_name] = dict( zip( native_chains, sys_chains ) )

	################################################################################
	################################################################################
	def create_xl_gt_features(
		self,
		model_key: str
		):
		"""
		Wrapper for creating XL ground truth features for all systems in
			the benchmark.
		"""
		self.xl_res_dict = {}

		for sys_name in self.benchmark["PDB ID"]:
			xl_dict = self.create_xl_gt_features_per_sys(
				sys_name = sys_name,
				xl_type = self.pred_metadata[model_key][sys_name]["xl_type"],
				frac_fp = self.pred_metadata[model_key][sys_name]["frac_fp"]
			)
			self.xl_res_dict[sys_name] = xl_dict


	def create_xl_gt_features_per_sys( self,
		sys_name: str,
		xl_type: str,
		frac_fp: float
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
			sys_name = sys_name,
			xl_type = xl_type,
			frac_fp = frac_fp
		)
		# Sanity check: at this stage XLs must exist.
		if any( [len( v ) == 0 for k, v in xl_flat_amb_dict.items()] ):
			raise ValueError( f"No XLs in xl_amb_flat_dict..." )
		xl_amb_dict = self.merge_ambiguous_xls(
			xl_flat_amb_dict = xl_flat_amb_dict
		)
		xl_dict = self.map_residues_to_sys_indices(
			sys_name = sys_name,
			xl_amb_dict = xl_amb_dict
		)
		return xl_dict


	def get_all_ambiguous_xls_per_sys( self,
		sys_name: str,
		xl_type: str,
		frac_fp: float
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
			"entity_id2", "chain_id2", "residue2",
			"label"
			]}
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = xl_type,
			frac_fp = frac_fp
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
				chain_id2, res1, res2,
					label ) = row
			xl_flat_amb_dict["entity_id1"].append( entity_id1 )
			xl_flat_amb_dict["entity_id2"].append( entity_id2 )
			xl_flat_amb_dict["chain_id1"].append( chain_id1 )
			xl_flat_amb_dict["chain_id2"].append( chain_id2 )
			xl_flat_amb_dict["residue1"].append( res1 )
			xl_flat_amb_dict["residue2"].append( res2 )
			xl_flat_amb_dict["label"].append( label )
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
					residue2: np.ndarray,
					label: np.ndarray
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
			xl_dict[xl_group_idx] = {k:[] for k in ["residue1", "residue2", "label"]}
			for i in range( len( xl_amb_dict[xl_group_idx]["entity_id1"] ) ):
				label = xl_amb_dict[xl_group_idx]["label"][i]
				chain_id1 = xl_amb_dict[xl_group_idx]["chain_id1"][i]
				res1 = xl_amb_dict[xl_group_idx]["residue1"][i]
				chain_id2 = xl_amb_dict[xl_group_idx]["chain_id2"][i]
				res2 = xl_amb_dict[xl_group_idx]["residue2"][i]

				sys_ind1 = mapping[chain_id1]["res_to_ind"][res1]
				sys_ind2 = mapping[chain_id2]["res_to_ind"][res2]
				xl_dict[xl_group_idx]["residue1"].append( sys_ind1 )
				xl_dict[xl_group_idx]["residue2"].append( sys_ind2 )
				xl_dict[xl_group_idx]["label"].append( label )
		return xl_dict

	################################################################################
	################################################################################
	def run_analysis_per_config( self ):
		"""
		Measure the following:
			Data satisfaction
			Structural similarity across all predicted model
			TM-score wrt native
			DockQ wrt native
			Select unique models based on structural similarity
			Molprobity
			Per-residue RMSF (also obtain the per-residue pLDDT)
		"""
		for idx, model_key in enumerate( self.pred_metadata ):
			print( f"\nRunning analysis for {model_key}..." )
			_, config_name = model_key.split( "_" )
			if model_key not in self.per_config_logs:
				self.per_config_logs[model_key] = {}
			else:
				if len( self.per_config_logs ) == len( self.benchmark["PDB ID"] ):
					print( "Already completed..." )
					continue
			xl_max_bound = self.get_xl_max_bound( config_name = config_name )
			self.create_xl_gt_features( model_key = model_key )

			total = len( self.pred_metadata[model_key] )
			for i, sys_name in enumerate( self.pred_metadata[model_key] ):
				if sys_name not in self.per_config_logs[model_key]:
					self.per_config_logs[model_key][sys_name] = {}

				print( f"\n{i}/{total}. System: {sys_name} " + "-"*20 )
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
					print( "Computing XL satisfaction..." )
					xl_metrics = self.run_data_satisfaction_calc_per_sys(
						sys_name = sys_name,
						model_ids = model_ids,
						model_files = model_files,
						xl_max_bound = xl_max_bound
					)
					self.per_config_logs[model_key][sys_name]["xl_metrics"] = xl_metrics

				if "struct_similarity" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing structural similarity..." )
					similarity_dict = self.run_struct_similarity_calc_per_sys(
						model_ids1 = model_ids,
						model_ids2 = model_ids,
						model_files1 = model_files,
						model_files2 = model_files
					)
					self.per_config_logs[model_key][sys_name]["struct_similarity"] = similarity_dict

				if "tm" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing structural similarity wrt native structure..." )
					dock_dict = self.run_struct_similarity_calc_per_sys(
						model_ids1 = model_ids,
						model_ids2 = [1000],
						model_files1 = model_files,
						model_files2 = [native_file]
					)
					self.per_config_logs[model_key][sys_name]["tm"]  = dock_dict

				if "interface_similarity" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing interface similarity..." )
					dock_dict = None
					dock_dict = self.run_dockq_calc_per_sys(
						model_ids1 = model_ids,
						model_ids2 = model_ids,
						model_files1 = model_files,
						model_files2 = model_files,
						native_sys_chain_map = native_sys_chain_map,
						use_native_chains_for_model2 = False
					)
					self.per_config_logs[model_key][sys_name]["interface_similarity"]  = dock_dict

				if "dockq" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing interface similarity wrt native structure..." )
					dock_dict = None
					dock_dict = self.run_dockq_calc_per_sys(
						model_ids1 = model_ids,
						model_ids2 = [1000],
						model_files1 = model_files,
						model_files2 = [native_file],
						native_sys_chain_map = native_sys_chain_map,
						use_native_chains_for_model2 = True
					)
					self.per_config_logs[model_key][sys_name]["dockq"] = dock_dict

				if "molprob" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing Molprobity metrics..." )
					molprob_dict = self.run_molprobity_calc_per_sys(
						model_ids = model_ids,
						model_files = model_files
					)
					self.per_config_logs[model_key][sys_name]["molprob"] = molprob_dict

				if "molprob_native" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing Molprobity metrics for native structure..." )
					molprob_dict = self.run_molprobity_calc_per_sys(
						model_ids = [1000],
						model_files = [native_file]
					)
					self.per_config_logs[model_key][sys_name]["molprob_native"] = molprob_dict

				if "unique_struct" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing no. of models with unique structure..." )
					similarity_dict = self.per_config_logs[model_key][sys_name]["struct_similarity"]
					xl_metrics = self.per_config_logs[model_key][sys_name]["xl_metrics"]
					tm_dict = self.per_config_logs[model_key][sys_name]["tm"]
					dock_dict = self.per_config_logs[model_key][sys_name]["dockq"]
					molprob_dict = self.per_config_logs[model_key][sys_name]["molprob"]

					unique_models = self.get_unique_models_per_sys(
						metric_dict = similarity_dict,
						metric_name = "tm",
						threshold = 0.7,
						xl_metrics = xl_metrics,
						tm_dict = tm_dict,
						dock_dict = dock_dict,
						molprob_dict = molprob_dict,
						model_files = model_files
					)
					self.per_config_logs[model_key][sys_name]["unique_struct"] = unique_models

				if "unique_interface" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing no. of models with unique interface..." )
					interface_dict = self.per_config_logs[model_key][sys_name]["interface_similarity"]
					xl_metrics = self.per_config_logs[model_key][sys_name]["xl_metrics"]
					molprob_dict = self.per_config_logs[model_key][sys_name]["molprob"]
					tm_dict = self.per_config_logs[model_key][sys_name]["tm"]
					dock_dict = self.per_config_logs[model_key][sys_name]["dockq"]
					unique_models = self.get_unique_models_per_sys(
						metric_dict = interface_dict,
						metric_name = "dockq",
						threshold = 0.8,
						xl_metrics = xl_metrics,
						tm_dict = tm_dict,
						dock_dict = dock_dict,
						molprob_dict = molprob_dict,
						model_files = model_files
					)
					self.per_config_logs[model_key][sys_name]["unique_interface"] = unique_models

				if "rmsf" not in self.per_config_logs[model_key][sys_name]:
					print( "Computing per-residue RMSF..." )
					rmsf_dict = self.run_rmsf_calc_per_sys(
						model_ids = model_ids,
						model_files = model_files
					)
					self.per_config_logs[model_key][sys_name]["rmsf"] = rmsf_dict

				t_e = time.perf_counter()
				time_taken = t_e - t_s

				if "time_taken" not in self.per_config_logs[model_key][sys_name]:
					self.per_config_logs[model_key][sys_name]["time_taken"] = time_taken
				# print( f"Time taken for {sys_name} = {time_taken/60} minutes..." )
				# if idx%self.save_every == 0:
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
		model_ids1: List[int],
		model_ids2: List[int],
		model_files1: List[str],
		model_files2: List[str]
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
			model_ids1 = model_ids1,
			model_files1 = model_files1,
			model_ids2 = model_ids2,
			model_files2 = model_files2,
			tmp_dir_path = self.struct_sim_tmp_dir_path,
			cpu_cores = self.cpu_cores
		)
		similarity_dict = sim_obj.forward()

		return similarity_dict

	################################################################################
	def get_unique_models_per_sys(
		self,
		metric_dict: Dict[int, Dict[int, Dict]],
		metric_name: str,
		threshold: float,
		xl_metrics: Dict[str, Any],
		tm_dict: Dict[int, Dict[int, Dict]],
		dock_dict: Dict[int, Dict[int, Dict]],
		molprob_dict: Dict[str, float],
		model_files: List[str]
	):
		"""
		Identify unique models based on the specified structural similarity
			metric (TM-score, DockQ).
		"""
		unique_models = {
			k: [] for k in ["model_id", "xl_satisfaction", "tm", "dockq", "struct_file", "molprob"]
		}
		representatives = UniqueStructures(
			metric_dict = metric_dict,
			metric_name = metric_name,
			threshold = threshold
		).forward()
		# Fraction of XLs satisfied per model.
		xl_satisfied = xl_metrics["xl_satisfaction"]
		for rep_model_id in representatives:
			unique_models["model_id"].append( rep_model_id )
			unique_models["xl_satisfaction"].append( xl_satisfied[rep_model_id] )
			unique_models["tm"].append( tm_dict[rep_model_id] )
			unique_models["dockq"].append( dock_dict[rep_model_id] )
			unique_models["molprob"].append( molprob_dict[rep_model_id] )
			unique_models["struct_file"].append( model_files[rep_model_id] )
		return unique_models

	################################################################################
	def run_dockq_calc_per_sys(
		self,
		model_ids1: List[int],
		model_ids2: List[int],
		model_files1: List[str],
		model_files2: List[str],
		native_sys_chain_map: Dict[str, str],
		use_native_chains_for_model2: bool
	):
		"""
		For a given system, compute the DockQ for all models wrt the
			native structure.
		For the native model, a random model_id is given - 1000.

		Inputs:
		----------
		model_ids: a list of integer identifiers for a model.
		model_files: a list of structure file paths for all experiemntal/predicted models.
		native_file: file path for the native structure of the system.
		native_sys_chain_map: dict containing mapping between the native
			and system chain IDs.

		Returns:
		----------
		dock_dict: dict contaiing DockQ of all models wrt the native structure.
		{
			model_id1: {
				model_id2: dockq
			}
		}
		"""
		dockq_obj = DockQ(
			model_ids1 = model_ids1,
			model_files1 = model_files1,
			model_ids2 = model_ids2,
			model_files2 = model_files2,
			native_sys_chain_map = native_sys_chain_map,
			use_native_chains_for_model2 = use_native_chains_for_model2,
			tmp_dir_path = self.dockq_tmp_dir_path,
			cpu_cores = self.cpu_cores
		)
		dockq_dict = dockq_obj.forward()
		return dockq_dict

	################################################################################
	def run_molprobity_calc_per_sys(
		self,
		model_ids: List[int],
		model_files: List[str]
	):
		"""
		For a given system, run Molprobity validation for models.

		Inputs:
		----------
		model_ids: a list of integer identifiers for a model.
		model_files: a list of file paths for the predicted model.
		native_file: file path for the native structure of the system.
		native_sys_chain_map: dict containing mapping between the native
			and system chain IDs.

		Returns:
		----------
		molprob_dict: dict containng various MolProbity metrics.
		{
			model_id: {
				molrpobity metric name: value
			}
		}
		"""
		molprob_obj = Molprobity(
			model_ids = model_ids,
			model_files = model_files,
			tmp_dir_path = self.molprob_tmp_dir_path,
			cpu_cores = self.cpu_cores
		)
		molprob_dict = molprob_obj.forward()
		return molprob_dict

	################################################################################
	def run_rmsf_calc_per_sys(
		self,
		model_ids: List[int],
		model_files: List[str]
	):
		"""
		For a given system, obtain the per-residue RMSF across all models.

		Inputs:
		----------
		model_ids: a list of integer identifiers for a model.
		model_files: a list of file paths for the predicted model.

		Returns:
		----------
		rmsf_dict: dict containng various per-residue RMSF and pLDDT.
		{
			model_id: {
				molrpobity metric name: value
			}
		}
		"""
		ensem_obj = EnsembleVariability(
			model_ids = model_ids,
			model_files = model_files,
			tmp_dir_path = self.molprob_tmp_dir_path,
			cpu_cores = self.cpu_cores
		)
		rmsf_dict = ensem_obj.forward()
		return rmsf_dict

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description = "Run analysis pipeline for all model predictions."
	)
	parser.add_argument(
		"-m", "--model",
		type = str, required = True,
		help = "Specify the model to use: grasp/alphalink2/boltz2." )

	args = parser.parse_args()
	Analysis( model = args.model ).forward()
