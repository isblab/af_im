"""
Run the IntegrativeLearning module on the entire benchmark.
Update the topology file with the configs for the simulation.
Check the distribution of violation loss, ccom loss, xl_restraint.
Compute DockQ wrt experimental structure.
Compare experimental structure resolution with MolProbity score.
"""
from typing import List, Dict
import os, glob, time, subprocess, traceback, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import torch
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

from openfold_wrapper import IntegrativeLearning

from topology import topology_dict
from utils.utils import ( open_file_handler, read_json, write_json, run_subprocess )
from utils.pdb_utils import Parser, get_chain_id


class BenchmarkModeling():
	def __init__( self, topo_dict: ConfigDict = None ):
		self.benchmark_name = "experiment"  # xlsim/abag/oreilly/experiment
		# Define the modeling objective.
		self.modeling_objective = "Testing new pose sampling + recycling pipeline."
		self.modeling_version = 0
		# PDB/CIF output for the predicted structure.
		self.struct_format = "pdb"
		# Suffix for modeling with TP+FP XLs.
		self.sys_conf_suff = ""
		# Maximum no. of epochs for fine-tuning.
		self.max_epochs = 100
		# Max epochs for pose sampling.
		self.max_pose_iters = 20
		self.skip_pose_sampling = False
		# If skipping pose sampling, use final_atom_positions=None.
		self.fill_none = False
		# Precision of the float values in the results.
		self.prec = 4
		# Set GPU to use.
		self.device = "cuda:0"
		# Use full or reduced dataset for prediction - full_dbs/reduced_dbs.
		self.db_preset = "full_dbs"
		# If True, skip re-running modeling if output files already exist.
		self.skip_rerun = True
		# Enable analysis.
		self.enable_analysis = False
		# If True run AMBER relaxation and MolProbity validation.
		self.enable_relax_validate = True
		# If True, will not show prompt for existing modeling dir.
		self.disable_overwrite_prompt = True
		# if True, deletes the existing system modeling dir.
		self.remove_sys_modeling_dir = False
		# if True, deletes the existing system analysis dir.
		self.remove_sys_analysis_dir = False
		# If True, create the required plots.
		self.create_summary_plots_and_files = False

		# Modify settings for losses to be used.
		self.violation = {"enabled": True, "add_penalty": True, "weight": 1.0}
		self.xlr = {"enabled": True, "add_penalty": True, "weight": 1.0}

		self.topo_dict = topo_dict

		self.dockq_dict = {}


	def forward( self ):
		"""
		"""
		tic = time.perf_counter()
		print( f"\033[1mModeling version {self.modeling_version} --> {self.modeling_objective}\033[0m\n\n" )
		self.create_required_paths()
		self.create_required_dirs()
		self.load_benchmark()
		self.initialize_logs_dict()

		self.run_modeling_for_benchmark()

		# self.compute_dockq()
		if self.create_summary_plots_and_files:
			self.plot_modeling_results()
			self.write_results_to_csv()
		toc = time.perf_counter()

		# self.write_misc_details( toc-tic )
		self.record_configs( total_time = toc-tic )

		for sys_name in self.logs["errored"]:
			print( f"\033[1mERROR:\033[0m {sys_name} -> {self.logs['errored'][sys_name]}" )

		print( "\nMay the Force be with you..." )

	################################################################################
	################################################################################
	def log_memory_usage( self, sys_name: str ):
		"""
		Log the device memory used during modeling.
		device must be in the following formar: cuda:0
		Reset the CUDA memory stats after logging.
		"""
		if self.device == "cpu":
			raise ValueError( "Pytorch does not provide " +
					"built-in functions to check CPU memory stats. " +
					"Change device to cuda[0/1]..." )
		else:
			device_num = int( self.device[-1] )

		# Get peak memory since last reset.
		max_allocated = torch.cuda.max_memory_allocated( device_num )
		max_reserved = torch.cuda.max_memory_reserved( device_num )
		# Memory used in GB.
		self.logs["memory_allocated"][sys_name] = max_allocated/ (1024**3)
		self.logs["memory_reserved"][sys_name] = max_reserved/ (1024**3)
		# torch.cuda.reset_peak_memory_stats( torch.device( self.device ) )


	def log_error( self, sys_name: str ):
		"""
		If an error occurs while modeling,
			Log the errorneous entry_id in the dataset sepcific metadata dir.
			log the traceback in the system dir.
		"""
		ver_path = self.get_sys_modeling_version_path( sys_name )
		current_datetime = datetime.now()
		timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
		error_file = os.path.join(
			self.benchmark_modeling_dir,
			f"error_{sys_name}_{timestamp}.txt" )
		w = open_file_handler( error_file, "w" )
		w.write( traceback.format_exc() )
		w.close()
		self.logs["errored"][sys_name] = error_file

		print( f"\033[1mAn error occured for system: {sys_name}. " +
				f"Check error log in {error_file}...\033[0m\n" )


	def remove_existing_dir( self, sys_name: str ):
		"""
		Remove the system modeling/analysis dir if specified.
		"""
		if self.remove_sys_modeling_dir:
			modeling_dir_path = self.get_sys_modeling_version_path( sys_name = sys_name )
			if os.path.exists( modeling_dir_path ):
				warnings.warn( f"Deleting modeling dir for system {sys_name} -> {modeling_dir_path}..." )
				print( "Changed your mind? Act now..." )
				time.sleep( 5 )
				cmd = ["rm", "-r", f"{modeling_dir_path}"]
				run_subprocess( command = cmd )

		if self.remove_sys_analysis_dir:
			ver_path = self.get_sys_modeling_version_path( sys_name )
			analysis_dir_path = os.path.join( ver_path, "analysis" )
			warnings.warn( f"Deleting analysis dir for system {sys_name} -> {analysis_dir_path}..." )
			print( "Changed your mind? Acts now..." )
			time.sleep( 5 )
			cmd = ["rm", "-r", f"{analysis_dir_path}"]
			run_subprocess( command = cmd )

	################################################################################
	################################################################################
	def modify_topology( self, sys_name: str ):
		"""
		Add modeling objective and version to topology for each system.
		"""
		if self.topo_dict == None:
			topo_dict = topology_dict()
			topo_dict.objective = f"{sys_name} {self.modeling_objective}"
			topo_dict.train.version = self.modeling_version
			topo_dict.train.device = self.device
			topo_dict.analysis.enabled = self.enable_analysis
			topo_dict.analysis.enable_relax_validate = self.enable_relax_validate
			topo_dict.db_preset = self.db_preset
			topo_dict.train.max_epochs = self.max_epochs
			topo_dict.train.max_pose_iters = self.max_pose_iters
			topo_dict.train.skip_pose_sampling = self.skip_pose_sampling
			topo_dict.train.fill_none = self.fill_none
			topo_dict.train.device = self.device

			# Violation loss settings.
			topo_dict.loss.violation.enabled = self.violation["enabled"]
			topo_dict.loss.violation.add_penalty = self.violation["add_penalty"]
			topo_dict.loss.violation.weight = self.violation["weight"]

			# XL restraint loss settings.
			topo_dict.loss.xlr.enabled = self.xlr["enabled"]
			topo_dict.loss.xlr.add_penalty = self.xlr["add_penalty"]
			topo_dict.loss.xlr.weight = self.xlr["weight"]
		else:
			topo_dict = self.topo_dict

		return topo_dict

	################################################################################
	################################################################################
	def run_modeling_for_benchmark( self ):
		"""
		Run the Integrative modeling pipeline for the entire benchmark.
		"""
		print( "\033[1mRunning modeling for the benchmark...\033[0m" )
		curr_dir = os.getcwd()
		for idx, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( "\n" + "-"*70 + "\n" + "-"*70 )
			print( f"{idx}/{self.num_systems} --> {sys_name}" )
			print( "-"*70 + "\n" + "-"*70 + "\n" )
			stat_file_path = self.get_stat_file_path( sys_name = sys_name )

			# remove system modleing and/or analysis dir if specified.
			self.remove_existing_dir( sys_name = sys_name )

			if os.path.exists( stat_file_path ) and self.skip_rerun:
				print( f"Summary file already present for {sys_name}..." )
			else:
				tic = time.perf_counter()
				# torch.cuda.reset_peak_memory_stats( torch.device( self.device ) )
				self.run_modeling_for_system( sys_name = sys_name )
				toc = time.perf_counter()

				self.logs["time"][sys_name] = toc-tic

				write_json( self.logs, self.logs_file )

			# Log memory used.
			self.log_memory_usage( sys_name = sys_name )
			# Clear cache.
			torch.cuda.empty_cache()

			# Return to base_dir.
			os.chdir( curr_dir )


	def run_modeling_for_system( self, sys_name: str ):
		"""
		Run the Integrative modeling pipeline for a give system.
		If summary file exists fo a run, do not run again.
		Empty the CUDA cache after each run.
		"""
		# Directory containing input data for the modeled system.
		data_dir = self.get_sys_data_dir_path( sys_name = sys_name )

		topo_dict = self.modify_topology( sys_name = sys_name )

		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				sys_config_file =  f"sys_config_{sys_name}{self.sys_conf_suff}.json",
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.disable_overwrite_prompt = self.disable_overwrite_prompt
		il_obj.forward()

	################################################################################
	################################################################################
	def assess_fp_satisfaction( self ):
		"""
		For all selected good-scoring models, check if the FP
			XLs have been satisfied or not.
			Distnace between the FP XL residue pair is below the XL max bound.
		For ambiguous cases, check if any copy satisfies the XL.
		"""
		print( "\n" + "-"*70 +
			"\n\t\t\033[1m--> Assessing FP XL satisfaction <--\033[0m\n" +
			"-"*70 )
		fp_sat_dict = {k:[] for k in ["complexes", "fp_satisfied", "total_fp"]}
		for sys_name in self.benchmark["PDB ID"]:
			analysis_dict = self.load_analysis_dict( sys_name = sys_name )

			auth_asym_ids = self.benchmark[self.benchmark["PDB ID"] == sys_name]["Auth Asym ID"].tolist()[0]
			auth_asym_ids = auth_asym_ids.split( "," )
			sys_chains = self.get_sys_chains( auth_asym_ids = auth_asym_ids )

			per_model_fp_sat = []
			for model_id in analysis_dict["selected_good_models"]:
				# Get the FP XLs.
				fp_xls = self.get_fp_XLs( sys_name = sys_name )
				# Get the modeled residues for all entities.
				modeled_res_dict = self.get_modeled_residues( sys_name = sys_name )
				fp_satisfaction = self.check_fp_xl_satisfaction(
						sys_name = sys_name,
						model_id = model_id,
						sys_chains = sys_chains,
						modeled_res_dict = modeled_res_dict,
						fp_xls = fp_xls )
				if len( per_model_fp_sat ) == 0:
					per_model_fp_sat = fp_satisfaction
				else:
					per_model_fp_sat = fp_satisfaction
			per_model_fp_sat = np.where( per_model_fp_sat > 0, 1, 0 )
			fp_sat_dict["complexes"].append( sys_name )
			fp_sat_dict["fp_satisfied"].append( np.count_nonzero( fp_satisfaction ) )
			fp_sat_dict["total_fp"].append( fp_xls.shape[0] )
			print( sys_name, " --> ", np.count_nonzero( fp_satisfaction ), "  ", fp_satisfaction.shape )
		return fp_sat_dict


	def get_sys_chains( self, auth_asym_ids: List[str] ):
		"""
		Given the system chain IDs, create the system chain IDs starting from "A".
		"""
		sys_idx = 0
		sys_chains = []
		for aa_id in auth_asym_ids:
			tmp = []
			for id_ in aa_id.split( "-" ):
				chain_id = get_chain_id( idx = sys_idx )
				tmp.append( chain_id )
				sys_idx += 1
			sys_chains.append( "-".join( tmp ) )
		return sys_chains


	def get_fp_XLs( self, sys_name: str ) -> pd.DataFrame:
		"""
		Parse the XLs .csv file for the given system
			and return the FP XLs.
		"""
		data_dir = self.get_sys_data_dir_path( sys_name = sys_name )
		xl_file = os.path.join(
			data_dir,
			f"interprotein_xls{self.sys_conf_suff}.csv" )
		xl_df = pd.read_csv( xl_file )
		fp_xls = xl_df[xl_df["label"] == 0]
		return fp_xls


	def get_modeled_residues( self, sys_name: str ) -> Dict[int, np.array]:
		"""
		Parse the start-end residue positions for the modeled sequence.
		Return dict contaiing residue positions form start to
			end for all entity_ids.
		"""
		data_dir = self.get_sys_data_dir_path( sys_name = sys_name )
		sys_config = self.get_sys_config( sys_name = sys_name )
		sys_key = list( sys_config.keys() )[0]
		modeled_res_dict = {}
		for entity in sys_config[sys_key]["entity"]:
			entity_id = int( entity["entity_id"] )
			start = int( entity["start"] )
			end = int( entity["end"] )

			modeled_res_dict[entity_id] = np.arange( start, end + 1, 1 )
		return modeled_res_dict


	def check_fp_xl_satisfaction( self,
			sys_name: str,
			model_id: str,
			sys_chains: List,
			modeled_res_dict: Dict[int, np.array],
			fp_xls: pd.DataFrame ):
		"""
		Load the structure file and get coordinates for all chains.
		For each XL,
			Get the protein name.
			Identify the chain or chains (homomers).
			Create all ambiguous chain pairs.
			Compute distance between the XL residues.
			FP XL satisfied if distance is within XL max bound.
		"""
		coords_dict = self.get_model_coords( sys_name = sys_name, model_id = model_id )
		fp_satisfaction = []
		for i in fp_xls.index:
			prot1, res1, prot2, res2, _ = fp_xls.loc[i]

			entity_id1 = int( prot1.split( "_" )[1] )
			entity_id2 = int( prot2.split( "_" )[1] )

			# print( modeled_res_dict.keys() )
			positions1 = modeled_res_dict[entity_id1]
			positions2 = modeled_res_dict[entity_id2]

			res_idx1 = self.get_residue_position_in_struct(
				modeled_pos = positions1, res = res1
				)
			res_idx2 = self.get_residue_position_in_struct(
				modeled_pos = positions2, res = res2
				)

			ambiguous_chain_pairs = self.get_ambiguous_chain_pairs(
				sys_chains = sys_chains,
				entity_id1 = entity_id1,
				entity_id2 = entity_id2
			)
			satisfied = self.compute_xl_distance(
				sys_name = sys_name,
				coords_dict = coords_dict,
				ambiguous_chain_pairs = ambiguous_chain_pairs,
				res_idx1 = res_idx1,
				res_idx2 = res_idx2
				)
			fp_satisfaction.append( int( satisfied ) )
		return np.array( fp_satisfaction )


	def get_residue_position_in_struct( self,
			modeled_pos: np.array, res: int ) -> np.array:
		"""
		Get the residue position in the structure (index)
			given the modeled residue position.
		Residue no. in the predicted structure start from 1 whereas
			the XL file contains the residues as per the 'seq_id'.
		So, the index of the XL residue in modeled sequence
			is the required residue in the predicted structure.
		"""
		idx = np.where( modeled_pos == res )[0][0]
		return idx


	def get_model_coords( self, sys_name: str, model_id: str ):
		"""
		Given the model_id, extract coordinates for all
			chains in the structure.
		"""
		ver_path = self.get_sys_modeling_version_path( sys_name )
		analysis_dir_path = os.path.join( ver_path, "analysis" )
		model_file = os.path.join(
			analysis_dir_path,
			f"relaxed_models/model_{model_id}.{self.struct_format}"
		)
		p = Parser( pdb_file = model_file )
		for m in p.get_models():
			coords_dict = p.get_coordinates( model = m )
		return coords_dict


	def get_ambiguous_chain_pairs( self,
		sys_chains: List,
		entity_id1: str, entity_id2: str ):
		"""
		Map each protein to the respective chain(s) and
			create all prot1/2 chain pairs.
		"""
		entity_ids = np.arange( 1, len( sys_chains ) + 1, 1 )

		idx1 = np.where( entity_ids == int( entity_id1 ) )[0][0]
		idx2 = np.where( entity_ids == int( entity_id2 ) )[0][0]

		chains1 = sys_chains[idx1].split( "-" )
		chains2 = sys_chains[idx2].split( "-" )

		ambiguous_chain_pairs = []
		# We only use inter-protein XLs.
		for c1 in chains1:
			for c2 in chains2:
				if c1 != c2:
					ambiguous_chain_pairs.append( [c1, c2] )

		return ambiguous_chain_pairs


	def compute_xl_distance( self,
			sys_name: str,
			coords_dict: Dict[str, np.array],
			ambiguous_chain_pairs: List[List],
			res_idx1: int,
			res_idx2: int ):
		"""
		For all given ambiguous chain pairs, compute
			the distance between the given residues.
		res1, res2 represent the 'seq_id' in the ground truth structure.
		"""
		sys_config = self.get_sys_config( sys_name = sys_name )
		sys_key = list( sys_config.keys() )[0]
		xl_max_bound = float( sys_config[sys_key]["data_gathering"]["xl_restraint"]["xl_max_bound"] )
		satisfied = []
		for amb_pair in ambiguous_chain_pairs:
			chain1, chain2 = amb_pair
			coord1 = coords_dict[chain1]
			coord2 = coords_dict[chain2]

			res1_xyz = coord1[res_idx1].reshape( 1, 3 )
			res2_xyz = coord2[res_idx2].reshape( 1, 3 )
			distance = np.linalg.norm( res1_xyz - res2_xyz )

			satisfied.append( distance <= xl_max_bound )
		return all( satisfied )

	################################################################################
	################################################################################
	def compute_dockq( self ):
		"""
		Compute the DockQ wrt the ground truth structure.
		"""
		print( "\n\033[1m--> Computing DockQ <--\033[0m]" )
		if os.path.exists( self.dockq_dict_file ):
			self.dockq_dict = read_json( self.dockq_dict_file )

		for sys_name in self.benchmark["PDB ID"]:
			if sys_name in self.dockq_dict:
				continue
			else:
				self.dockq_dict[sys_name] = {
					"init_struct": 0,
					"selected_good_models": []
					}
				self.compute_per_sys_dockq( sys_name = sys_name )

		write_json( self.dockq_dict, self.dockq_dict_file )


	def dockq( self, native_file: str, model_file: str ):
		"""
		Given the path to the native and the predicted structure (model),
			compute the DockQ metric.
		"""
		model = load_PDB( model_file )
		native = load_PDB( native_file )
		dockq_result = run_on_all_native_interfaces( model, native )
		dockq = dockq_result[1]

		return dockq


	def compute_per_sys_dockq( self, sys_name: str ):
		"""
		Compute the DockQ for:
			Initial predicted structure.
			All selected good-scoring models.
		"""
		data_dir = self.get_sys_data_dir_path( sys_name = sys_name )
		native_file = os.path.join( data_dir, f"{sys_name}.cif" )

		init_struct_dir = os.path.join( data_dir, f"{sys_name}_output/predictions" )
		init_model_file = glob.glob( f"{init_struct_dir}/*_unrelaxed.cif" )
		if not init_model_file:
			raise FileNotFoundError( f"Initial predicted structure not found for {sys_name}..." )
		else:
			init_model_file = init_model_file[0]

		# Get DockQ for initial predicted structure.
		init_struct_dockq = self.dockq( native_file = native_file, model_file = init_model_file )

		self.dockq_dict[sys_name]["init_struct"] = init_struct_dockq

		analysis_dict = self.load_analysis_dict( sys_name = sys_name )
		selected_good_models = analysis_dict["selected_good_models"]

		for i in selected_good_models:
			model_file = os.path.join( 
				self.get_sys_modeling_version_path( sys_name = sys_name ),
				f"analysis/relaxed_models/model_{i}.{self.struct_format}" )
			dockq = self.dockq( native_file = native_file, model_file = model_file )
			self.dockq_dict[sys_name]["selected_good_models"].append( dockq )

	################################################################################
	################################################################################
	def plot_modeling_results( self ):
		"""
		For violation loss, ccom loss, and XL satisfaction,
			Plot distribution of per epoch values for each system.
			Plot distribution of avg values across systems.
		"""
		print( "\n" + "-"*70 + "\n\t\t\t\033[1m--> Creating plots <--\033[0m\n" + "-"*70 )
		self.plot_per_epoch_distribution()
		if self.enable_relax_validate:
			if self.sys_conf_suff:
				self.plot_fpxl_ssatisfaction()
			# self.plot_dockq_score()
			self.plot_molrobity_scores()
		# self.plot_avg_distribution()


	def get_dict_for_source( self, sys_name: str, source: str ):
		"""
		Return the stats_dict (all_sampled) or analysis_dict
			(good_scoring) for the given system.
		"""
		if source == "all_sampled":
			data_dict = self.load_stat_file( sys_name = sys_name )
		elif source == "good_scoring":
			data_dict = self.load_analysis_dict( sys_name = sys_name )
		else:
			raise ValueError( f"Incorrect source name: {source}. " +
							"Supported 'all_sampled' or 'good-scoring'..." )
		return data_dict


	def get_input_for_per_epoch_plots( self, source: str ):
		"""
		Obtain the following for plotting per-epoch distribution plots:
			1. sys_name for all complexes.
			2. per-epoch violations ccom, xl satisfaction.
		"""
		xl_satisfaction_list = []
		global_satisfaction_list = []
		epoch0_xl_satisfaction_list = []
		violations_list = []
		ccom_list = []
		complexes_list = []
		for sys_name in self.benchmark["PDB ID"]:
			data_dict = self.get_dict_for_source(
				sys_name = sys_name,
				source = source )
			complexes_list.append( sys_name )
			if source == "all_sampled":
				xl_satisfaction_list.append(
					data_dict["metrics"]["xlr"]
				)
				epoch0_xl_satisfaction_list.append(
					data_dict["metrics"]["xlr"][0]
				)
				violations_list.append(
					data_dict["loss"]["violation"]
				)

			else:
				x = self.get_dict_for_source(
					sys_name = sys_name,
					source = "all_sampled" )
				epoch0_xl_satisfaction_list.append(
					x["metrics"]["xlr"][0]
				)

				xl_satisfaction_list.append( data_dict["per_model_xl_sat"] )
				global_satisfaction_list.append( data_dict["global_data_satisfaction"] )
				violations_list.append( data_dict["per_model_viol"] )
				#ccom_list.append( data_dict["per_model_ccom"] )

		return ( complexes_list, xl_satisfaction_list, epoch0_xl_satisfaction_list,
				global_satisfaction_list, violations_list )


	def create_violin( self, data: List, ax, r: int, color: str, ylabel: str ):
		"""
		Create a violinplot with the required formatting.
		"""
		if r == None:
			vp = ax.violinplot( dataset = data, orientation = "vertical",
										showmeans = True, showextrema = True )
		else:
			vp = ax[r].violinplot( dataset = data, orientation = "vertical",
										showmeans = True, showextrema = True )

		for body in vp["bodies"]:
			# body.set_alpha( 0.4 )
			body.set_facecolor( color )
		# Change color and width of the central line.
		vp["cbars"].set_color( "black" )
		vp["cbars"].set_linewidth( 1 )
		# Change color and width of the minimum line.
		vp["cmins"].set_color( "black" )
		vp["cmins"].set_linewidth( 1 )
		# Change color and width of the maximum line.
		vp["cmaxes"].set_color( "black" )
		vp["cmaxes"].set_linewidth( 1 )
		# Change color and width of the mean line.
		vp["cmeans"].set_color( "blue" )
		vp["cmeans"].set_linewidth( 2 )
		if r == None:
			ax.tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax.set_ylabel( ylabel, fontsize = 20 )
		else:
			ax[r].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[r].set_ylabel( ylabel, fontsize = 20 )


	def plot_per_epoch_distribution( self ):
		"""
		Plot the distribution of per epoch values for
			 loss loss, and xl satisfaction for each systems.
		 Create separate plots for each term.
		"""
		all_sampled = self.get_input_for_per_epoch_plots( source = "all_sampled" )
		good_scoring = self.get_input_for_per_epoch_plots( source = "good_scoring" )
		color = ["lightblue", "orange"]

		# i = 0
		for name, out in zip( ["all_sampled", "good_scoring"], [all_sampled, good_scoring] ):
			plt.rcParams["font.family"] = "sans"
			_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )

			( complexes_list, xl_satisfaction_list, epoch0_xl_satisfaction_list,
					global_satisfaction_list, violations_list, ccom_list ) = out

			self.create_violin( data = xl_satisfaction_list, ax = ax, r = 0,
								color = "tab:blue", ylabel = "Per model XL satisfaction" )
			self.create_violin( data = violations_list, ax = ax, r = 1,
								color = "tab:blue", ylabel = "Per model Violations" )

			complex_idx = np.arange( 1, len( complexes_list ) + 1 )
			ax[0].set_xticks( complex_idx, complexes_list )
			ax[1].set_xticks( complex_idx, complexes_list )
			# Plot global XL satisfaction as a triangle.
			if len( global_satisfaction_list ) != 0:
				ax[0].scatter( complex_idx, global_satisfaction_list,
								c = "green",
								alpha = 1, linewidth = 2 )
			ax[0].scatter( complex_idx, epoch0_xl_satisfaction_list,
							c = "red",
							alpha = 1, linewidth = 2 )

			# i += 1

			path = os.path.join( self.benchmark_modeling_dir, f"{name}_per_model_metrics.png" )
			plt.savefig( path, dpi = 300 )
			complex_idx = np.arange( 1, len( complexes_list ) + 1 )
		plt.close()


	def plot_fpxl_ssatisfaction( self ):
		"""
		Scatter plot to show fraction of FP XLs satisfied among
			the good-scoring models.
		"""
		fp_sat_dict = self.assess_fp_satisfaction()
		complexes = fp_sat_dict["complexes"]
		fp_satisfied = np.array( fp_sat_dict["fp_satisfied"] )
		total_fp = np.array( fp_sat_dict["total_fp"] )
		frac_fp_sat = np.round( fp_satisfied/total_fp, 2 )

		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 1, 1, figsize = ( 20, 10 ) )

		complex_idx = np.arange( 1, len( complexes ) + 1 )

		ax.scatter( complex_idx, frac_fp_sat,
					c = "green", alpha = 1, linewidth = 2 )

		x_offset = ( ax.get_xlim()[1] - ax.get_xlim()[0] ) * 0.005
		y_offset = ( ax.get_ylim()[1] - ax.get_ylim()[0] ) * 0.02
		for i, label in enumerate( total_fp ):
			x = i + 1 + x_offset
			y = frac_fp_sat[i] + y_offset
			ax.text( x, y, str( label ), fontsize = 10 )

		ax.set_xlabel( "Complexs", fontsize = 20 )
		ax.set_ylabel( "Fraction of FP XLs satisfied", fontsize = 20 )
		ax.set_xticks( complex_idx, complexes )
		ax.tick_params( axis = "both" , labelsize = 10, length = 10, width = 3 )
		plt.savefig( self.fp_satisfaction_plot_file, dpi = 300 )
		plt.close()


	def plot_dockq_score( self ):
		"""
		Plot the DockQ score for all selected
			models as a violin plot.
		Also plot the DockQ of the initial predicted structure.
		"""
		per_sys_dockq = []
		init_dockq = []
		complexes = list( self.dockq_dict.keys() )

		for sys_name in self.dockq_dict:
			per_sys_dockq.append( self.analysis_dict[sys_name]["selected_good_models"] )
			init_dockq.append( self.dockq_dict[sys_name]["init_dockq"] )

		complex_idx = np.arange( 1, len( complexes ) + 1 )

		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )

		self.create_violin( data = per_sys_dockq, ax = ax, r = None,
							color = "tab:blue", ylabel = "DockQ score" )
		ax.scatter( complex_idx, init_dockq,
					color = "red", marker = "s", s = 70,
					alpha = 1, linewidth = 2  )
		plt.savefig( self.dockq_plot_file, dpi = 300 )
		plt.close()


	def plot_molrobity_scores( self ):
		"""
		Plot the distribution of MolProbity scores for each complex.
		Also plot the resolution of the experimental structure.
		"""
		molprobity_score = []
		complexes = []
		resolution = []
		reference_y = []

		self.resolution_dict = read_json( self.resolution_dict_file )

		for sys_name in self.benchmark["PDB ID"]:
			complexes.append( sys_name )
			analysis_dict = self.load_analysis_dict( sys_name = sys_name )
			molprob_dict = analysis_dict["molprob"]

			resolution.append( self.resolution_dict[sys_name] )
			reference_y.append( self.resolution_dict[sys_name] )

			tmp = []
			for model_id in molprob_dict:
				tmp.append( molprob_dict[model_id]["relaxed"]["MolProbity score"] )
			molprobity_score.append(  np.round( np.mean( tmp ), 2 ) )

		complex_idx = np.arange( 1, len( complexes ) + 1 )

		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )

		ax.scatter( resolution, molprobity_score,
					marker = "s", s = 70,
					alpha = 1, linewidth = 2  )
		ax.plot( resolution, reference_y,
					alpha = 1, linewidth = 2  )
		ax.set_xlabel( "Resolution of experimental structure", fontsize = 25 )
		ax.set_ylabel( "Molprobity score", fontsize = 25 )
		# self.create_violin( data = molprobity_score, ax = ax, r = None,
		# 					color = "tab:blue", ylabel = "MolProbity score" )
		# ax.scatter( complex_idx, resolution,
		# 			color = "red", marker = "s", s = 70,
		# 			alpha = 1, linewidth = 2  )
		plt.savefig( self.molprob_plot_file, dpi = 300 )
		plt.close()

	################################################################################
	################################################################################
	def write_results_to_csv( self ):
		"""
		Write the following for each complex to a .csv file:
			XL satisfaction at epoch 0
			Avg. XL satisfaction for good-scoring models
			Global XL satisfaction for good-scoring models
			Violations at epoch 0
			Avg. Violations for good-scoring models
			Avg. MolProbity score for good-scoring models
		"""
		flat_dict = {"metrics": []}
		flat_dict["metrics"].extend( [k for k in [
			"num_models", "epoch0_xl", "avg_xl", "global_xl",
			"epoch0_viol", "avg_viol", "avg_molprob_score_unrelax",
			"avg_molprob_score_relax"]] )

		flat_dict.update( {k:[] for k in self.benchmark["PDB ID"]} )

		for sys_name in self.benchmark["PDB ID"]:
			stats_dict = self.load_stat_file( sys_name = sys_name )
			analysis_dict = self.load_analysis_dict( sys_name = sys_name )

			flat_dict[sys_name].append( len( analysis_dict["selected_good_models"] ) )
			flat_dict[sys_name].append( stats_dict["metrics"]["xlr"][0] )
			flat_dict[sys_name].append( np.round( np.mean( analysis_dict["per_model_xl_sat"] ), self.prec ) )
			flat_dict[sys_name].append( np.round( analysis_dict["global_data_satisfaction"], self.prec ) )

			flat_dict[sys_name].append( stats_dict["loss"]["violation"][0] )
			flat_dict[sys_name].append( np.round( np.mean( analysis_dict["per_model_viol"] ), self.prec ) )

			if self.enable_relax_validate:
				unrelax_avg, relax_avg = [], []
				molprob_dict = analysis_dict["molprob"]
				for model_id in molprob_dict:
					unrelax_avg.append( molprob_dict[model_id]["unrelaxed"]["MolProbity score"] )
					relax_avg.append( molprob_dict[model_id]["relaxed"]["MolProbity score"] )
				unrelax_avg = np.round( np.mean( unrelax_avg ), self.prec )
				relax_avg = np.round( np.mean( relax_avg ), self.prec )
				flat_dict[sys_name].append( unrelax_avg )
				flat_dict[sys_name].append( relax_avg )
			else:
				flat_dict[sys_name].append( 0 )
				flat_dict[sys_name].append( 0 )

		df = pd.DataFrame( flat_dict )
		median = df.iloc[:, 1:].median( axis = 1 )
		df.insert( 1, "median", median )
		df.to_csv( self.results_file, index = False )


	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		self.base_dir = os.path.join( os.path.abspath( "./benchmark" ) )
		self.meta_dir = os.path.join( self.base_dir,
									f"{self.benchmark_name}_metadata" )
		# Name for the dir to store modeling output for all systems.
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"
		# Output dir for storing each benchmark run results.
		self.benchmark_output_dir = os.path.join(
												self.base_dir,
												f"{self.benchmark_name}_benchmark_results" )
		self.benchmark_modeling_dir = os.path.join( self.benchmark_output_dir,
												f"version_{self.modeling_version}" )
		self.benchmark_csv_file = os.path.join( self.meta_dir,
												f"selected_{self.benchmark_name}_benchmark.csv" )
		# Dict containing the resolution of the experimental structure.
		self.resolution_dict_file = os.path.join( self.meta_dir, "resolution_dict.json" )
		# File to store DOckQ.
		self.dockq_dict_file = os.path.join( self.benchmark_modeling_dir, f"dockq_dict.json" )
		# File to write benchmark results to a csv file.
		self.results_file = os.path.join( self.benchmark_modeling_dir, f"Results_v{self.modeling_version}.csv" )
		self.fp_satisfaction_plot_file = os.path.join( self.benchmark_modeling_dir, f"fp_xl_satisfaction.png" )
		self.dockq_plot_file = os.path.join( self.benchmark_modeling_dir, f"dockq_plot.png" )
		self.molprob_plot_file = os.path.join( self.benchmark_modeling_dir, f"molprob_plot.png" )

		# File to store time and memory used per system.
		self.logs_file = os.path.join( self.benchmark_modeling_dir, f"Logs_v{self.modeling_version}.json" )
		# File to store configs used fo rmodeling the benchmark.
		self.config_file = os.path.join( self.benchmark_modeling_dir, "Configs.json" )


	def create_required_dirs( self ):
		"""
		Create the required directories.
		"""
		os.makedirs( self.benchmark_output_dir, exist_ok = True )
		os.makedirs( self.benchmark_modeling_dir, exist_ok = True )


	def initialize_logs_dict( self ):
		"""
		Log the following info:
			Time taken by each system.
			Memory consumed by each system.
		"""
		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )
		else:
			self.logs = {k:{} for k in ["time",
											"memory_allocated",
											"memory_reserved",
											"errored"]}

	################################################################################
	################################################################################
	def load_benchmark( self ):
		"""
		Load the benchmak .csv file.
		Note the total no. of systems to be modeled.
		"""
		self.benchmark = pd.read_csv( self.benchmark_csv_file )
		self.num_systems = self.benchmark.shape[0]


	def get_sys_path( self, sys_name: str ):
		sys_path = os.path.join( 
					os.path.abspath(
						f"{self.base_dir}/{self.modeling_dir_name}/{sys_name}"
						)
			)
		return sys_path


	def get_sys_data_dir_path( self, sys_name: str ):
		"""
		Return the path to the system directory.
		"""
		data_dir = os.path.join(
			self.base_dir,
			f"{self.benchmark_name}_benchmark/{sys_name}" )
		return data_dir


	def get_sys_config( self, sys_name: str ):
		"""
		Return the sys_config dict.
		"""
		data_dir = self.get_sys_data_dir_path( sys_name = sys_name )
		sys_config = read_json(
			os.path.join(
				data_dir,
				f"sys_config_{sys_name}{self.sys_conf_suff}.json" )
			)
		return sys_config


	def get_sys_modeling_version_path( self, sys_name: str ):
		"""
		Return the path to the system modeling version dir.
		"""
		sys_path = self.get_sys_path( sys_name )
		ver_path = os.path.join( sys_path,
							f"version_{self.modeling_version}"
							)
		return ver_path


	def get_stat_file_path( self, sys_name: str ) -> pd.DataFrame:
		"""
		Return the path to the stats file for the given system.
		"""
		ver_path = self.get_sys_modeling_version_path( sys_name )
		stat_file_path = os.path.join( ver_path, "Stats.npy" )
		return stat_file_path


	def load_stat_file( self, sys_name: str ) -> Dict[str, Dict]:
		"""
		Load the stat file on memory.
		"""
		stat_file_path = self.get_stat_file_path( sys_name = sys_name )
		stats_dict = np.load( stat_file_path, allow_pickle = True ).item()
		return stats_dict


	def get_analysis_file_path( self, sys_name: str ) -> pd.DataFrame:
		"""
		Return the path to the analysis_dict file for the given system.
		"""
		ver_path = self.get_sys_modeling_version_path( sys_name )
		analysis_dict_file = os.path.join( ver_path, "analysis/analysis_dict.npy" )
		return analysis_dict_file


	def load_analysis_dict( self, sys_name: str ) -> Dict[str, Dict]:
		"""
		Load the analysis_dict on memory.
		"""
		analysis_dict_file = self.get_analysis_file_path( sys_name )
		analysis_dict = np.load( analysis_dict_file, allow_pickle = True ).item()
		return analysis_dict

	################################################################################
	################################################################################
	def record_configs( self, total_time: float ):
		"""
		Record all configs used for modeling the benchmark.
		Save on disk as a JSON file.
		"""
		if os.path.exists( self.config_file ):
			configs = read_json( self.config_file )
		else:
			configs = {}

		summed_time = 0

		if "time_taken" not in configs:
			for sys_name in self.logs["time"]:
				t = self.logs["time"][sys_name]
				summed_time += t

			time_taken = total_time if total_time > summed_time else summed_time
			configs = {
				"time_taken": f"{time_taken} seconds OR {time_taken/3600} hours",
			}

		current_datetime = datetime.now()
		timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
		with subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE ) as proc:
			system = proc.communicate()[0].decode().strip()

		configs.update( {
				"timestamp": timestamp,
				"system": str( system )
			} )

		configs.update( {
				"benchmark": self.benchmark_name,
				"objective": self.modeling_objective,
				"version": str( self.modeling_version ),
				"max_epochs": str( self.max_epochs ),
				"max_pose_iters": str( self.max_pose_iters ),
				"skip_pose_sampling": self.skip_pose_sampling,
				"fill_none": self.fill_none,
				"prec": self.prec,
				"device": self.device,
				"db_preset": self.db_preset,
				"sys_conf_suff": self.sys_conf_suff,
				"enable_analysis": self.enable_analysis,
				"enable_relax_validate": self.enable_relax_validate,
				"disable_overwrite_prompt": self.disable_overwrite_prompt,
				"remove_sys_modeling_dir": self.remove_sys_modeling_dir,
				"remove_sys_analysis_dir": self.remove_sys_analysis_dir,
				"violation": self.violation,
				"xlr": self.xlr,
				"base_dir": self.base_dir,
				"meta_dir": self.meta_dir,
				"modeling_dir_name": self.modeling_dir_name,
				"modeling_version": self.modeling_version,
				"benchmark_output_dir": self.benchmark_output_dir,
				"benchmark_modeling_dir": self.benchmark_modeling_dir,
				"benchmark_csv_file": self.benchmark_csv_file,
				"dockq_dict_file": self.dockq_dict_file,
				"molprob_plot_file": self.molprob_plot_file,
				"logs_file": self.logs_file,
				"config_file": self.config_file
			} )
		write_json( configs, self.config_file )

if __name__ == "__main__":
	BenchmarkModeling().forward()

