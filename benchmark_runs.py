"""
Wrapper class for running the IntegrativeLearning module on a benchmark.
	Update the topology file with the configs for the simulation.
Also creates analysis plots for the benchmark.
Allows limited utility to specify hyperparameters in the topology file.
"""
from typing import List, Dict
import os, glob, time, subprocess, traceback, warnings
from datetime import datetime
import numpy as np
import pandas as pd
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import torch

from openfold_wrapper import IntegrativeLearning

from topology import topology_dict
from utils.utils import ( open_file_handler, read_json, write_json, run_subprocess )
from utils.pdb_utils import Parser, get_chain_id
from utils.paths import (
	get_meta_dir_path,
	get_sys_data_dir_path,
	get_sys_modeling_path,
	get_benchmark_analysis_dir_path,
	get_benchmark_analysis_version_dir_path,
	get_benchmark_csv_file,
	get_sys_config_path,
	get_stat_file_path,
	get_analysis_dict_path,
	get_unrelaxed_model_file,
	get_relaxed_model_file,
	get_native_struct_file,
	get_init_struct_file )
from utils.tools import usalign, get_alignment_score
from xlmerged_hparams import xlmerged_hyperparameters

class BenchmarkModeling():
	def __init__( self, topo_dict: ConfigDict = None ):
		self.benchmark_name = "experiment"  # xlsim/abag/oreilly/experiment/xlmerged
		# Define the modeling objective.
		self.modeling_objective = "No pose sampling. Recycling with MSA subsampling neff=25 and column masking mask_frac=0.3."
		self.modeling_version = 3
		# PDB/CIF output for the predicted structure.
		self.struct_format = "pdb"
		# Suffix for modeling with TP+FP XLs.
		self.sys_conf_suff = ""
		# No. of recyling iters for OpenFold.
		self.num_recycles = 1
		# Maximum no. of epochs for fine-tuning.
		self.num_frames = 50
		# Sample rigid transformations at random.
		self.sample_random_pose = False
		# Initialize final_atom_positions to 0 or initial predicted structure.
		self.init_coord = "zero"
		# "init": Initialize final_atom_positions again; "prev_frame": use from previous epoch.
		self.reinit_frame = "prev_frame"
		# Precision of the float values in the results.
		self.prec = 4
		# Set GPU to use.
		self.device = "cuda:0"
		# Use full or reduced dataset for prediction - full_dbs/reduced_dbs.
		self.db_preset = "full_dbs"
		# If True, skip re-running modeling if output files already exist.
		self.skip_rerun = True
		# USalign script.
		self.usalign_script = "USalign"
		# Enable analysis.
		self.enable_analysis = True
		# If True run AMBER relaxation and MolProbity validation.
		self.enable_relax_validate = True
		# If True, will not show prompt for existing modeling dir.
		self.disable_overwrite_prompt = True
		# if True, deletes the existing system modeling dir.
		self.remove_sys_modeling_dir = False
		# if True, deletes the existing system analysis dir.
		self.remove_sys_analysis_dir = False
		# If True, create the required plots.
		self.create_summary_plots_and_files = True

		# Modify settings for losses to be used.
		self.violation = {"enabled": True, "add_penalty": True, "weight": 1.0
		}
		# self.violation = {"enabled": True, "add_penalty": True, "weight": 1.0,
		# 	"ev": {"intra_chain_dist": 2.0, "inter_chain_dist": 2.0, "weight": 1.0}
		# }
		self.xlr = {"enabled": True, "add_penalty": True, "weight": 1.0}
		# For analysis
		self.model_selection = {
			"quant_filter": {
				"enabled": True,
				"quantiles": {"xlr": 0.75, "violation": 0.25}
			}
		}

		self.topo_dict = topo_dict

		self.tm_dict = {}


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
			Log the errorneous entry_id in the system specific mdeling dir..
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
			topo_dict.model.num_recycles = self.num_recycles
			topo_dict.train.num_frames = self.num_frames
			topo_dict.train.sample_random_pose = self.sample_random_pose
			topo_dict.train.init_coord = self.init_coord
			topo_dict.train.reinit_frame = self.reinit_frame
			topo_dict.train.device = self.device

			# Violation loss settings.
			topo_dict.loss.violation.enabled = self.violation["enabled"]
			topo_dict.loss.violation.add_penalty = self.violation["add_penalty"]
			topo_dict.loss.violation.weight = self.violation["weight"]

			# XL restraint loss settings.
			topo_dict.loss.xlr.enabled = self.xlr["enabled"]
			topo_dict.loss.xlr.add_penalty = self.xlr["add_penalty"]
			topo_dict.loss.xlr.weight = self.xlr["weight"]

			# Model selection.
			topo_dict.analysis.model_selection.method.quant_filter = self.model_selection["quant_filter"]
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
			stat_file_path = get_stat_file_path(
				base_dir = self.base_dir,
				sys_name = sys_name,
				modeling_dir_name = self.modeling_dir_name,
				modeling_version = self.modeling_version )

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
		# data_dir = self.get_sys_data_dir_path( sys_name = sys_name )
		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

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
	def compute_tm_score( self ):
		"""
		Compute TM-score wrt the ground truth structure across the entire benchmark for:
			All sampled models
			Good-scoring models
		"""
		print( "\n" + "-"*70 +
			"\n\t\033[1m--> Computing TM-score wrt the ground truth structure <--\033[0m\n" +
			"-"*70 )
		if os.path.exists( self.tm_dict_file ):
			self.tm_dict = read_json( self.tm_dict_file )

		for sys_name in self.benchmark["PDB ID"]:
			if sys_name in self.tm_dict:
				continue
			else:
				all_sampled_tm = self.compute_tm_all_sampled_models( sys_name = sys_name )
				good_models_tm = self.compute_tm_good_scoring_models( sys_name = sys_name )
				init_tm = self.compute_tm_init_struct( sys_name = sys_name )
				self.tm_dict[sys_name] = {
					"init_struct": init_tm,
					"all_sampled": all_sampled_tm,
					"good_scoring": good_models_tm
					}
				print( f"{sys_name}: {init_tm}\n{good_models_tm}\n" )

			write_json( self.tm_dict, self.tm_dict_file )


	def create_tmp_dir( self ):
		"""
		Create a temporary dir for storing intermediate
			files for rmsd computation.
		"""
		self.tmp_dir = os.path.join( self.benchmark_modeling_dir, "tm_tmp" )
		os.makedirs( self.tmp_dir, exist_ok = True )


	def remove_tmp_dir( self ):
		cmd = ["rm", "-r", f"{self.tmp_dir}"]
		run_subprocess( cmd )


	def run_usalign( self,
		sys_name: str,
		model_id2: int,
		model2_file: str
		):
		native_file = get_native_struct_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
		)
		stdout_file = usalign(
			usalign_script = self.usalign_script,
			model_id1 = 00,  # Arbitrary model_id for the native struct.
			model1_file = native_file,
			model_id2 = model_id2,  # Arbitrary model_id for the init struct.
			model2_file = model2_file,
			tmp_dir = self.tmp_dir,
			mol = "prot",
			mm = 1,
			ter = 1
			)
		_, tm = get_alignment_score( stdout_file )
		return tm


	def compute_tm_init_struct( self, sys_name: str ):
		"""
		Compute TM-score for the initial OpenFold predicted structure.
		"""
		self.create_tmp_dir()
		init_struct_file = get_init_struct_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
		)
		tm = self.run_usalign(
			sys_name = sys_name,
			model_id2 = 00,
			model2_file = init_struct_file
		)
		self.remove_tmp_dir()
		return tm


	def compute_tm_all_sampled_models( self, sys_name: str ):
		"""
		Compute TM-score for all sampled models in the given system.
		"""
		self.create_tmp_dir()
		stats_dict = self.load_stat_dict( sys_name = sys_name )

		all_sampled_tm = []
		for model_id in stats_dict["model_id"]:
			model_file = get_unrelaxed_model_file(
				base_dir = self.base_dir,
				modeling_dir_name = self.modeling_dir_name,
				sys_name = sys_name,
				modeling_version = self.modeling_version,
				model_id = model_id
			)
			tm = self.run_usalign(
				sys_name = sys_name,
				model_id2 = model_id,
				model2_file = model_file
			)
			all_sampled_tm.append( tm )
		self.remove_tmp_dir()
		return all_sampled_tm


	def compute_tm_good_scoring_models( self, sys_name: str ):
		"""
		Compute TM-score for good-scoring models in the given system.
		Using the structure post-relaxation.
		"""
		self.create_tmp_dir()
		analysis_dict = self.load_analysis_dict( sys_name = sys_name )

		good_models_tm = []
		for model_id in analysis_dict["selected_good_models"]:
			model_file = get_relaxed_model_file(
				base_dir = self.base_dir,
				modeling_dir_name = self.modeling_dir_name,
				sys_name = sys_name,
				modeling_version = self.modeling_version,
				model_id = model_id
			)
			tm = self.run_usalign(
				sys_name = sys_name,
				model_id2 = model_id,
				model2_file = model_file
			)
			good_models_tm.append( tm )
		self.remove_tmp_dir()
		return good_models_tm

	################################################################################
	################################################################################
	def plot_modeling_results( self ):
		"""
		For violation loss, and XL satisfaction,
			Plot distribution of per epoch values for each system.
			Plot distribution of avg values across systems.
		"""
		print( "\n" + "-"*70 + "\n\t\t\t\033[1m--> Creating plots <--\033[0m\n" + "-"*70 )
		self.plot_per_epoch_distribution()
		if self.enable_relax_validate:
			self.plot_molrobity_scores()


	def get_dict_for_source( self, sys_name: str, source: str ):
		"""
		Return the stats_dict (all_sampled) or analysis_dict
			(good_scoring) for the given system.
		"""
		if source == "all_sampled":
			data_dict = self.load_stat_dict( sys_name = sys_name )
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
			2. per-epoch violations, xl satisfaction.
		"""
		xl_satisfaction_list = []
		global_satisfaction_list = []
		epoch0_xl_satisfaction_list = []
		violations_list = []
		complexes_list = []
		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			data_dict = self.get_dict_for_source(
				sys_name = sys_name,
				source = source )
			complexes_list.append( sys_name )
			if source == "all_sampled":
				xl_satisfaction_list.append(
					data_dict["metrics"]["xlr"]
				)
				epoch0_xl_satisfaction_list.append(
					self.benchmark.iloc[i, 2]
					# data_dict["metrics"]["xlr"][0]
				)
				violations_list.append(
					data_dict["loss"]["violation"]
				)

			else:
				epoch0_xl_satisfaction_list.append(
					self.benchmark.iloc[i, 2]
				)

				xl_satisfaction_list.append( data_dict["per_model_xl_sat"] )
				global_satisfaction_list.append( data_dict["global_data_satisfaction"] )
				violations_list.append( data_dict["per_model_viol"] )

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

	################################################################################
	def plot_per_epoch_distribution( self ):
		"""
		Plot the distribution of per epoch values for
			 loss loss, and xl satisfaction for each systems.
		 Create separate plots for each term.
		"""
		all_sampled = self.get_input_for_per_epoch_plots( source = "all_sampled" )
		good_scoring = self.get_input_for_per_epoch_plots( source = "good_scoring" )

		# i = 0
		for name, out in zip( ["all_sampled", "good_scoring"], [all_sampled, good_scoring] ):
			plt.rcParams["font.family"] = "sans"
			_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )

			( complexes_list, xl_satisfaction_list, epoch0_xl_satisfaction_list,
					global_satisfaction_list, violations_list ) = out

			self.create_violin( data = xl_satisfaction_list, ax = ax, r = 0,
								color = "tab:blue", ylabel = "Per model XL satisfaction" )
			self.create_violin( data = violations_list, ax = ax, r = 1,
								color = "tab:blue", ylabel = "Per model Violations" )

			complex_idx = np.arange( 1, len( complexes_list ) + 1 )

			ax[0].set_ylim( -0.1, 1.1 )
			ax[1].set_ylim( -0.05 )
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


			path = os.path.join( self.benchmark_modeling_dir, f"{name}_per_model_metrics.png" )
			plt.tight_layout()
			plt.savefig( path, dpi = 300 )
			complex_idx = np.arange( 1, len( complexes_list ) + 1 )
		plt.close()


	################################################################################
	def plot_tm_score_distribution( self ):
		"""
		Plot the distribution of TM-score wrt the ground truth structure for:
			All sampled models
			Good scoring models
			Initial predicted structure
		"""
		# Compute the TM-score.
		self.compute_tm_score()

		all_sampled_list, good_scoring_list, init_list = [], [], []
		for sys_name in self.tm_dict:
			all_sampled_list.append( self.tm_dict[sys_name]["all_sampled"] )
			good_scoring_list.append( self.tm_dict[sys_name]["good_scoring"] )
			init_list.append( self.tm_dict[sys_name]["init_struct"] )

		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )

		complexes_list = list( self.tm_dict.keys() )
		complex_idx = np.arange( 1, len( complexes_list ) + 1 )

		self.create_violin( data = all_sampled_list, ax = ax, r = 0,
							color = "tab:blue", ylabel = "TM score" )
		self.create_violin( data = good_scoring_list, ax = ax, r = 1,
							color = "tab:blue", ylabel = "TM score" )
		# Plot the TM-score for the initial predicted structure.
		ax[0].scatter( complex_idx, init_list,
					c = "red", alpha = 1, linewidth = 2 )
		ax[1].scatter( complex_idx, init_list,
					c = "red", alpha = 1, linewidth = 2 )

		ax[0].set_title( "TM-score for all sampled models", fontsize = 25 )
		ax[0].set_ylim( -0.05, 1.05 )
		ax[0].set_xticks( complex_idx, complexes_list )
		ax[1].set_title( "TM-score for good scoring models", fontsize = 25 )
		ax[1].set_ylim( -0.05, 1.05 )
		ax[1].set_xticks( complex_idx, complexes_list )

		plt.tight_layout()
		plt.savefig( self.tm_plot_file, dpi = 300 )
		plt.close()


	################################################################################
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
			ax.text( x, y, str( label ), fontsize = 16 )

		ax.set_xlabel( "Complexs", fontsize = 20 )
		ax.set_ylabel( "Fraction of FP XLs satisfied", fontsize = 20 )
		ax.set_ylim( -0.1, 1.1 )
		ax.set_xticks( complex_idx, complexes )
		ax.tick_params( axis = "both" , labelsize = 16, length = 10, width = 3 )

		plt.tight_layout()
		plt.savefig( self.fp_satisfaction_plot_file, dpi = 300 )
		plt.close()


	################################################################################
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
		ax.set_ylim( 0 )
		
		plt.tight_layout()
		plt.savefig( self.dockq_plot_file, dpi = 300 )
		plt.close()


	################################################################################
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
		_, ax = plt.subplots( 1, 1, figsize = ( 20, 10 ) )

		ax.scatter( resolution, molprobity_score,
					marker = "s", s = 40.0,
					alpha = 1, linewidth = 2  )
		ax.plot( resolution, reference_y,
					alpha = 1, linewidth = 2  )
		ax.set_xlabel( "Resolution of experimental structure", fontsize = 16 )
		ax.set_ylabel( "Molprobity score", fontsize = 16 )
		ax.tick_params( axis = "both" , labelsize = 16, length = 8, width = 3 )
		ax.set_xlim( 0 )
		ax.set_ylim( 0 )

		plt.tight_layout()
		plt.savefig( self.molprob_plot_file, dpi = 300 )
		plt.close()

	################################################################################
	################################################################################
	def write_results_to_csv( self ):
		"""
		Write the following for each complex to a .csv file:
			No. of good scoring models
			Init XL satisfaction
			Max XL satisfaction (all sampled)
			Max XL satisfaction (good scoring)
			Global XL satisfaction (all sampled)
			Global XL satisfaction (good scoring)
			Avg. MolProbity score for good scoring models
		"""
		flat_dict = {"metrics": []}
		flat_dict["metrics"].extend( [k for k in [
			"num_models", "init_xl_sat", "max_xl_as", "max_xl_gs",
			"global_xl_as", "global_xl_gs",
			"avg_molprob_score_unrelax",
			"avg_molprob_score_relax"]] )

		flat_dict.update( {k:[] for k in self.benchmark["PDB ID"]} )

		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			stats_dict = self.load_stat_dict( sys_name = sys_name )
			analysis_dict = self.load_analysis_dict( sys_name = sys_name )

			# No. of good scoring models.
			flat_dict[sys_name].append(
				len( analysis_dict["selected_good_models"] ) )
			# Initial XL satisfaction.
			flat_dict[sys_name].append(
				self.benchmark.loc[i, "xl_satisfaction"] )
			# Max Xl satisfaction for all sampled models.
			flat_dict[sys_name].append(
				np.round( np.max( stats_dict["metrics"]["xlr"] ), self.prec ) )
			# Max Xl satisfaction for good scoring models.
			flat_dict[sys_name].append(
				np.round( np.max( analysis_dict["per_model_xl_sat"] ), self.prec ) )
			# Global Xl satisfaction for all sampled models.
			flat_dict[sys_name].append(
				np.round( stats_dict["metadata"]["xlr"]["global_satisfaction"], self.prec ) )
			# Global Xl satisfaction for good scoring models.
			flat_dict[sys_name].append(
				np.round( analysis_dict["global_data_satisfaction"], self.prec ) )
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
		df.to_csv( self.results_file, index = False )

	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		# self.base_dir = os.path.join( os.path.abspath( "/data/kartik/IMP_Rewired/imp_dl/benchmark" ) )
		self.base_dir = os.path.join( os.path.abspath( "./benchmark" ) )
		# self.meta_dir = os.path.join( self.base_dir,
		# 							f"{self.benchmark_name}_metadata" )
		self.meta_dir = get_meta_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name )
		# Name for the modleing dir.
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"

		# Output dir for storing the benchmark analysis results for all versions.
		self.benchmark_output_dir = get_benchmark_analysis_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name )
		# Output dir for storing version-specific benchmark analsyis results.
		self.benchmark_modeling_dir = get_benchmark_analysis_version_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			modeling_version = self.modeling_version )
		# selected complexes in the benchmark.
		self.benchmark_csv_file = get_benchmark_csv_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name )

		# Dict containing the resolution of the experimental structure.
		self.resolution_dict_file = os.path.join( self.meta_dir, "resolution_dict.json" )
		# File to store TM-score.
		self.tm_dict_file = os.path.join( self.benchmark_modeling_dir, f"tm_dict.json" )
		# File to write benchmark results to a csv file.
		self.results_file = os.path.join( self.benchmark_modeling_dir, f"Results_v{self.modeling_version}.csv" )
		self.tm_plot_file = os.path.join( self.benchmark_modeling_dir, f"tm_score_dist.png" )
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


	def get_sys_config( self, sys_name: str ):
		"""
		Return the sys_config dict.
		"""
		# data_dir = self.get_sys_data_dir_path( sys_name = sys_name )
		sys_config_path = get_sys_config_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			sys_conf_suff = self.sys_conf_suff )
		sys_config = read_json( sys_config_path )
		return sys_config


	def get_sys_modeling_version_path( self, sys_name: str ):
		"""
		Return the path to the system modeling version dir.
		"""
		sys_modeling_path = get_sys_modeling_path(
			base_dir = self.base_dir,
			modeling_dir_name = self.modeling_dir_name,
			sys_name = sys_name
		)
		ver_path = os.path.join( sys_modeling_path,
							f"version_{self.modeling_version}"
							)
		return ver_path


	def load_stat_dict( self, sys_name: str ) -> Dict[str, Dict]:
		"""
		Load the stat file on memory.
		"""
		stat_file_path = get_stat_file_path(
			base_dir = self.base_dir,
			sys_name = sys_name,
			modeling_dir_name = self.modeling_dir_name,
			modeling_version = self.modeling_version )
		stats_dict = np.load( stat_file_path, allow_pickle = True ).item()
		return stats_dict


	def load_analysis_dict( self, sys_name: str ) -> Dict[str, Dict]:
		"""
		Load the analysis_dict on memory.
		"""
		analysis_dict_file =get_analysis_dict_path(
			base_dir = self.base_dir,
			sys_name = sys_name,
			modeling_dir_name = self.modeling_dir_name,
			modeling_version = self.modeling_version )
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

		for k, v in vars( BenchmarkModeling() ).items():
			if not k.startswith( "__" ) and not callable( v ):
				configs[k] = v
		write_json( configs, self.config_file )


if __name__ == "__main__":
	# BenchmarkModeling().forward()

	hyperparameters = xlmerged_hyperparameters
	for v in [13]:
		if f"experiment_{v}" in hyperparameters:
			obj = BenchmarkModeling()
			for k, v in hyperparameters[f"experiment_{v}"].items():
				setattr( obj, k, v )
			obj.forward()
			del obj
		else:
			raise ValueError( f"version {v} not present in hyperparameters dict..." )
