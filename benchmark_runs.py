"""
Run the IntegrativeLearning module on the entire benchmark.
Check the distribution of violation loss, ccom loss, xl_restraint.
The configs for the simulation must be specified in the topology file.
"""
from typing import List, Dict
import os, glob, time, subprocess, traceback, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from openfold_wrapper import IntegrativeLearning

from topology import topology_dict
from utils.utils import ( open_file_handler, read_json, write_json, run_subprocess )


class ModelingBenchmark():
	def __init__( self ):
		self.benchmark_name = "xlsim"  # xlsim/abag/oreilly
		# Define the modeling objective.
		self.modeling_objective = "test."
		self.modeling_version = 1
		# Maximum no. of epochs for fine-tuning.
		self.max_epochs = 10
		# Set GPU to use.
		self.device = "cuda:0"
		# Use full or reduced dataset for prediction - full_dbs/reduced_dbs.
		self.db_preset = "full_dbs"
		# If True run AMBER relaxation and MolProbity validation.
		self.enable_relax_validate = True
		# If True, will not show prompt for existing modeling dir.
		self.disable_overwrite_prompt = True
		# if True, deletes the existing system modeling dir.
		self.remove_sys_modeling_dir = True
		# if True, deletes the existing system analysis dir.
		self.remove_sys_analysis_dir = False

		# modify settings for losses to be used.
		self.fape = {"enabled": True, "add_penalty": False, "weight": 1.0,
					"interfape": True, "interfape_weight": 0.5}
		self.supervised_chi = {"enabled": True, "add_penalty": False, "weight": 1.0}
		self.violation = {"enabled": True, "add_penalty": True, "weight": 0.03}
		self.ccom = {"enabled": True, "add_penalty": True, "weight": 0.05}
		self.xlr = {"enabled": True, "add_penalty": True, "weight": 0.05}


	def forward( self ):
		"""
		"""
		tic = time.perf_counter()
		self.create_required_paths()
		self.create_required_dirs()
		self.load_benchmark()
		self.initialize_logs_dict()

		self.run_modeling_for_benchmark()
		self.plot_modeling_results()
		toc = time.perf_counter()

		# self.write_misc_details( toc-tic )
		self.record_configs( total_time = toc-tic )

	################################################################################
	################################################################################
	def log_memory_usage( self, sys_name: str ):
		"""
		Log the device memory used during modeling.
		device must be in the following formar: cuda[0]
		Reset the CUDA memory stats after logging.
		"""
		if self.device == "cpu":
			raise valueError( "Pytorch does not provide " +
					"built-in functions to check CPU memory stats. " +
					"Change device to cuda[0/1]..." )
		else:
			device_num = int( self.device[-1] )
		# Get peak memory since last reset.
		max_allocated = torch.cuda.max_memory_reserved( device_num )
		max_reserved = torch.cuda.max_memory_reserved( device_num )
		self.logs["memory_allocated"][sys_name] = max_allocated
		self.logs["memory_reserved"][sys_name] = max_reserved


	def log_error( self, sys_name: str ):
		"""
		If an error occurs while modeling,
			Log the errorneous entry_id in the dataset sepcific metadata dir.
			log the traceback in the system dir.
		"""
		self.logs["errored"][sys_name] = None

		ver_path = self.get_sys_modeling_version_path( sys_name )
		current_datetime = datetime.now()
		timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
		error_file = os.path.join(
			self.benchmark_modeling_dir,
			f"error_{sys_name}_{timestamp}.txt" )
		w = open_file_handler( error_file, "w" )
		w.write( traceback.format_exc() )
		w.close()

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
				print( "Changed your mind? Acts now..." )
				time.sleep( 5 )
				cmd = ["rm", "-r", f"{modeling_dir_path}"]
				run_subprocess( command = cmd )

		if self.remove_sys_analysis_dir:
			ver_path = self.get_sys_modeling_version_path( sys_name )
			analysis_dir_path = os.path.join( ver_path, "analysis/" )
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
		# topo_dict = topology_dict()
		# topo_dict.objective = f"{sys_name} {self.modeling_objective}"
		# topo_dict.train.version = self.modeling_version
		# topo_dict.train.device = self.device
		topo_dict = topology_dict()
		topo_dict.objective = f"{sys_name} {self.modeling_objective}"
		topo_dict.train.version = self.modeling_version
		topo_dict.train.device = self.device
		topo_dict.analysis.enable_relax_validate = self.enable_relax_validate
		topo_dict.db_preset = self.db_preset
		topo_dict.train.max_epochs = self.max_epochs

		# FAPE loss settings.
		topo_dict.loss.fape.enabled = self.fape["enabled"]
		topo_dict.loss.fape.add_penalty = self.fape["add_penalty"]
		topo_dict.loss.fape.weight = self.fape["weight"]
		topo_dict.loss.fape.interface_backbone.enabled = self.fape["interfape"]
		topo_dict.loss.fape.interface_backbone.weight = self.fape["interfape_weight"]

		# Supervised chi loss settings.
		topo_dict.loss.supervised_chi.enabled = self.supervised_chi["enabled"]
		topo_dict.loss.supervised_chi.add_penalty = self.supervised_chi["add_penalty"]
		topo_dict.loss.supervised_chi.weight = self.supervised_chi["weight"]

		# Violation loss settings.
		topo_dict.loss.violation.enabled = self.violation["enabled"]
		topo_dict.loss.violation.add_penalty = self.violation["add_penalty"]
		topo_dict.loss.violation.weight = self.violation["weight"]

		# Chain center of mass loss settings.
		topo_dict.loss.chain_center_of_mass.enabled = self.ccom["enabled"]
		topo_dict.loss.chain_center_of_mass.add_penalty = self.ccom["add_penalty"]
		topo_dict.loss.chain_center_of_mass.weight = self.ccom["weight"]

		# XL restraint loss settings.
		topo_dict.loss.xlr.enabled = self.xlr["enabled"]
		topo_dict.loss.xlr.add_penalty = self.xlr["add_penalty"]
		topo_dict.loss.xlr.weight = self.xlr["weight"]

		return topo_dict

	################################################################################
	################################################################################
	def run_modeling_for_benchmark( self ):
		"""
		Run the Integrative modeling pipeline for the entire benchmark.
		"""
		print( "\033[1mRunning modeling for the benchmark...\033[0m" )
		# Clear torch cache before starting.
		torch.cuda.empty_cache()

		curr_dir = os.getcwd()
		for idx, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( "\n" + "-"*70 + "\n" + "-"*70 )
			print( f"{idx}/{self.num_systems} --> {sys_name}" )
			print( "-"*70 + "\n" + "-"*70 + "\n" )
			sys_path = self.get_sys_path( sys_name )
			stat_file_path = self.get_stat_file_path( sys_path )

			# remove system modleing and/or analysis dir if specified.
			self.remove_existing_dir( sys_name = sys_name )

			if not os.path.exists( stat_file_path ):
				tic = time.perf_counter()
				self.run_modeling_for_system( sys_name )
				toc = time.perf_counter()

				self.logs["time"][sys_name] = toc-tic

				write_json( self.logs, self.logs_file )
			else:
				print( f"Summary file already present for {sys_name}..." )

			# Log memory used.
			self.log_memory_usage( sys_name = sys_name )
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
		data_dir = os.path.join( self.base_dir,
								f"{self.benchmark_name}_benchmark/{sys_name}/" )

		topo_dict = self.modify_topology( sys_name = sys_name )
		# topo_dict = topology_dict()
		# topo_dict.objective = f"{sys_name} {self.modeling_objective}"
		# topo_dict.train.version = self.modeling_version
		# topo_dict.analysis.enable_relax_validate = self.enable_relax_validate
		# try:
		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				sys_config_file =  f"sys_config_{sys_name}.json",
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.disable_overwrite_prompt = self.disable_overwrite_prompt
		il_obj.forward()
		# except:
		# 	self.log_error( sys_name = sys_name )

	################################################################################
	################################################################################
	def plot_modeling_results( self ):
		"""
		For violation loss, ccom loss, and XL satisfaction,
			Plot distribution of per epoch values for each system.
			Plot distribution of avg values across systems.
		"""
		print( "Creating plots..." )
		self.plot_per_epoch_distribution()
		self.plot_avg_distribution()


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


	def get_input_for_per_epoch_plots( self, source: str ):
		"""
		Obtain the following for plotting per-epoch distribution plots:
			1. sys_name for all complexes.
			2. per-epoch violations ccom, xl satisfaction.
		"""
		xl_satisfaction_list = []
		global_satisfaction_list = []
		violations_list = []
		ccom_list = []
		complexes_list = []
		for sys_name in self.benchmark["PDB ID"]:
			data_dict = self.get_dict_for_source(
				sys_name = sys_name,
				source = source )
			complexes.append( sys_name )
			if source == "all_sampled":
				xl_satisfaction.append(
					data_dict["metrics"]["xlr"]
				)
				violations.append(
					data_dict["loss"]["violation"]
				)
				ccom.append(
					data_dict["loss"]["chain_center_of_mass"]
				)
			else:
				xl_satisfaction.append( data_dict["per_model_xl_sat"] )
				violations.append( data_dict["per_model_viol"] )
				ccom.append( data_dict["per_model_ccom"] )
				global_satisfaction.append( data_dict["global_data_satisfaction"] )

		return ( complexes_list, xl_satisfaction_list,
				global_satisfaction_list, violations_list, ccom_list )


	def create_violin( self, data: List, ax, r: int, c: int, color: str, ylabel: str ):
		"""
		Create a violinplot with the required formatting.
		"""
		vp = ax[r, c].violinplot( dataset = xl_satisfaction, orientation = "vertical",
									showmeans = True, showextrema = True )
		for body in vp["bodies"]:
			body.set_alpha( 0.7 )
			body.set_facecolor( color )
		# Change color and width of the central line.
		vp["cbars"].set_color( "black" )
		vp["cbars"].set_linewidth( 2 )
		# Change color and width of the minimum line.
		vp["cmins"].set_color( "black" )
		vp["cmins"].set_linewidth( 2 )
		# Change color and width of the maximum line.
		vp["cmaxes"].set_color( "black" )
		vp["cmaxes"].set_linewidth( 2 )
		# Change color and width of the mean line.
		vp["cmeans"].set_color( "blue" )
		vp["cmeans"].set_linewidth( 4 )
		ax[r, c].tick_params( axis = "both" , labelsize = 25, length = 10, width = 4 )
		ax[r, c].set_ylabel( ylabel, fontsize = 25 )


	def plot_per_epoch_distribution( self ):
		"""
		Plot the distribution of per epoch values for
			 loss, ccom loss, and xl satisfaction for each systems.
		 Create separate plots for each term.
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 3, 1, figsize = ( 30, 20 ) )
		all_sampled = self.get_input_for_per_epoch_plots( source = "all_sampled" )
		good_scoring = self.get_input_for_per_epoch_plots( source = "good_scoring" )
		color = ["lightblue", "orange"]

		complex_idx = np.arange( 1, len( complexes_list ) + 1 )
		plt.xticks( complex_idx, complexes_list )

		i = 0
		for out in [all_sampled, good_scoring]:
			( complexes_list, xl_satisfaction_list,
					global_satisfaction_list, violations_list, ccom_list ) = out

			self.create_violin( data = xl_satisfaction_list, ax = ax, r = 0, c = 0,
								color = color[i], ylabel = "Per model XL satisfaction" )
			self.create_violin( data = violations_list, ax = ax, r = 1, c = 0,
								color = color[i], ylabel = "Per model Violations" )
			self.create_violin( data = ccom_list, ax = ax, r = 2, c = 0,
								color = color[i], ylabel = "Per model Chain center of mass" )

			# Plot global XL satisfaction as a triangle.
			if len( global_satisfaction_list ) != 0:
				ax[0, 0].scatter( global_satisfaction_list, complex_idx,
								c = "green", marker = "^", s = 70,
								alpha = 1, linewidth = 2 )
			i += 1

		path = os.path.join( self.benchmark_modeling_dir, f"per_model_metrics.png" )
		plt.savefig( path, dpi = 300 )
		plt.close()

	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		self.base_dir = os.path.join( os.path.abspath( "./benchmark/" ) )
		self.meta_dir = os.path.join( self.base_dir,
									f"{self.benchmark_name}_metadata" )
		# Name for the dir to store modeling output for all systems.
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"
		# Output dir for storing each benchmark run results.
		self.benchmark_output_dir = os.path.join(
												self.base_dir,
												f"{self.benchmark_name}_benchmark_results/" )
		self.benchmark_modeling_dir = os.path.join( self.benchmark_output_dir,
												f"version_{self.modeling_version}/" )
		self.benchmark_csv_file = os.path.join( self.meta_dir,
												f"selected_{self.benchmark_name}_benchmark.csv" )
		# File to store time and memory used per system.
		self.logs_file = os.path.join( self.benchmark_modeling_dir, f"Logs_{self.modeling_version}.json" )
		# File to write system name, data and time taken.
		# self.misc_file = os.path.join( self.benchmark_modeling_dir, "Time_taken.txt" )
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


	def get_sys_modeling_version_path( self, sys_name: str ):
		"""
		Return the path to the system modeling version dir.
		"""
		sys_path = self.get_sys_path( sys_name )
		ver_path = os.path.join( sys_path,
							f"version_{self.modeling_version}/"
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
		stat_file_path = self.get_stat_file_path( sys_name )
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
			self.configs = read_json( self.config_file )
		else:
			self.configs = {}

		summed_time = 0

		if "time_taken" in self.configs:
			for sys_name in self.logs:
				t = self.logs[sys_name]["time"]
				summed_time += t

			time_taken = total_time if total_time > summed_time else summed_time
			self.modeling_configs = {
				"time_taken": f"{time_taken} seconds OR {time_taken/3600} hours",
			}

		current_datetime = datetime.now()
		timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
		with subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE ) as proc:
			system = proc.communicate()[0]

		self.configs.update( {
				"timestamp": timestamp,
				"system": system
			} )

		self.configs.update( {
				"benchmark": self.benchmark_name,
				"objective": self.modeling_objective,
				"version": str( self.modeling_version ),
				"max_epochs": self.max_epochs,
				"device": self.device,
				"db_preset": self.db_preset,
				"enable_relax_validate": self.enable_relax_validate,
				"disable_overwrite_warning": self.disable_overwrite_warning,
				"remove_sys_modeling_dir": self.remove_sys_modeling_dir,
				"remove_sys_analysis_dir": self.remove_sys_analysis_dir,
				"fape": self.fape,
				"supervised_chi": self.supervised_chi,
				"violation": self.violation,
				"ccom": self.ccom,
				"xlr": self.xlr,
				"base_dir": self.base_dir,
				"meta_dir": self.meta_dir,
				"modeling_dir_name": self.modeling_dir_name,
				"modeling_version": self.modeling_version,
				"benchmark_output_dir": self.benchmark_output_dir,
				"benchmark_modeling_dir": self.benchmark_modeling_dir,
				"benchmark_csv_file": self.benchmark_csv_file,
				"logs_file": self.logs_file,
				# "misc_file": self.misc_file,
				"config_file": self.config_file
			} )


	# def plot_avg_distribution( self ):
	# 	"""
	# 	Plot the distribution of avg values across modeling run for,
	# 		 loss, ccom loss, and xl satisfaction for each systems.
	# 	 Create separate plots for each term.
	# 	"""
	# 	plt.rcParams["font.family"] = "sans"
	# 	for qty in ["violation", "chain_center_of_mass", "xlr"]:
	# 		init_sys_data = []
	# 		avg_sys_data = []
	# 		categories = ["Initial", "Average"]
	# 		_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )
	# 		for sys_name in self.benchmark["PDB ID"]:
	# 			sys_path = self.get_sys_path( sys_name )
	# 			stats_dict = self.load_stat_file( sys_path )

	# 			if qty == "xlr":
	# 				# metrics_dict = self.get_metrics_dict( sys_path )
	# 				metrics_dict = stats_dict["metrics"]
	# 				init_sys_data.append( np.round( metrics_dict[qty][0], 3 ) )
	# 				avg_sys_data.append( np.round( np.mean( metrics_dict[qty] ), 3 ) )
	# 			else:
	# 				# loss_dict = self.get_loss_dict( sys_path )
	# 				loss_dict = stats_dict["loss"]
	# 				init_sys_data.append( np.round( loss_dict[qty][0], 3 ) )
	# 				avg_sys_data.append( np.round( np.mean( loss_dict[qty] ), 3 ) )

	# 		ax.violinplot( [init_sys_data, avg_sys_data] )
	# 		ax.tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )

	# 		plt.xticks( np.arange(1, len( categories ) + 1 ), categories )

	# 		path = os.path.join( self.modeling_version_dir, f"{qty}_avg.png" )
	# 		plt.savefig( path, dpi = 300 )
	# 		plt.close()
	################################################################################
	################################################################################
	# def write_misc_details( self, time_taken: float ):
	# 	"""
	# 	Write down the system used, date, and objective of the simulation.
	# 	Assumes time is provided in seconds
	# 	"""
	# 	w = open_file_handler( self.misc_file, "w" )
	# 	with subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE ) as proc:
	# 		system = proc.communicate()[0]
	# 	with subprocess.Popen( "date", shell = True, stdout = subprocess.PIPE ) as proc:
	# 		sys_date = proc.communicate()[0]
	# 	w.writelines( f"System = {system} \t Date = {sys_date}\n" )
	# 	w.writelines( f"{time_taken/60} minutes OR {time_taken/3600} hours." )
	# 	w.close()

if __name__ == "__main__":
	ModelingBenchmark().forward()

