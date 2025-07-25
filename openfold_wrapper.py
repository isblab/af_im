"""
Contains a wrapper class that runs all the stages of integrative modeling pipeline.
	1. Data gathering
	2 System representation
	3. Sampling --> Fit to data
	4. Analysis --> Assay
"""

from typing import Dict, Optional
import os, sys, subprocess, math, time, random, copy
import numpy as np
import pandas as pd
import ml_collections as mlc

import torch

from openfold.config import model_config
from topology import topology_dict

from data_gathering import DataGathering
from system_representation import SystemRepresentation
from fit_to_data import FitToData
# from assay import Assay
from utils.create_plots import ( create_plot_from_dict,
								plot_scalar_metrics,
								plot_xl_map )
from utils.utils import ( read_json, write_configdict_to_json,
							open_file_handler, run_subprocess )


class IntegrativeLearning():
	"""
	A wrapper class that runs all the stages of integrative modeling pipeline.
	"""
	def __init__( self, sys_name: str, base_dir: str,
					data_dir: str,
					sys_config_file: str,
					modeling_dir_name: str,
					topology_dict: mlc.ConfigDict ):
		# Name of the system to be modeled.
		self.sys_name = sys_name # "2ayo" # H1129
		self.base_dir = base_dir
		# Directory containing input data for the modeled system.
		self.data_dir = data_dir
		# System config file name.
		self.sys_config_file = sys_config_file
		# Name for the dir to store modeling output.
		self.modeling_dir_name = modeling_dir_name
		# If True, will overwrite an existing dir without warning.
		self.disable_overwrite_warning = False
		# mono/multi
		self.pred_mode = "multi"

		# No. of CPU cores to be used.
		self.cpu_cores = 16
		self.prec = 4

		if self.pred_mode == "multi":
			# config_preset for multimer.
			self.config_preset = "model_1_multimer_v3"
			self.ckpt_path = None
		else:
			raise ValueError( f"Invalid mode = {self.pred_mode} specififed..." )

		# Load OpenFold configs file.
		self.ofold_config = model_config( self.config_preset )
		# Load the full system specific configs.
		self.topology = topology_dict

		self.add_sys_conf_to_topology()
		self.set_device()

		# Set the PRNG seed.
		self.seed_worker()


	def seed_worker( self ):
		"""
		Set seed for PRNG.
		"""
		seed = 1
		torch.manual_seed( seed )
		# torch.cuda.manual_seed( worker_seed )
		torch.cuda.manual_seed_all( seed )
		np.random.seed( seed )
		random.seed( seed )


	def set_device( self ):
		"""
		Set the device to be used.
		"""
		self.device = self.topology.train.device
		print( f"Using device = {self.device}" )
		time.sleep( 1 )


	def forward( self ):
		"""
		Serially run all the stages of the pipeline.
		"""
		tic = time.time()

		self.create_required_paths_dirs()
		# Check if modeling output dir exists.
		self.output_dir_exists()

		self.create_dirs()

		self.write_objective_file()

		# Save topology file on disk.
		self.save_topology_file()

		print( "\n" + "-"*70 + "\n" +"-"*27 + " Data gathering " + "-"*27 + "\n" + "-"*70 + "\n" )
		restraint_features = self.run_data_gathering()

		print( "\n" + "-"*70 + "\n" +"-"*24 + " System representation " + "-"*24 + "\n" + "-"*70 + "\n" )
		system_features = self.run_system_representation()
		# Add restraint features to system features dict.
		system_features["restraint_features"] = restraint_features

		print( "\n" + "-"*70 + "\n" +"-"*29 + " Fit to Data " + "-"*29 + "\n" + "-"*70 + "\n" )
		fit = self.run_fit_to_data( restraint_features, system_features )

		print( "\n" + "-"*70 + "\n" +"-"*26 + " \033[9m Analysis \033[0m Assay " + "-"*26 + "\n" + "-"*70 + "\n" )
		# self.run_analysis( fit )

		toc = time.time()
		time_file = os.path.join( self.modeling_output_dir, "Time_taken.txt" )
		if not os.path.exists( time_file ):
			w = open_file_handler( time_file, "w" )
			w.writelines( f"Time taken: {( toc-tic )/3600} hours OR {( toc-tic )/60} minutes" )
		run_subprocess( ["rm", "-r", f"{self.fasta_dir}"] )
		print( "May the Force be with you.." )
		print( f"Time taken: {( toc-tic )/3600} hours OR {( toc-tic )/60} minutes" )



	def run_data_gathering( self ):
		"""
		Instantiate and run the DataGathering module.
		"""
		# Get the restraint features.
		data_gathering = DataGathering( sys_name = self.sys_name,
									 		base_dir = self.data_dir,
									 		fasta_dir = self.fasta_dir,
									 		sys_config = self.topology.system )
		data_gathering.forward()
		restraint_features = data_gathering.restraint_features
		return restraint_features


	def run_system_representation( self ):
		"""
		Instantiate and run the SystemRepresentation module.
		"""
		init_dir = os.getcwd()
		# Get an initial structure and the ground truth features.
		system_features = SystemRepresentation(
							sys_name = self.sys_name,
							sys_rep_config = self.topology.system_representation,
							fasta_dir = self.fasta_dir,
							alignment_dir = self.alignment_dir,
							ofold_output_dir = self.ofold_output_dir,
							seed_worker = self.seed_worker,
							device = self.device
							).forward()

		# Move back to base dir.
		os.chdir( init_dir )
		return system_features


	def run_fit_to_data( self, restraint_features: Dict, system_features: Dict ):
		"""
		Instantiate and run the FitToData module.
		Save the stats file from simulation output.
		Create the required plots and save summary metrics.
		"""
		# Load the models and fit to data.
		fit = FitToData( sys_name = self.sys_name,
						ofold_config = self.ofold_config,
						topology = self.topology,
						mode = self.pred_mode,
						system_features = system_features,
						ofold_output_dir = self.ofold_output_dir,
						modeling_output_dir = self.modeling_output_dir,
						prec = self.prec,
						seed_worker = self.seed_worker,
						device = self.device )

		# If the stats file form simulation output doesn't already exist.
		if os.path.exists( self.stats_file ):
			stats_dict = np.load( self.stats_file, allow_pickle = True ).item()
		else:
			fit.forward()
			# Store metrics metadata.
			fit.store_metrics_metadata()
			stats_dict = fit.stats_dict

			self.save_stats( stats_dict )

		self.save_sampling_results( stats_dict = stats_dict,
									restraint_features = restraint_features )

		return fit


	def save_sampling_results( self, stats_dict: Dict[str, Dict],
								restraint_features: Dict[str, Dict] ):
		"""
		Create relevant plots for the sampling output and save summary metrics.
		"""
		loss_dict = copy.deepcopy( stats_dict["loss"] )
		metrics_dict = copy.deepcopy( stats_dict["metrics"] )
		metadata = copy.deepcopy( stats_dict["metadata"] )

		self.plot_metrics( loss_dict = loss_dict,
							metrics_dict = metrics_dict )
		self.write_summary( loss_dict = loss_dict,
							metrics_dict = metrics_dict,
							metadata = metadata,
							restraint_features = restraint_features )


	def run_analysis( self, fit: FitToData ):
		"""
		Instantiate and run the Analysis module.
		"""
		# Model IDs are just the epoch numbers.
		models_ids = np.arange( 0, self.topology.train.max_epochs, 1 )
		Assay(
			sys_name = self.sys_name,
			# base_dir = self.base_dir,
			model_ids = models_ids,
			model_dir = fit.ensemble_dir,
			# ensmeble_file = f"{fit.ensemble_file}.pdb",
			output_dir = self.modeling_output_dir,
			seed_worker = self.seed_worker,
			cores = self.cpu_cores,
			prec = self.prec
		 ).forward()



	################################################################################
	################################################################################
	def add_sys_conf_to_topology( self ):
		"""
		Add the system configs to the topology dict.
		"""
		# Load the system specific configs.
		sys_conf = read_json(
							os.path.join( self.data_dir, self.sys_config_file )
							)
		# Add system specific config to the topology dict.
		sys_name = list( sys_conf.keys() )[0]
		self.topology.system = sys_conf[sys_name]


	def create_dirs( self ):
		"""
		Check if required directories exist or not.
		Create all the required directories.
		"""
		if not os.path.exists( self.base_dir ):
			raise RuntimeError( f"Base directory - {self.base_dir} does not exist..." )
		if not os.path.exists( self.data_dir ):
			raise RuntimeError( f"Data directory - {self.data_dir} does not exist..." )

		os.makedirs( self.base_modeling_dir, exist_ok = True )
		os.makedirs( self.sys_modeling_dir, exist_ok = True )
		# os.makedirs( self.modeling_mode, exist_ok = True )
		os.makedirs( self.modeling_output_dir, exist_ok = True )



	def create_required_paths_dirs( self ):
		"""
		Given the base_dir, create all the required paths.
		"""
		# Dir to store modeling outputs.
		self.base_modeling_dir = os.path.join( self.base_dir, self.modeling_dir_name )

		# # Path to the OpenFold dir.
		# self.openfold_dir = os.path.join( os.path.abspath( "./openfold/" ) )
		# # Path to the OpenFold params to be used.
		# self.openfold_params = os.path.join(
		# 						os.path.abspath( f"openfold/resources/params/params_{self.config_preset}.npz" )
		# 						)

		# Directory containing the fasta file for the system to be modeled.
		self.fasta_dir = os.path.abspath( 
							os.path.join( self.base_dir, f"{self.sys_name}_fasta_dir/" )
							)
		# Output directory path for OpenFold output.
		self.ofold_output_dir = os.path.abspath(
								os.path.join( self.data_dir, f"{self.sys_name}_output" )
								)
		# Directory storing the precomputed alignments.
		self.alignment_dir = os.path.join( self.ofold_output_dir, "alignments" )

		# Path for the OpenFold inference script.
		# self.script = os.path.abspath( "./openfold/run_pretrained_openfold.py" )

		# Create directory to store output.
		# 	separate directory is created for mode = test/prod.
		version = self.topology.train.version
		# mode = self.topology.train.mode

		# self.modeling_mode = os.path.join( self.base_dir, f"{mode}" )

		# System specific dir modeling outputs.
		self.sys_modeling_dir = os.path.join( self.base_modeling_dir, self.sys_name )
		# Dir to store all modeling results for a version.
		self.modeling_output_dir = os.path.join( self.sys_modeling_dir,
												f"version_{version}/" )

		self.topology_file = os.path.join( self.modeling_output_dir, f"topology_{version}.json" )
		self.objective_file = os.path.join( self.modeling_output_dir, f"objective_{version}.txt" )

		# File path for the loss plot.
		self.loss_plot_file = os.path.join( self.modeling_output_dir, "Loss.png" )
		# File path for the metric plot.
		self.metrics_plot_file = os.path.join( self.modeling_output_dir, "Metrics.png" )
		# File path for XL map plot.
		self.xl_map_plot_file = os.path.join( self.modeling_output_dir, "XL_map.png" )

		# File path for the stats file.
		self.stats_file = os.path.join( self.modeling_output_dir, "Stats.npy" )
		# # File path for the loss dict.
		# self.loss_dict_file = os.path.join( self.modeling_output_dir, "Loss.npy" )
		# # File path for the metrics dict.
		# self.scalar_metric_dict_file = os.path.join( self.modeling_output_dir, "Metrics_scalar.npy" )
		# # File path for the metrics dict.
		# self.other_metric_dict_file = os.path.join( self.modeling_output_dir, "Metrics_other.npy" )

		# Output summary file.
		self.summary_file = os.path.join( self.modeling_output_dir, "Summary.csv" )



	def write_objective_file( self ):
		"""
		Write down the system used, date, and objective of the simulation.
		"""
		w = open_file_handler( self.objective_file, "w" )
		with subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE ) as proc:
			system = proc.communicate()[0]
		with subprocess.Popen( "date", shell = True, stdout = subprocess.PIPE ) as proc:
			sys_date = proc.communicate()[0]
		w.writelines( f"System = {system} \t Date = {sys_date}\n" )
		w.writelines( f"Objective: {self.topology.objective}" )
		w.close()


	def output_dir_exists( self ):
		"""
		Check if the output directory exists or not.
		Just to avoid accidently overwriting.
		"""
		
		if not self.disable_overwrite_warning:
			if os.path.exists( self.modeling_output_dir ):
				overwrite = input( f"Output directory: '{self.modeling_output_dir}' exists. Wanna continue (Y or n)? " )
				if overwrite:
					pass
				else:
					sys.exit()
			else:
				os.makedirs( self.modeling_output_dir, exist_ok = True )


	################################################################################
	################################################################################
	def save_topology_file( self ):
		"""
		Save the topology file in the output directory.
		"""
		write_configdict_to_json( self.topology,
									self.topology_file
								 )


	def save_stats( self, stats_dict: Dict[str, Dict] ):
		"""
		Save the stats dict on disk.
		"""
		np.save( self.stats_file, stats_dict, allow_pickle = True )
		# np.save( self.loss_dict_file, loss_dict, allow_pickle = True )
		# np.save( self.scalar_metric_dict_file, scalar_metric_dict, allow_pickle = True )
		# np.save( self.other_metric_dict_file, other_metric_dict, allow_pickle = True )



	def plot_metrics( self,
						loss_dict: Dict[str, float],
						metrics_dict: Dict[str, float] ):
		"""
		Create plots for all metrics.
		"""
		create_plot_from_dict( loss_dict, self.loss_plot_file )
		plot_scalar_metrics( metrics_dict, self.metrics_plot_file )



	def write_summary( self,
						loss_dict: Dict[str, float],
						metrics_dict: Dict[str, float],
						metadata: Dict,
						restraint_features: Dict[str, Dict] ):
		"""
		For all loss and metrics, compute the following:
			epoch0 - value at 0th epoch.
			last_epoch - value at the last epoch.
			avg - average value across all epochs.
			avg_first_0.1 - average value across 1st 10% epochs.
			avg_last_0.1 - average value across last 10% epochs.
			global satisfaction for all metrics.
		Write the results to a csv file.
		"""
		df_dict = {"labels": []}
		df_dict["labels"] = ["epoch0", "last_epoch", "avg", "avg_first_0.1",
								"avg_last_0.1"]
		num_per_epoch_labels = len( df_dict["labels"] )
		df_dict["labels"].extend( [f"{k}_global_satisfaction" for k in metrics_dict.keys()] )
		num_global_labels = len( df_dict["labels"] ) - num_per_epoch_labels

		df_dict.update( {k:[] for k in loss_dict.keys()} )
		last_n = math.ceil( self.topology.train.max_epochs*0.9 )
		first_n = math.ceil( self.topology.train.max_epochs*0.1 )

		for k, v in loss_dict.items():
			df_dict[k].extend(
							[v[0],
							v[-1],
							round( np.mean( v ), self.prec ),
							round( np.mean( v[:first_n] ), self.prec ),
							round( np.mean( v[last_n:] ), self.prec )]
							)
			df_dict[k].extend( "" for i in range( num_global_labels ) )

		print( metadata.keys() )
		print( metadata["xlr"].keys() )
		df_dict.update( {f"{k}_metric":[] for k in metrics_dict.keys()} )
		for k, v in metrics_dict.items():
			df_dict[f"{k}_metric"].extend(
							[v[0],
							v[-1],
							np.mean( v ),
							np.mean( v[:first_n] ),
							np.mean( v[last_n:] ),
							round( metadata[k]["global_satisfaction"], self.prec )
							]
							)


			# if k == "xlr":
			# 	global_xl_satisfied = int( torch.count_nonzero( other_metric_dict[k] ) )
			# 	total_xls = restraint_features["xl_restraint"]["total_xls"]
			# 	print( global_xl_satisfied, "  ", total_xls )
			# 	global_xl_satisfied = round( global_xl_satisfied/total_xls, self.prec )


		df = pd.DataFrame( df_dict )
		df.to_csv( self.summary_file, index = False )


if __name__ == "__main__":
	sys_name = "8gtj"
	topology_dict = topology_dict()
	base_dir = os.path.join( "./benchmark/" )
	# Directory containing input data for the modeled system.
	data_dir = os.path.join( base_dir,
							f"afu_benchmark/{sys_name}/" )
	modeling_dir_name = "modeling"

	IntegrativeLearning( sys_name,
						base_dir,
						data_dir,
						f"sys_config_{sys_name}.json",
						# base_path,
						modeling_dir_name,
						topology_dict ).forward()

