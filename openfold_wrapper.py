"""
Contains a wrapper class that runs all the stages of integrative modeling pipeline.
	1. Data gathering
	2 System representation
	3. Sampling --> Fit to data
	4. Analysis
"""

import math
import time
import random
from typing import Dict
import os
import sys
import subprocess
import numpy as np
import pandas as pd

import torch

from openfold.config import model_config
from topology import topology_dict

from data_gathering import DataGathering
from system_representation import SystemRepresentation
from fit_to_data import FitToData
from assay import Assay
from create_plots import create_plot_from_dict, plot_scalar_metrics, plot_xl_map
from utils import read_json, write_configdict_to_json, open_file_handler


class IntegrativeLearning():
	"""
	A wrapper class that runs all the stages of integrative modeling pipeline.
	"""
	def __init__( self ):
		# Name of the system to be modeled.
		self.sys_name = "2ayo" # H1129
		# Main directory for the modeled system.
		self.base_dir = os.path.join( os.path.abspath( f"./benchmark/{self.sys_name}/" ) )
		# mono/multi
		self.pred_mode = "multi"
		# Path for the OpenFold inference script.
		self.script = os.path.abspath( "./openfold/run_pretrained_openfold.py" )
		# No. of CPU cores to be used.
		self.cpu_cores = 16
		self.prec = 4
		# cpu/cuda
		self.device = "cuda"
		if self.pred_mode == "mono":
			# config_preset for monomer.
			self.config_preset = "model_1_ptm"
			self.ckpt_path = os.path.abspath( "./openfold/resources/openfold_params/finetuning_ptm_2.pt" )
		elif self.pred_mode == "multi":
			# config_preset for multimer.
			self.config_preset = "model_1_multimer_v3"
			self.ckpt_path = None
		else:
			raise ValueError( f"Invalid mode = {self.pred_mode} specififed..." )

		# Load OpenFold configs file.
		self.ofold_config = model_config( self.config_preset )
		# Load the full system specific configs.
		self.topology = topology_dict()

		self.create_required_paths()


	def seed_worker( self ):
		"""
		Seed for PRNG.
		"""
		seed = 1
		torch.manual_seed( seed )
		# torch.cuda.manual_seed( worker_seed )
		torch.cuda.manual_seed_all( seed )
		np.random.seed( seed )
		random.seed( seed )


	def forward( self ):
		"""
		Serially run all the stages of the pipeline.
		"""
		tic = time.time()
		self.seed_worker()
		# The base directory should exist.
		if not os.path.exists( self.base_dir ):
			raise FileNotFoundError( f"Base dir: {self.base_dir}  does not exist..." )
		# Move to the base directory.
		os.chdir( self.base_dir )

		print( "\n----------------------------------------------------------------------\n" +
				"--------------------------- Data gathering ---------------------------\n" +
				"----------------------------------------------------------------------\n" )

		# Get the restraint features.
		data_gathering = DataGathering( sys_name = self.sys_name,
									 		base_dir = self.base_dir,
									 		fasta_dir = self.fasta_dir,
									 		sys_config = self.topology.system )
		data_gathering.forward()
		restraint_features = data_gathering.restraint_features


		print( "\n----------------------------------------------------------------------\n" +
				"----------------------- System representation ------------------------\n" +
				"----------------------------------------------------------------------\n" )
		# Get an initial structure and the ground truth features.
		system_features = SystemRepresentation( sys_name = self.sys_name,
											ofold_dir = self.openfold_dir,
											ofold_script = self.script,
											fasta_dir = self.fasta_dir,
											alignment_dir = self.alignment_dir,
											ofold_output_dir = self.ofold_output_dir,
											config_preset = self.config_preset,
											ckpt_path = self.ckpt_path,
											mode = self.pred_mode,
											cpu_cores = self.cpu_cores,
											seed_worker = self.seed_worker,
											device = self.device ).forward()

		# Add restraint features to system features dict.
		system_features["restraint_features"] = restraint_features
		# Move back to base dir.
		os.chdir( self.base_dir )


		print( "\n----------------------------------------------------------------------\n" +
				"---------------------------- Fit to Data -----------------------------\n" +
				"----------------------------------------------------------------------\n" )
		# Load the models and fit to data.
		fit = FitToData( ofold_config = self.ofold_config,
						topology = self.topology,
						mode = self.pred_mode,
						system_features = system_features,
						ofold_output_dir = self.ofold_output_dir,
						output_dir = self.output_dir,
						prec = self.prec,
						seed_worker = self.seed_worker,
						device = self.device )

		# If the simulation output doesn;t already exist.
		if not fit.ensemble_exists():
			fit.forward()

			self.save_topology_file()

			self.save_metrics( fit.loss_dict,
								fit.scalar_metric_dict,
								fit.other_metric_dict )
			self.plot_metrics( fit.loss_dict,
								fit.scalar_metric_dict,
								fit.other_metric_dict, restraint_features )
			self.write_summary( fit.loss_dict, fit.scalar_metric_dict,
								fit.other_metric_dict, restraint_features )


		print( "\n----------------------------------------------------------------------\n" +
				"-------------------------- \033[9m Analysis \033[0m Assay --------------------------\n"
				"----------------------------------------------------------------------\n" )
		# Model IDs are just the epoch numbers.
		models_ids = np.arange( 0, self.topology.train.max_epochs, 1 )
		Assay(
			sys_name = self.sys_name,
			base_dir  =self.base_dir,
			cores = self.cpu_cores,
			prec = self.prec,
			model_ids = models_ids,
			ensmeble_dir = fit.relax_ensemble_dir,
			# ensmeble_file = f"{fit.ensemble_file}.pdb",
			output_dir = self.output_dir
		 ).forward()


		toc = time.time()
		time_file = os.path.join( self.output_dir, "Time_taken.txt" )
		if not os.path.exists( time_file ):
			w = open_file_handler( time_file, "w" )
			w.writelines( f"Time taken: {( toc-tic )/3600} hours OR {( toc-tic )/60} minutes" )
		print( "May the Force be with you.." )
		print( f"Time taken: {( toc-tic )/3600} hours OR {( toc-tic )/60} minutes" )



	def create_required_paths( self ):
		"""
		Given the base_dir, create all the required paths.
		"""
		# Path to the OpenFold dir.
		self.openfold_dir = os.path.join( os.path.abspath( "./openfold/" ) )
		# Path to the OpenFold params to be used.
		self.openfold_params = os.path.join(
								os.path.abspath( f"openfold/resources/params/params_{self.config_preset}.npz" )
								)
		# Load the system specific configs.
		sys_conf = read_json(
							os.path.join( self.base_dir, f"sys_conf_{self.sys_name}.json" )
							)
		# Add system specific config to the topology dict.
		sys_name = list( sys_conf.keys() )[0]
		self.topology.system = sys_conf[sys_name]

		# Directory containing the fasta file for the system to be modeled.
		self.fasta_dir = os.path.join( self.base_dir, "fasta_dir" )
		# Output directory path for OpenFold output.
		self.ofold_output_dir = os.path.join( self.base_dir, f"{self.sys_name}_output" )
		# Directory storing the precomputed alignments.
		self.alignment_dir = os.path.join( self.ofold_output_dir, "alignments" )

		# Create directory to store output.
		# 	separate directory is created for mode = test/prod.
		version = self.topology.train.version
		mode = self.topology.train.mode

		dir_ = os.path.join( self.base_dir, f"{mode}" )
		if not os.path.exists( dir_ ):
			os.makedirs( dir_ )

		self.output_dir = os.path.join( self.base_dir, f"{mode}/version_{version}" )

		self.output_dir_exists()

		self.topology_file = os.path.join( self.output_dir, f"topology_{version}.json" )
		self.objective_file = os.path.join( self.output_dir, f"objective_{version}.txt" )

		# File path for the loss plot.
		self.loss_plot_file = os.path.join( self.output_dir, "Loss.png" )
		# File path for the metric plot.
		self.scalar_metric_plot_file = os.path.join( self.output_dir, "Metrics.png" )
		# File path for XL map plot.
		self.xl_map_plot_file = os.path.join( self.output_dir, "XL_map.png" )

		# File path for the loss dict.
		self.loss_dict_file = os.path.join( self.output_dir, "Loss.npy" )
		# File path for the metrics dict.
		self.scalar_metric_dict_file = os.path.join( self.output_dir, "Metrics_scalar.npy" )
		# File path for the metrics dict.
		self.other_metric_dict_file = os.path.join( self.output_dir, "Metrics_other.npy" )

		# Output summary file.
		self.summary_file = os.path.join( self.output_dir, "Summary.csv" )

		# Write down the system used, date, and objective of the simulation.
		w = open_file_handler( self.objective_file, "w" )
		with subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE ) as proc:
			system = proc.communicate()[0]
		with subprocess.Popen( "date", shell = True, stdout = subprocess.PIPE ) as proc:
			sys_date = proc.communicate()[0]
		w.writelines( f"System = {system} \t Date = {sys_date}\n" )
		w.writelines( f"Objective: {self.topology.objective}" )


	def output_dir_exists( self ):
		"""
		Check if the output directory exists or not.
		Just to avoid accidently overwriting.
		"""
		if os.path.exists( self.output_dir ):
			overwrite = input( f"Output directory: '{self.output_dir}' exists. Wanna continue (Y or n)? " )
			if overwrite:
				pass
			else:
				sys.exit()
		else:
			os.makedirs( self.output_dir )



	def save_topology_file( self ):
		"""
		Save the topology file in the output directory.
		"""
		write_configdict_to_json( self.topology,
									self.topology_file
								 )


	def save_metrics( self, loss_dict: Dict[str, float],
							scalar_metric_dict: Dict[str, float],
							other_metric_dict: Dict[str, float] ):
		"""
		Save the loss and metric dict on disk.
		Also create their plots.
		"""
		np.save( self.loss_dict_file, loss_dict, allow_pickle = True )
		np.save( self.scalar_metric_dict_file, scalar_metric_dict, allow_pickle = True )
		np.save( self.other_metric_dict_file, other_metric_dict, allow_pickle = True )



	def plot_metrics( self, loss_dict: Dict[str, float],
							scalar_metric_dict: Dict[str, float],
							other_metric_dict: Dict[str, float],
							restraint_features: Dict ):
		"""
		Create plots for all metrics.
		"""
		create_plot_from_dict( loss_dict, self.loss_plot_file )
		plot_scalar_metrics( scalar_metric_dict, self.scalar_metric_plot_file )
		xl_res_mask = restraint_features["xl_restraint"]["xl_res_mask"]
		plot_xl_map( other_metric_dict["xlr"], xl_res_mask, self.xl_map_plot_file )



	def write_summary( self, loss_dict: Dict[str, float],
							scalar_metric_dict: Dict[str, float],
							other_metric_dict: Dict,
							restraint_features ):
		"""
		Write all relevant losses and metrics to a csv file.
		"""
		df_dict = {"labels": []}
		df_dict["labels"] = ["epoch0", "last_epoch", "avg", "avg_first_0.1",
								"avg_last_0.1", "global_xl_satisfied"]

		df_dict.update( {k:[] for k in loss_dict.keys()} )
		last_n = math.ceil( self.topology.train.max_epochs*0.9 )
		first_n = math.ceil( self.topology.train.max_epochs*0.1 )

		for k, v in loss_dict.items():
			df_dict[k].extend(
							[v[0],
							v[-1],
							round( np.mean( v ), self.prec ),
							round( np.mean( v[:first_n] ), self.prec ),
							round( np.mean( v[last_n:] ), self.prec ),
							""]
							)

		df_dict.update( {f"{k}_metric":[] for k in scalar_metric_dict.keys()} )
		for k, v in scalar_metric_dict.items():
			if k == "xlr":
				global_xl_satisfied = int( torch.count_nonzero( other_metric_dict[k] ) )
				xl_res_mask = restraint_features["xl_restraint"]["xl_res_mask"]
				total_xls = int( torch.count_nonzero( xl_res_mask ) )
				print( global_xl_satisfied, "  ", total_xls )
				global_xl_satisfied = round( global_xl_satisfied/total_xls, self.prec )

			df_dict[f"{k}_metric"].extend(
							[v[0],
							v[-1],
							np.mean( v ),
							np.mean( v[:first_n] ),
							np.mean( v[last_n:] ),
							global_xl_satisfied]
							)

		df = pd.DataFrame( df_dict )
		df.to_csv( self.summary_file, index = False )


if __name__ == "__main__":
	IntegrativeLearning().forward()
