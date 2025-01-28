import numpy as np
import torch
import os
import subprocess
import glob
import json
import ml_collections as mlc
import pickle as pkl
import time
import random

from openfold.config import model_config
from topology import topology_dict

from data_gathering import DataGathering
from system_representation import SystemRepresentation
from fit_to_data import FitToData
from create_plots import plot_loss
from utils import read_configdict_from_json, write_configdict_to_json

from typing import Dict


class IntegrativeLearning():
	def __init__( self ):
		# Name of the system to be modeled.
		self.sys_name = "2ayo"
		# Main directory for the modeled system.
		self.base_dir = os.path.join( os.path.abspath( f"./benchmark/{self.sys_name}/" ) )
		# mono/multi
		self.pred_mode = "multi"
		# Path for the OpenFold inference script.
		self.script = os.path.abspath( "./openfold/run_pretrained_openfold.py" )
		# No. of CPU cores to be used.
		self.cpu_cores = 4
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
			raise Exception( f"Invalid mode = {self.pred_mode} specififed..." )

		# Load OpenFold configs file.
		self.ofold_config = model_config( self.config_preset )
		# Load the full system specific configs.
		self.topology = topology_dict()

		self.create_required_paths()


	def seed_worker( self ):
		# Seed for PRNG.
		seed = 1
		torch.manual_seed( seed )
		# torch.cuda.manual_seed( worker_seed )
		torch.cuda.manual_seed_all( seed )
		np.random.seed( seed )
		random.seed( seed )


	def forward( self ):
		tic = time.time()
		# The base directory should exist.
		if not os.path.exists( self.base_dir ):
			raise Exception( f"Base dir: {self.base_dir}  does not exist..." )
		# Move to the base directory.
		os.chdir( self.base_dir )

		# Get the restraint features.
		data_gathering = DataGathering( sys_name = self.sys_name,
									 		base_dir = self.base_dir,
									 		fasta_dir = self.fasta_dir,
									 		sys_config = self.topology.system )
		data_gathering.forward()
		restraint_features = data_gathering.restraint_features

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

		# Load the models and fit to data.
		fit = FitToData( ofold_config = self.ofold_config, 
						sys_config = self.topology,
						mode = self.pred_mode,
						system_features = system_features,
						ofold_output_dir = self.ofold_output_dir,
						output_dir = self.output_dir,
						seed_worker = self.seed_worker )
		fit.forward()

		plot_loss( fit.loss_dict, self.loss_plot_file )

		self.save_topology_file()


		toc = time.time()
		with open( f"./Time_taken.txt", "w" ) as w:
			w.writelines( f"Time taken: {( toc-tic )/3600} hours" )
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
		sys_conf = read_configdict_from_json( 
									os.path.join( self.base_dir, f"sys_conf_{self.sys_name}.json" )
									 )
		# Add system specific config to the topology dict.
		self.topology.system = sys_conf["System1"]

		# Directory containing the fasta file for the system to be modeled.
		self.fasta_dir = os.path.join( self.base_dir, "fasta_dir" ) # os.path.abspath( f"{self.base_dir}/fasta_dir/" )
		# Output directory path for OpenFold output.
		self.ofold_output_dir = os.path.join( self.base_dir, f"{self.sys_name}_output" ) # os.path.abspath( f"{self.base_dir}/{self.sys_name}_output/" )
		# Directory storing the precomputed alignments.
		self.alignment_dir = os.path.join( self.ofold_output_dir, "alignments" ) # os.path.abspath( f"{self.output_dir}/alignments/" )

		# Create directory to store output.
		# 	separate directory is created for mode = test/prod.
		version = self.topology.train.version
		mode = self.topology.train.mode

		dir_ = os.path.join( self.base_dir, f"{mode}" )
		if not os.path.exists( dir_ ):
			os.makedirs( dir_ )		
		
		self.output_dir = os.path.join( self.base_dir, f"{mode}/version_{version}" )

		if not os.path.exists( self.output_dir ):
			os.makedirs( self.output_dir )

		# File name for the loss plot.
		self.loss_plot_file = os.path.join( self.output_dir, "Loss.png" ) # f"{self.base_dir}/Loss.png"
		self.topology_file = os.path.join( self.output_dir, f"topology_{version}.json" )
		self.objective_file = os.path.join( self.output_dir, f"objective_{version}.txt" )

		with open( self.objective_file, "w" ) as w:
			w.writelines( self.topology.objective )


	def save_topology_file( self ):
		"""
		Save the topology file in the output directory.
		"""
		write_configdict_to_json( self.topology, 
									self.topology_file
								 )



if __name__ == "__main__":
	IntegrativeLearning().forward()

