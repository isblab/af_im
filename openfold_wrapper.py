import torch
import os
import subprocess
import glob
import json
import ml_collections as mlc
import pickle as pkl
import time

from openfold.config import model_config
from topology import topology_dict

from data_gathering import DataGathering
from system_representation import SystemRepresentation
from fit_to_data import FitToData
from create_plots import plot_loss
from utils import read_configdict_from_json

from typing import Dict


class IntegrativeLearning():
	def __init__( self ):
		# Name of the system to be modeled.
		self.sys_name = "2ayo"
		# Main directory for the modeled system.
		self.base_dir = os.path.join( os.path.abspath( f"./benchmark/{self.sys_name}/" ) )
		# mono/multi
		self.mode = "multi"
		# Path for the OpenFold inference script.
		self.script = os.path.abspath( "./openfold/run_pretrained_openfold.py" )
		# No. of CPU cores to be used.
		self.cpu_cores = 4
		# cpu/cuda
		self.device = "cuda"
		if self.mode == "mono":
			# config_preset for monomer.
			self.config_preset = "model_1_ptm"
			self.ckpt_path = os.path.abspath( "./openfold/resources/openfold_params/finetuning_ptm_2.pt" )
		elif self.mode == "multi":
			# config_preset for multimer.
			self.config_preset = "model_1_multimer_v3"
			self.ckpt_path = None
		else:
			raise Exception( f"Invalid mode = {self.mode} specififed..." )

		# Load OpenFold configs file.
		self.ofold_config = model_config( self.config_preset )
		# Load the full system specific configs.
		self.topology = topology_dict()

		self.create_required_paths()



	def forward( self ):
		tic = time.time()
		# The base directory should exist.
		if not os.path.exists( self.base_dir ):
			raise Exception( f"Base dir: {self.base_dir}  does not exist..." )
		# Move to the base directory.
		os.chdir( self.base_dir )

		# Get the restraint features.
		# restraint_features = DataGathering( sys_name = self.sys_name,
		# 							 		base_dir = self.base_dir,
		# 							 		fasta_dir = self.fasta_dir,
		# 							 		sys_config = self.topology.system
		# 							 		).forward()

		# Get an initial structure and the ground truth features.
		system_features = SystemRepresentation( sys_name = self.sys_name,
											ofold_dir = self.openfold_dir, 
											ofold_script = self.script,
											fasta_dir = self.fasta_dir, 
											alignment_dir = self.alignment_dir, 
											output_dir = self.output_dir,
											config_preset = self.config_preset, 
											ckpt_path = self.ckpt_path,
											mode = self.mode,
											cpu_cores = self.cpu_cores,
											device = self.device ).forward()
		
		# Add restraint features to system features dict.
		# system_features["restraint_features"] = restraint_features
		# Move back to base dir.
		os.chdir( self.base_dir )

		# Load the models and fit to data.
		fit = FitToData( ofold_config = self.ofold_config, 
						sys_config = self.topology,
						mode = self.mode,
						system_features = system_features,
						output_dir = self.output_dir )
		fit.forward()

		plot_loss( fit.loss_dict, self.loss_plot_file )

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
		self.output_dir = os.path.join( self.base_dir, f"{self.sys_name}_output" ) # os.path.abspath( f"{self.base_dir}/{self.sys_name}_output/" )
		# Directory storing the precomputed alignments.
		self.alignment_dir = os.path.join( self.output_dir, "alignments" ) # os.path.abspath( f"{self.output_dir}/alignments/" )

		# File name for the loss plot.
		self.loss_plot_file = os.path.join( self.base_dir, "Loss.png" ) # f"{self.base_dir}/Loss.png"



if __name__ == "__main__":
	IntegrativeLearning().forward()

