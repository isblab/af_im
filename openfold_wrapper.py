import torch
import os
import subprocess
import glob
import pickle as pkl
import time

from openfold.config import model_config
from system_config import system_config

from system_representation import SystemRepresentation
from fit_to_data import FitToData

class IntegrativeLearning():
	def __init__( self ):
		# Name of the system to be modeled.
		self.sys_name = "2ayo"
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
			self.ckpt_path = "openfold/resources/openfold_params/finetuning_ptm_2.pt"
		elif self.mode == "multi":
			# config_preset for multimer.
			self.config_preset = "model_1_multimer_v3"
			self.ckpt_path = None
		else:
			raise Exception( f"Invalid mode = {self.mode} specififed..." )
		
		# Load OpenFold configs file.
		self.ofold_config = model_config( self.config_preset )
		# Load the system specific configs.
		self.sys_config = system_config()

		# Path to the OpenFold params to be used.
		self.openfold_params = os.path.abspath( f"openfold/resources/params/params_{self.config_preset}.npz" )
		# Main directory for the modeled system.
		self.base_dir = os.path.abspath( f"./benchmark/{self.sys_name}/" )
		# Path to the OpenFold dir.
		self.openfold_dir = os.path.abspath( "./openfold/" )
		# Directory containing the fasta file for the system to be modeled.
		self.fasta_dir = os.path.abspath( f"{self.base_dir}/fasta_dir/" )
		# Output directory path for OpenFold output.
		self.output_dir = os.path.abspath( f"{self.base_dir}/{self.sys_name}_output/" )
		# Directory storing the precomputed alignments.
		self.alignment_dir = os.path.abspath( f"{self.output_dir}/alignments/" )



	def forward( self ):
		tic = time.time()
		# The base directory should exist.
		if not os.path.exists( self.base_dir ):
			raise Exception( f"Base dir: {self.base_dir}  does not exist..." )
		# Move to the base directory.
		os.chdir( self.base_dir )

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
		# Move back to base dir.
		os.chdir( self.base_dir )

		# Load the models and fit to data.
		FitToData( ofold_config = self.ofold_config, 
					sys_config = self.sys_config,
					mode = self.mode,
					system_features = system_features,
					output_dir = self.output_dir ).forward()

		toc = time.time()
		with open( f"./Time_taken.txt", "w" ) as w:
			w.writelines( f"Time taken: {( toc-tic )/3600} hours" )
		print( f"Time taken: {( toc-tic )/3600} hours OR {( toc-tic )/60} minutes" )


if __name__ == "__main__":
	IntegrativeLearning().forward()

