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
		self.base_dir = os.path.abspath( f"./{self.sys_name}/" )
		# Path to the OpenFold dir.
		self.openfold_dir = os.path.abspath( "./openfold/" )
		# Directory containing the fasta file for the system to be modeled.
		self.fasta_dir = os.path.abspath( f"{self.base_dir}/fasta_dir/" )
		# Output directory path for OpenFold output.
		self.output_dir = os.path.abspath( f"{self.base_dir}/{self.sys_name}_output/" )
		# Directory storing the precomputed alignments.
		self.alignment_dir = os.path.abspath( f"{self.output_dir}/alignments/" )
		# OpenFold predicted initial structure.
		# self.init_struct_pdb = glob.glob( f"{self.output_dir}/predictions/*relaxed.pdb" )[0]
		# OpenFold predicted initial structure in CIF format.
		# self.init_struct_cif = os.path.abspath( f"{self.output_dir}/predictions/{self.sys_name}.cif" )



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
											# init_struct_pdb = self.init_struct_pdb,
											# init_struct_cif = self.init_struct_cif, 
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

		# Obtain the initial structure.
		# self.get_initial_structure()

		# self.get_system_embeddings()

		# # Now loading the models.
		# self.load_model()

		toc = time.time()
		with open( f"./Time_taken.txt", "w" ) as w:
			w.writelines( f"Time taken: {( toc-tic )/3600} hours" )
		print( f"Time taken: {( toc-tic )/3600} hours OR {( toc-tic )/60} minutes" )


	# def get_initial_structure( self ):
	# 	"""
	# 	Run monomer/multimer prediction based on mode.
	# 	openfold/script_utils.run_model has been modified ot save the 
	# 		MSA, Pair, and Single representations on disk.

	# 	Input:
	# 	----------
	# 	Does not take any arguments.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	if self.mode == 'mono':
	# 		self.run_monomer_prediction()
	# 	elif self.mode == 'multi':
	# 		self.run_multimer_prediction()



	# def get_system_embeddings( self ):
	# 	"""
	# 	Obtain the MSA, Pair, and Single representation from OpenFold.
	# 	These are stored in a .pkl file in the OpenFold output directory.

	# 	Input:
	# 	----------
	# 	Does not take any arguments.

	# 	Returns:
	# 	----------
	# 	msa_rep --> MSA representation for the system [N, L, 256].
	# 	pair_rep --> Pair representation for the system [L, L, 128].
	# 	single_rep --> Single representation for the system [L, 384].
	# 	( N --> no. of seq in MSA; L --> no. of residues in system. )
	# 	"""
	# 	pkl_path = glob.glob( f"{self.output_dir}/predictions/*.pkl" )
	# 	if len( pkl_path ) == 0:
	# 		raise Exception( f"Incorrect path -- {pkl_path}..." )

	# 	pkl_path = pkl_path[0]

	# 	with open( pkl_path, "rb" ) as f:
	# 		ofold_output = pkl.load( f )

	# 	msa_rep = ofold_output["msa"]
	# 	pair_rep = ofold_output["pair"]
	# 	single_rep = ofold_output["single"]

	# 	print( msa_rep.shape, "  ", pair_rep.shape, "  ", single_rep.shape )
	# 	return msa_rep, pair_rep, single_rep



	# def run_monomer_prediction( self ):
	# 	"""
	# 	Use OpenFold for monomer structure prediction.
	# 		Use precomputed alignments if available.

	# 	Input:
	# 	----------
	# 	training --> (bool) specifies if OpenFold is to be used for training/prediction.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""

	# 	# if training:
	# 	# 	script = "unfrozen_pretrained.py"
	# 	# else:
	# 	# 	script = "run_pretrained_openfold.py"

	# 	# Move to OpenFold dir.
	# 	os.chdir( self.openfold_dir )
		
	# 	if os.path.exists( self.alignment_dir ):
	# 		print( "using precomputed alignments for monomer prediction..." )
	# 		command = monomer_precomp_aln_cmd( self.script, self.fasta_dir, 
	# 										self.alignment_dir, self.output_dir,
	# 										self.config_preset, self.ckpt_path,
	# 										self.cpu_cores, self.device )

	# 	else:
	# 		command = monomer_cmd( self.script, self.fasta_dir, self.output_dir,
	# 								self.config_preset,  self.ckpt_path,
	# 								self.cpu_cores, self.device )

	# 	print( "Running monomer prediction..." )
	# 	run_subprocess( command )

	# 	# Move back to base dir.
	# 	os.chdir( self.base_dir )



	# def run_multimer_prediction( self ):
	# 	"""
	# 	Use OpenFold for multimer structure prediction.
	# 		Use precomputed alignments if available.

	# 	Input:
	# 	----------
	# 	training --> (bool) specifies if OpenFold is to be used for training/prediction.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	# Move to OpenFold dir.
	# 	os.chdir( self.openfold_dir )

	# 	if os.path.exists( self.alignment_dir ):
	# 		print( "using precomputed alignments for multimer prediction..." )
	# 		command = multimer_precomp_aln_cmd( self.script, self.fasta_dir, 
	# 											self.alignment_dir, self.output_dir,
	# 											self.config_preset, self.cpu_cores, self.device )

	# 	else:
	# 		command = multimer_cmd( self.script, self.fasta_dir, self.output_dir,
	# 								self.config_preset, self.cpu_cores, self.device )

	# 	print( "Running multimer prediction..." )
	# 	run_subprocess( command )

	# 	# Move back to base dir.
	# 	os.chdir( self.base_dir )


	# def load_model( self ):
	# 	"""
	# 	Load the Structure module, pLDDT and TM heads.
	# 	Intialize with the pretrained weights.

	# 	Input:
	# 	----------
	# 	Does not take any arguments.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	sm_weights, aux_heads_weights  = self.get_pretrained_weights()
		
	# 	# Instantiate the structure module and aux heads.
	# 	self.load_structure_module()
	# 	self.load_aux_modules()

	# 	# Initialize the models with the pretrained weights.
	# 	self.structure_module.load_state_dict( sm_weights )
	# 	self.aux_heads.load_state_dict( aux_heads_weights )

	# 	# Set the aux heads to eval mode.
	# 	self.aux_heads.eval()

	# 	# Set the structure module to train mode.
	# 	self.structure_module.train()


	# def load_structure_module( self ):
	# 	"""
	# 	Initialize the Structure module.

	# 	Input:
	# 	----------
	# 	Does not take any arguments.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	self.structure_module = StructureModule(
	# 		is_multimer = self.config.globals.is_multimer,
	# 		**self.config["model"]["structure_module"]
	# 	)


	# def load_aux_modules( self ):
	# 	"""
	# 	Initialize the Axillary heads module.

	# 	Input:
	# 	----------
	# 	Does not take any arguments.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	self.aux_heads = AuxiliaryHeads(
	# 	    self.config["model"]["heads"],
	# 	)


	# def get_pretrained_weights( self ):
	# 	"""
	# 	Get the weights for the pretrained OpenFold monomer and multimer models.
	# 	Extract weights for only Structure modeule, pLDDT head, and TM head.

	# 	Input:
	# 	----------
	# 	Does not take any arguments.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	if self.mode == "mono":
	# 		pretrained_weights = torch.load( "../monomer_params.pt" )

	# 	elif self.mode == "multi":
	# 		pretrained_weights = torch.load( os.path.abspath( "../multimer_params.pt" ) )

	# 	# Obtain weights for the structure module only.
	# 	sm_weights = OrderedDict( 
	# 		( ".".join( key.split( "." )[1:] ), pretrained_weights[key] ) 
	# 		for key in pretrained_weights.keys() if "structure_module" in key 
	# 		)
	# 	# Obtain weights for the auxillary heads module.
	# 	aux_heads_weights = OrderedDict( 
	# 		( ".".join( key.split( "." )[1:] ), pretrained_weights[key] ) 
	# 		for key in pretrained_weights.keys() if "aux_heads" in key 
	# 		)

	# 	return sm_weights, aux_heads_weights



if __name__ == "__main__":
	IntegrativeLearning().forward()

