"""
We extract the following features using the OpenFold pipeline:
aatype
between_segment_residues
residue_index
seq_length
all_atom_positions
all_atom_mask
resolution
is_distillation
asym_id
sym_id
entity_id
seq_mask
gt_features
	all_atom_positions: --> [480, 37, 3]
	all_atom_mask: --> [480, 37]
	asym_id: -->  [480]
	sym_id: --> [480]
	entity_id: --> 480]
	aatype: --> [480]
	atom14_atom_exists: --> [480, 14]
	residx_atom14_to_atom37: --> [480, 14]
	residx_atom37_to_atom14: --> [480, 37]
	atom37_atom_exists: --> [480, 37]
	atom14_gt_exists: --> [480, 14]
	atom14_gt_positions: --> [480, 14, 3]
	atom14_alt_gt_positions  --> [480, 14, 3]
	atom14_alt_gt_exists: --> [480, 14]
	atom14_atom_is_ambiguous: --> [480, 14]
	rigidgroups_gt_frames: --> [480, 8, 4, 4]
	rigidgroups_gt_exists: --> [480, 8]
	rigidgroups_group_exists: --> [480, 8]
	rigidgroups_group_is_ambiguous: --> [480, 8]
	rigidgroups_alt_gt_frames: --> [480, 8, 4, 4]
	torsion_angles_sin_cos: --> [480, 7, 2]
	alt_torsion_angles_sin_cos: --> [480, 7, 2]
	torsion_angles_mask: --> [480, 7]
	pseudo_beta: --> [480, 3])
	pseudo_beta_mask: --> [480]
	backbone_rigid_tensor: --> [480, 4, 4]
	backbone_rigid_mask: --> [480]
	chi_angles_sin_cos: --> [480, 4, 2]
	chi_mask: --> [480, 4]
"""
import os
import glob
from typing import Optional
import numpy as np
import ml_collections as mlc

from openfold.config import model_config
from mod_openfold import parse, process_mmcif, np_example_to_features

from utils.commands import OpenfoldCommand
from utils.utils import run_subprocess


class SystemRepresentation():
	def __init__( self,
			sys_name: str,
			sys_rep_config: mlc.ConfigDict,
			# ofold_dir: str, ofold_script: str,
			fasta_dir: str,
			alignment_dir: str,
			ofold_output_dir: str, 
			# config_preset: str,
			# ofold_db_preset: str,
			# ckpt_path: Optional[str], init_model_prefix: str,
			seed_worker,
			device: str = "cpu"
		):
		self.sys_name = sys_name
		self.openfold_dir = sys_rep_config.ofold_dir
		self.script = sys_rep_config.ofold_script
		self.db_dir = sys_rep_config.db_dir
		self.config_preset = sys_rep_config.config_preset
		self.db_preset = sys_rep_config.db_preset
		self.tool_base = sys_rep_config.ofold_tools
		self.model_ckpt = sys_rep_config.model_checkpoint

		self.init_model_prefix = sys_rep_config.init_model_prefix
		self.ofold_seed = sys_rep_config.seed
		self.cpu_cores = sys_rep_config.cpu_cores
		self.mode = sys_rep_config.mode
		self.max_template_date = sys_rep_config.max_template_date

		self.fasta_dir = fasta_dir
		self.alignment_dir = alignment_dir
		self.ofold_output_dir = ofold_output_dir
		self.device = device

		seed_worker()


	def forward( self ):
		init_model_path = glob.glob(
				f"{self.ofold_output_dir}/predictions/*{self.init_model_prefix}.cif"
				)
		if len( init_model_path ) != 0:
			print( "\nInitial structure for the system exists..." )
			print( f"Using {self.init_model_prefix} model as inital model..." )
			self.init_struct_cif = init_model_path[0]
		
		else:
			print( "\nPredicting the initial structure for the system..." )
			# if not os.path.exists( os.path.abspath( init_model_path ) ):
			self.predict_initial_structure()
			# OpenFold predicted initial structure in CIF format.
			print( f"Using {self.init_model_prefix} model as inital model..." )
			init_model_path = glob.glob(
					f"{self.ofold_output_dir}/predictions/*{self.init_model_prefix}.cif"
					)
			print( f"Init model path: {init_model_path}" )
			self.init_struct_cif = init_model_path[0]

		print( "\nCreating ground truth features from the initial structure..." )
		data = self.get_feature_from_init_struct()

		print( data.keys() )
		for k in data["gt_features"].keys():
			print( k, ": --> ", data["gt_features"][k].shape )
		return data


	def predict_initial_structure( self ):
		"""
		Run monomer/multimer prediction based on mode.
		openfold/script_utils.run_model has been modified ot save the 
			MSA, Pair, and Single representations on disk.
		**Note: To get the struct with correct residue indices in .cif format, 
				I've modified the function openfold.utils.script_utils.prep_output().

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		# Move to OpenFold dir.
		os.chdir( self.openfold_dir )
		mode = "monomer" if self.mode == "mode" else "multimer"
		
		if os.path.exists( self.alignment_dir ):
			print( f"Using precomputed alignments for {mode} prediction..." )
		else:
			self.alignment_dir = None
		
		obj = OpenfoldCommand(
			db_dir = self.db_dir,
			script = self.script,
			config_preset = self.config_preset,
			db_preset = self.db_preset,
			tool_base = self.tool_base,
			model_ckpt = self.model_ckpt,
			fasta_dir = self.fasta_dir,
			alignment_dir = self.alignment_dir,
			output_dir = self.ofold_output_dir,
			max_template_date = self.max_template_date,
			mode = self.mode,
			seed = self.ofold_seed,
			cpu_cores = self.cpu_cores,
			device = self.device
		)
		
		print( f"Running {mode} prediction..." )
		command = obj.get()
		stderr_file = os.path.join( self.ofold_output_dir, f"error_{self.sys_name}" )
		run_subprocess( command, stderr_file = stderr_file )


	def get_feature_from_init_struct( self ):
		# Parse the .cif file to get an mmcif_object.
		# This mmcif_object is not the same as Biopython structure object.
		with open( self.init_struct_cif, "r" ) as f:
		    mmcif_string = f.read()

		mmcif_object = parse( 
		    file_id = self.sys_name, mmcif_string = mmcif_string
		).mmcif_object

		print( os.path.exists( self.init_struct_cif ) )
		# Extract relevant relevant features from the mmcif_object.
		np_example = process_mmcif( mmcif_object )

		print( "Load config file..." )
		config = model_config( self.config_preset )

		# Obtain the ground truth features.
		data = np_example_to_features(
				np_example = np_example,
				config = config["data"],
				mode = "train",
				is_multimer = True )

		return data


if __name__ == "__main__":
	SystemRepresentation().forward()

