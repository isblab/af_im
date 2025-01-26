import numpy as np
import os
import glob

from mod_openfold import parse, process_mmcif, np_example_to_features
from openfold.config import model_config

from commands import OpenfoldCommand #, monomer_cmd, monomer_precomp_aln_cmd,
						# multimer_cmd, multimer_precomp_aln_cmd )
from utils import run_subprocess

from typing import Optional

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


class SystemRepresentation():
	def __init__( self, sys_name: str, ofold_dir: str, ofold_script: str, 
					fasta_dir: str, alignment_dir: str, 
					ofold_output_dir: str, config_preset: str, 
					ckpt_path: Optional[str], mode: str,
					# init_struct_pdb: str, init_struct_cif: str,
					cpu_cores: int, seed_worker, device: str = "cpu" ):
		self.sys_name = sys_name
		self.openfold_dir = ofold_dir
		self.script = ofold_script
		self.fasta_dir = fasta_dir 
		self.alignment_dir = alignment_dir
		self.ofold_output_dir = ofold_output_dir
		self.config_preset = config_preset
		self.ckpt_path = ckpt_path
		self.cpu_cores = cpu_cores
		self.device = device
		self.mode = mode

		seed_worker()


	def forward( self ):
		if len( glob.glob( f"{self.ofold_output_dir}/predictions/*unrelaxed.cif" ) ) != 0:
			print( "\nInitial structure for the system exists..." )
			self.init_struct_cif = glob.glob( f"{self.ofold_output_dir}/predictions/*unrelaxed.cif" )[0]
		
		else:
			print( "\nPredicting the initial structure for the system..." )
			if not os.path.exists( os.path.abspath( f"{self.ofold_output_dir}/predictions/*unrelaxed.cif" ) ):
				self.get_initial_structure()
			# OpenFold predicted initial structure in CIF format.
			self.init_struct_cif = glob.glob( f"{self.ofold_output_dir}/predictions/*unrelaxed.cif" )[0]

		print( "\nCreating ground truth features from the initial structure..." )
		data = self.get_feature_from_init_struct()

		print( data.keys() )
		for k in data["gt_features"].keys():
			print( k, ": --> ", data["gt_features"][k].shape )
		return data


	def get_initial_structure( self ):
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
			print( f"using precomputed alignments for {mode} prediction..." )
		else:
			self.alignment_dir = None
		
		obj = OpenfoldCommand( script = self.script, 
								fasta_dir = self.fasta_dir, 
								config_preset = self.config_preset, 
								alignment_dir = self.alignment_dir, 
								output_dir = self.ofold_output_dir,
								mode = self.mode,
								cpu_cores = self.cpu_cores, 
								device = self.device )
		
		print( f"Running {mode} prediction..." )
		command = obj.get()
		run_subprocess( command )
		# if self.mode == 'mono':
		# 	self.run_monomer_prediction()
		# elif self.mode == 'multi':
		# 	self.run_multimer_prediction()


	def run_monomer_prediction( self ):
		"""
		Use OpenFold for monomer structure prediction.
			Use precomputed alignments if available.

		Input:
		----------
		training --> (bool) specifies if OpenFold is to be used for training/prediction.

		Returns:
		----------
		None
		"""

		# if training:
		# 	script = "unfrozen_pretrained.py"
		# else:
		# 	script = "run_pretrained_openfold.py"

		# Move to OpenFold dir.
		os.chdir( self.openfold_dir )
		
		if os.path.exists( self.alignment_dir ):
			print( "using precomputed alignments for monomer prediction..." )
			command = monomer_precomp_aln_cmd( self.script, self.fasta_dir, 
											self.alignment_dir, self.ofold_output_dir,
											self.config_preset, self.ckpt_path,
											self.cpu_cores, self.device )

		else:
			command = monomer_cmd( self.script, self.fasta_dir, self.ofold_output_dir,
									self.config_preset,  self.ckpt_path,
									self.cpu_cores, self.device )

		print( "Running monomer prediction..." )
		run_subprocess( command )



	def run_multimer_prediction( self ):
		"""
		Use OpenFold for multimer structure prediction.
			Use precomputed alignments if available.

		Input:
		----------
		training --> (bool) specifies if OpenFold is to be used for training/prediction.

		Returns:
		----------
		None
		"""
		# Move to OpenFold dir.
		os.chdir( self.openfold_dir )

		if os.path.exists( self.alignment_dir ):
			print( "using precomputed alignments for multimer prediction..." )
			command = multimer_precomp_aln_cmd( self.script, self.fasta_dir, 
												self.alignment_dir, self.ofold_output_dir,
												self.config_preset, self.cpu_cores, self.device )

		else:
			command = multimer_cmd( self.script, self.fasta_dir, self.ofold_output_dir,
									self.config_preset, self.cpu_cores, self.device )

		print( "Running multimer prediction..." )
		run_subprocess( command )


	def get_feature_from_init_struct( self ):
		# Parse the .cif file to get an mmcif_object.
		# This mmcif_object is not the same as Biopython structure object.
		with open( self.init_struct_cif, "r" ) as f:
		    mmcif_string = f.read()

		mmcif_object = parse( 
		    file_id = "2ayo", mmcif_string = mmcif_string
		).mmcif_object

		print( os.path.exists( self.init_struct_cif ) )
		# Extract relevant relevant features from the mmcif_object.
		np_example = process_mmcif( mmcif_object )

		print( "Load config file..." )
		config = model_config( "model_1_multimer_v3" )

		# Obtain the ground truth features.
		data = np_example_to_features(
				np_example = np_example,
				config = config["data"],
				mode = "train",
				is_multimer = True )

		return data


if __name__ == "__main__":
	SystemRepresentation().forward()

