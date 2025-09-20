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
from typing import Tuple, Dict, Any, Optional
import os, glob, pickle as pkl
import numpy as np
import ml_collections as mlc

import torch

from openfold.config import model_config
from mod_openfold import parse, process_mmcif, np_example_to_features

from utils.commands import OpenfoldCommand
from utils.utils import run_subprocess, parse_nested_dict


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

		feature_dict = self.load_feature_dict()
		data["feature_dict"] = feature_dict
		output_dict = self.load_outputs_dict()
		data["output_dict"] = output_dict
		data["gt_features"]["distance_map"] = self.get_distance_map(
			final_atom_position = data["output_dict"]["final_atom_positions"]
			)
		mean, var = self.get_distance_distribution( distogram_logits = output_dict["distogram_logits"] )
		data["gt_features"]["distogram_mean"] = mean
		data["gt_features"]["distogram_var"] = var

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


	def get_feature_from_init_struct( self ) -> Dict[str, Any]:
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


	def load_feature_dict( self ) -> Dict[str, Any]:
		"""
		Load the feature_dict saved as a .pkl file in the system's director.
		"""
		feature_dict_path = glob.glob( f"{self.ofold_output_dir}/predictions/*feature_dict.pkl" )
		if len( feature_dict_path ) == 0:
			raise FileNotFoundError( f"Incorrect path -- {feature_dict_path}..." )

		with open( feature_dict_path[0], "rb" ) as f:
			feature_dict = pkl.load( f )
		# feature_dict = parse_nested_dict( feature_dict, "to_tensor" )

		return feature_dict	


	def load_outputs_dict( self ) -> Dict[str, Any]:
		"""
		Load the outputs dict obtained from the initial OpenFold run.
		Contains all the structure module output.
		"""
		output_dict_path = glob.glob( f"{self.ofold_output_dir}/predictions/*output_dict.pkl" )
		if len( output_dict_path ) == 0:
			raise FileNotFoundError( f"Incorrect path -- {output_dict_path}..." )

		with open( output_dict_path[0], "rb" ) as f:
			output_dict = pkl.load( f )
		output_dict = parse_nested_dict( output_dict, "to_tensor" )
		return output_dict


	def get_distance_map( self, final_atom_position: torch.Tensor ) -> torch.Tensor:
		"""
		Given the final_atom_position, create a Ca-ca distance map,
			to be used as pseudo ground truth.
		final_atom_position -> [N, 37, 3]
		"""
		ca_coords = final_atom_position[:, 1, :]
		# [N, N, 3]
		diff = ca_coords[:, None, :] - ca_coords[None, :, :]
		# [N, N, 3]
		squared_diff = torch.sum( diff**2, dim = -1 )
		gt_distance_map = torch.sqrt( squared_diff )
		return gt_distance_map


	def get_distogram_bins( self ):
		# These are taken from openFold.utils.loss.distogram_loss().
		min_bin = 2.3125
		max_bin = 21.6875
		no_bins = 64

		# Last bin is a catch all bin.
		boundaries = np.linspace(
			min_bin,
			max_bin,
			no_bins - 1 )
		# Add the last bin.
		bin_width = boundaries[1] - boundaries[0]
		last_bin = boundaries[-1] + bin_width
		boundaries = np.append( boundaries, last_bin )

		return torch.from_numpy( boundaries )


	def get_distance_distribution( self, distogram_logits: np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Given the predicted distogram, obtain the per-residue pair
			mean and variance.
		"""
		print( type( distogram_logits ) )
		distogram = torch.softmax( distogram_logits, dim = -1 )
			# torch.from_numpy( distogram_logits ), dim = -1
			# ).numpy()

		boundaries = self.get_distogram_bins()

		# E(x) = x*p(x)
		E_x = torch.sum( boundaries[None, None, :]*distogram, dim = -1 )

		E_x2 = torch.sum( distogram*boundaries[None, None, :]**2, dim = -1 )
		# var = E(x**2) - E_x**2
		var = E_x2 - E_x**2

		assert E_x.shape == var.shape, "Shape mismatch: mean and variance matrices..."

		return E_x, var

if __name__ == "__main__":
	SystemRepresentation().forward()


