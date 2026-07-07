"""
Create fake multi-state binary complexes from a given experimental PDB.
"""
from typing import List, Tuple, Dict, Any
import os, warnings
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch
from Bio.PDB import PDBIO, MMCIFIO, Atom, NeighborSearch

from data.simulate_data import SimulateCrosslinks
from config import get_config_dict
from utils.pdb_utils import Parser
from utils.paths import (
	get_meta_dir_path,
	get_sys_data_dir_path
	)
from utils.mappings import map_xls_to_seq_id

warnings.filterwarnings( "ignore" )


class MultiStateGenerator():
	"""
	Module to simulate alternate rigid transformed states
		for binary complexes.
	"""
	def __init__( self ):
		self.config_dict = get_config_dict()
		self.base_dir = os.path.join(
			os.path.abspath( self.config_dict.models.base_dir )
			)
		self.benchmark_name = self.config_dict.benchmark.globals.benchmark_name

		self.clash_cutoff = 8.0
		self.interface_dist = 10.0
		self.overlap_tolerace = 0.2
		self.cpu_cores = 10

		self.rotation_angles = np.arange( 1, 360, 0.5 )
		self.max_translation = 2
		self.num_samples = 1000

		# Complexes for which to simulate multiple states.
		# Selected based on data satisfaction and size.
		self.systems = ["1sc1", "8g0p", "8sjj"]
		# Keep tabs on the selected systems.
		self.sys_selected = []


	def forward( self ):
		"""
		Extract all-atom coordinates from input PDB file.
		Run sanity checks for binary complex.
		Sample rigid transformations to create alternate poses.
		Validate the simulates poses.
		Simulate XLs for the selected pose.
		"""
		self.run_multi_state_pipeline()

	################################################################################
	################################################################################
	def run_multi_state_pipeline( self ):
		"""
		Run the multistate creation pipeline for all systems.
		"""
		for sys_name in self.systems:
			print( f"\n{sys_name}\n" + "-"*80 )
			self.run_multi_state_pipeline_per_sys( sys_name = sys_name )


	def run_multi_state_pipeline_per_sys( self, sys_name: str ):
		"""
		For the given system, run the multi-state pipeline, comprising:
			Parse coordinates from experimental struct.
			Simulate altrenate docked poses.
			Validate the simulate poses.
			Craete the two states.
			Simulate Xls for the two states.
			Save Xls on disk.
		"""
		sys_data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name
		)
		struct_file = os.path.join( sys_data_dir, f"{sys_name}.pdb" )
		coords_dict, ca_mask = self.pdb_to_coords( pdb_file = struct_file )
		self.check_binary( coords_dict = coords_dict )

		com_dict = self.get_interface_com(
			coords_dict = coords_dict,
			ca_mask = ca_mask,
			interface_com = True
			)
		pose_dict = self.create_fake_multi_state(
			coords_dict = coords_dict,
			com_dict = com_dict
		)
		pose_dict,ignore = self.validate_poses(
			pose_dict = pose_dict,
			ca_mask = ca_mask
		)
		if ignore:
			pass
		else:
			two_states = self.get_two_states(
				sys_name = sys_name,
				pose_dict = pose_dict
			)
			two_state_file = self.save_multi_state(
				orig_pdb_file = struct_file,
				two_states = two_states
			)
			ignore = self.get_xls_for_multistate(
				sys_name = sys_name,
				two_state_file = two_state_file
			)
			if not ignore:
				self.sys_selected.append( sys_name )
		print( f"Selected systems: {len( self.sys_selected )}" )

	################################################################################
	def check_binary( self, coords_dict: Dict[str, List] ):
		"""
		Raise an error if the >2 chains are present.
		"""
		if len( coords_dict ) > 2:
			raise ValueError( "Non-binary complexes not-supported..." )

	################################################################################
	################################################################################
	def pdb_to_coords(
		self,
		pdb_file: str
		) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
		"""
		Extract all all-atom coordinates from the pdb file.

		Inputs:
		----------
		pdb_file: file path for the structure file to overwritten.

		Returns:
		----------
		coords_dict: a dict containing all-atom coordinates for each chain.
		ca_mask: a dict contaiing binary mask for Ca-atoms for all chains.
		"""
		p = Parser( pdb_file = pdb_file )
		coords_dict = {}
		ca_mask = {}
		models = [m for m in p.get_models()]
		if len( models ) > 1:
			print( f">1 models detected in the PDB. Selecting the 1st one..." )
		for chain in models[0]:
			chain_id = chain.id
			for residue in chain:
				if chain_id not in coords_dict:
					coords_dict[chain_id] = []
					ca_mask[chain_id] = []
					if residue.id[0] != " ":
						continue
				for atom in residue:
					coords_dict[chain_id].append( atom.get_coord() )
					ca_mask[chain_id].append( atom.get_name() == "CA" )

		for chain_id in coords_dict:
			coords_dict[chain_id] = np.asarray(
				coords_dict[chain_id],
				dtype = np.float32
			).reshape( -1, 3 )
			ca_mask[chain_id] = np.asarray(
				ca_mask[chain_id],
				dtype = bool
			).reshape( -1 )

		return coords_dict, ca_mask


	def save_coords_to_pdb(
		self,
		pdb_file: str,
		coords_dict: Dict[str, List],
		new_pdb_file: str
	):
		"""
		Overwrite the coordinates in the input pdb-file with
			the provided coordinates and save on disk.
		Here we assume that the updated coordinates belong to
			the given pdb_file.
		So, we just take the structure from the input file
			and update the coordinates.

		Inputs:
		----------
		pdb_file: file path for the structure file to overwritten.
		coords_dict: a dict containing all-atom coordinates for each chain.
		new_pdb_file: file path for the structure file with the updated coordinates.
		"""
		base, ext = os.path.splitext( pdb_file )
		p = Parser( pdb_file = pdb_file )
		structure = p.structure
		models = [m for m in p.get_models()]
		if len( models ) > 1:
			print( f">1 models detected in the PDB. Selecting the 1st one..." )
		for chain in models[0]:
			chain_id = chain.id
			i = 0
			for residue in chain:
				for atom in residue:
					atom.set_coord( coords_dict[chain_id][i] )
					i+=1
		
		if "pdb" in ext:
			io = PDBIO()
		elif "cif" in ext:
			io = MMCIFIO
		else:
			raise ValueError( f"Unsupported file format: {ext}..." )
		
		io.set_structure( structure )
		io.save( new_pdb_file )

	################################################################################
	################################################################################
	def get_interface_com(
		self,
		coords_dict: Dict[str, np.ndarray],
		ca_mask: Dict[str, np.ndarray],
		interface_com: bool
		):
		"""
		Obtain the interface COM for all chains.

		Inputs:
		----------
		coords_dict: a dict containing all-atom coordinates for each chain.
		ca_mask: a dict contaiing binary mask for Ca-atoms for all chains.
		interface_com: if True, compute COM based on the Ca-coords of only
			interface residues else all Ca residues.

		Returns:
		----------
		com_dict: a dict containing interface com for all chains.
		"""
		chain1, chain2 = list( coords_dict.keys() )
		coords1 = coords_dict[chain1]
		ca_mask1 = ca_mask[chain1]
		ca_coords1 = coords1[ca_mask1].reshape( -1, 3 )
		coords2 = coords_dict[chain2]
		ca_mask2 = ca_mask[chain2]
		ca_coords2 = coords2[ca_mask2].reshape( -1, 3 )

		dist_mat = np.linalg.norm(
			ca_coords1[:, None, :] - ca_coords2[None, :, :],
			axis = -1
		)
		if interface_com:
			interface = np.where( dist_mat <= self.interface_dist )
			ires1, ires2 = interface
			if len( ires1 ) == 0 or len( ires2 ) == 0:
				raise ValueError(
					f"No interface residues found at cutoff = {self.interface_dist} angstrom"
				)
			com_dict = {
				chain1: ca_coords1[ires1].mean( axis = 0 ),
				chain2: ca_coords2[ires2].mean( axis = 0 )
			}
		else:
			com_dict = {
				chain1: ca_coords1.mean( axis = 0 ),
				chain2: ca_coords2.mean( axis = 0 )
			}
		return com_dict

	################################################################################
	def get_rotation_matrix( self, axis: str, angle: float ) -> np.ndarray:
		"""
		Given the axis and angle of rotation, compute a rotation matrix using scipy.

		Inputs:
		----------
		axis: axis of rotation.
		angle: angle of rotation.

		Returns:
		----------
		R : [3, 3] rotation matrix.
		"""
		axis = axis.lower()
		if axis not in ["x", "y", "z"]:
			raise ValueError( f"Invalid axis '{axis}'. Must be one of x,y,z." )

		R = Rotation.from_euler( axis, angle, degrees = True ).as_matrix()
		return R


	def rotate(
		self,
		coords: np.ndarray,
		com: np.ndarray,
		R: np.ndarray
	):
		"""
		Apply the rotation matrix to the specified coordinates
			for rottaion along the given com.

		Inputs:
		----------
		coords: [N, 3] coordinates.
		com: [1, 3] COM coordinates; acts as
			the centre of rotation.
		R: [3, 3] rotation matrix.

		Returns:
		----------
		rotated_coords: [N, 3] rotated coordinates.
		"""
		coords_centered = coords # - com
		rotated_coords = coords_centered@R.T
		# rotated_coords += com

		return rotated_coords


	def create_fake_poses(
		self,
		moving_coords: np.ndarray,
		com: np.ndarray
	) -> Dict[str, Dict[int, np.ndarray]]:
		"""
		Create fake poses by apply rigid transformation
			over the moving chain.
		Given the input coordinates, perform a
			grid search across rotational axis
			and angle to obtain rotated coordinates.
		Add a randomly sampled translation vector.

		Inputs:
		----------
		moving_coords: [N, 3] coordinates of the chain to
			be moved.
		com: [1, 3] COM coordinates.

		Returns:
		----------
		rotated_coords: a dict containing rigid transformed
			poses for the given chain coordinates.
		"""
		np.random.seed( self.config_dict.prng_seed )

		rotated_coords = {}
		rotation_angles = np.arange( 1, 360, 1 )

		for axis in ["x", "y", "z"]:
			# for i in range( self.num_samples ):
			for angle in self.rotation_angles:
				# Sample a random translation
				# angle = np.random.choice( self.rotation_angles, 1 )[0]
				key = f"{axis}_{angle}"
				trans = np.random.uniform( 0, self.max_translation, 3 ).reshape( -1, 3 )
				# key = f"{i}_{axis}_{angle}"
				R = self.get_rotation_matrix(
					axis = axis, angle = angle
				)
				rot_coords = self.rotate(
					coords = moving_coords,
					com = com,
					R = R
				)
				rotated_coords[key] = rot_coords + trans
		print( "Total rotated pose created: ", len( rotated_coords ) )
		return rotated_coords


	# def create_fake_poses(
	# 	self,
	# 	moving_coords: np.ndarray,
	# 	com: np.ndarray
	# ) -> Dict[str, Dict[int, np.ndarray]]:
	# 	"""
	# 	Create fake poses by apply rigid transformation
	# 		over the moving chain.
	# 	Given the input coordinates, perform a
	# 		grid search across rotational axis
	# 		and angle to obtain rotated coordinates.
	# 	Add a randomly sampled translation vector.

	# 	Inputs:
	# 	----------
	# 	moving_coords: [N, 3] coordinates of the chain to
	# 		be moved.
	# 	com: [1, 3] COM coordinates.

	# 	Returns:
	# 	----------
	# 	rotated_coords: a dict containing rigid transformed
	# 		poses for the given chain coordinates.
	# 	"""
	# 	np.random.seed( self.config_dict.prng_seed )

	# 	rotated_coords = {}
	# 	rotation_angles = np.arange( 1, 360, 1 )

	# 	for axis in ["x", "y", "z"]:
	# 		for angle in rotation_angles:
	# 			# Sample a random translation
	# 			trans = np.random.uniform( 0, 5, 3 ).reshape( -1, 3 )
	# 			key = f"{axis}_{angle}"
	# 			R = self.get_rotation_matrix(
	# 				axis = axis, angle = angle
	# 			)
	# 			rot_coords = self.rotate(
	# 				coords = moving_coords,
	# 				com = com,
	# 				R = R
	# 			)
	# 			rotated_coords[key] = rot_coords + trans
	# 	print( "Total rotated pose created: ", len( rotated_coords ) )
	# 	return rotated_coords


	def create_fake_multi_state(
		self,
		coords_dict: Dict[str, np.ndarray],
		com_dict: Dict[str, np.ndarray]
	) -> Dict[str, Dict]:
		"""
		Keeping the longer chain fixed, obtain fake
			poses by rotating the smaller chain.

		Inputs:
		----------
		coords_dict: dict containing coordinates for each chain.
		com_dict: dict containing the COM coordinates for each chain.

		Returns:
		----------
		pose_dict: dict containing the coordinates for the native
			chains and the rigid transformed poses.
			{
				fixed: {
				# chain and coordinates for the fixed chain in native struct
					chain_id: ,
					coords: 
				},
				moving: {
				# chain and coordinates for the moving chain in native struct
					chain_id: ,
					coords: 
				},
				pose: {
					chain_id: ,
					coords: {
						{axis}_{angle}: coordinates after rigid transform
					}
				}
			}
		"""
		chain1, chain2 = list( coords_dict.keys() )

		if coords_dict[chain1].shape[0] >= coords_dict[chain2].shape[0]:
			fixed_chain = chain1
			moving_chain = chain2
		else:
			fixed_chain = chain2
			moving_chain = chain1
		fixed_coords = coords_dict[fixed_chain]
		moving_coords = coords_dict[moving_chain]

		rotated_coords = self.create_fake_poses(
			moving_coords = moving_coords,
			com = com_dict[moving_chain]
		)

		pose_dict = {
			# native chain kept fixed.
			"fixed": {
				"chain_id": fixed_chain,
				"coords": fixed_coords
			},
			# native chain to be moved.
			"moving": {
				"chain_id": moving_chain,
				"coords": moving_coords
			},
			"pose": {
				"chain_id": moving_chain,
				"coords": rotated_coords
			}
		}
		return pose_dict

	################################################################################
	################################################################################
	def compute_interchain_clashes(
		self,
		coords1: np.ndarray,
		coords2: np.ndarray,
		ca_mask1: np.ndarray,
		ca_mask2: np.ndarray,
		eps: float = 0.2
	) -> int:
		"""
		Given a pair of coordinates, compute the distance
			map based on the ca-coordinates.
		Compute no. of clashes using a 4 angstrom distance cutoff.
		"""
		ca_coords1 = coords1[ca_mask1]
		ca_coords2 = coords2[ca_mask2]

		dist_map = np.linalg.norm(
			ca_coords1[:, None, :] - ca_coords2[None, :, :],
			axis = -1
		)
		num_clashes = int( np.sum( dist_map <= self.clash_cutoff+eps ) )
		return num_clashes


	def exclude_with_clashes(
		self,
		pose_dict: Dict[str, Dict[str, Any]],
		ca_mask: Dict[str, np.ndarray],
	) -> Dict[str, Dict[str, Any]]:
		"""
		Compute the no. of inter-chain clashes in the native structure.
			Inter-chain clashes considered at a 8 abgstrom cutoff.
		Exclude all rotated poses which have higher inter-chain clashes
			than the native.

		Inputs:
		----------
		pose_dict: dict containing the coordinates for the native
			chains and the rigid transformed poses.
		ca_mask: bool mask for Ca-atoms in each chain.

		Returns:
		----------
		pose_dict: dict containing the poses after removing clashes.
		"""
		print( "\nRemoving poses with interchain clashes..." )
		fixed_coords = pose_dict["fixed"]["coords"]
		fixed_chain = pose_dict["fixed"]["chain_id"]
		moving_coords = pose_dict["moving"]["coords"]
		moving_chain = pose_dict["moving"]["chain_id"]

		fixed_ca_mask = ca_mask[fixed_chain]
		moving_ca_mask = ca_mask[moving_chain]

		native_clashes = self.compute_interchain_clashes(
			coords1 = fixed_coords,
			coords2 = moving_coords,
			ca_mask1 = fixed_ca_mask,
			ca_mask2 = moving_ca_mask,
		)

		to_remove = []
		for k in pose_dict["pose"]["coords"]:
			pose_coords = pose_dict["pose"]["coords"][k]
			pose_clashes = self.compute_interchain_clashes(
				coords1 = fixed_coords,
				coords2 = pose_coords,
				ca_mask1 = fixed_ca_mask,
				ca_mask2 = moving_ca_mask,
			)
			# print( native_clashes, "  ", pose_clashes )
			if pose_clashes > native_clashes:
				to_remove.append( k )
		for k in to_remove:
			_ = pose_dict["pose"]["coords"].pop( k )
		print( f"Poses remove: {len( to_remove )}" )
		print( f"Remaining poses: {len( pose_dict['pose']['coords'] )}" )
		# print( pose_dict["pose"]["coords"].keys() )
		return pose_dict

	################################################################################
	def coords_to_contact(
		self,
		coords1: np.ndarray,
		coords2: np.ndarray,
		ca_mask1: np.ndarray,
		ca_mask2: np.ndarray,
	) -> np.ndarray:
		"""
		Given the coordinates, compute and return the contact map.

		Inputs:
		----------
		coords1: [N,3] coordinates for chain1.
		coords2: [N,3] coordinates for chain2.
		ca_mask1: [N,3] bool Ca-maks for chain1.
		ca_mask2: [N,3] bool Ca-maks for chain2.

		returns:
		----------
		contact_map: [N,N] binary contact map created at the
			specified cutoff.
		"""
		ca_coords1 = coords1[ca_mask1]
		ca_coords2 = coords2[ca_mask2]
		dist_map = np.linalg.norm(
			ca_coords1[:, None, :] - ca_coords2[None, :, :],
			axis = -1
		)
		contact_map = np.where( dist_map <= self.interface_dist, 1, 0 )
		return contact_map


	def exclude_overlapping_interfaces(
		self,
		pose_dict: Dict[str, Dict[str, Any]],
		ca_mask: Dict[str, np.ndarray],
	) -> Dict[str, Dict[str, Any]]:
		"""
		Compute the overlap in the interface residues
			bewteen the native and predicted poses.
		If the fraction of overlapping contact is greater
			than the overlap tolerance, remove th pose.

		Inputs:
		----------
		pose_dict: dict containing the coordinates for the native
			chains and the rigid transformed poses.
		ca_mask: bool mask for Ca-atoms in each chain.

		Returns:
		----------
		pose_dict: dict containing the poses after removing
			similar interfaces to native.
		"""
		print( "\nRemoving poses with overlapping interface..." )
		fixed_coords = pose_dict["fixed"]["coords"]
		fixed_chain = pose_dict["fixed"]["chain_id"]
		moving_coords = pose_dict["moving"]["coords"]
		moving_chain = pose_dict["moving"]["chain_id"]

		fixed_ca_mask = ca_mask[fixed_chain]
		moving_ca_mask = ca_mask[moving_chain]
		C_native = self.coords_to_contact(
			coords1 = fixed_coords,
			coords2 = moving_coords,
			ca_mask1 = fixed_ca_mask,
			ca_mask2 = moving_ca_mask,
		)
		total_native = np.count_nonzero( C_native )
		to_remove = []
		interface_overlap = {}
		for k in pose_dict["pose"]["coords"]:
			pose_coords = pose_dict["pose"]["coords"][k]
			C_pose = self.coords_to_contact(
			coords1 = fixed_coords,
			coords2 = pose_coords,
			ca_mask1 = fixed_ca_mask,
			ca_mask2 = moving_ca_mask,
			)
			overlapping_contacts = C_native*C_pose
			frac_overlap = np.count_nonzero( overlapping_contacts )/total_native
		
			if frac_overlap >= self.overlap_tolerace:
				to_remove.append( k )
			else:
				interface_overlap[k] = frac_overlap
				# print( frac_overlap )
		for k in to_remove:
			_ = pose_dict["pose"]["coords"].pop( k )
		print( f"Poses remove: {len( to_remove )}" )
		print( f"Remaining poses: {len( pose_dict['pose']['coords'] )}" )
		# print( pose_dict["pose"]["coords"].keys() )
		return pose_dict, interface_overlap

	################################################################################
	def validate_poses(
		self,
		pose_dict: Dict[str, Dict[str, Any]],
		ca_mask: Dict[str, np.ndarray],
	) -> Tuple[Dict[str, Dict[str, Any]], bool]:
		"""
		Run validation for the smaple poses.
			Remove pose with clashes.
			Remove poses with high interface similarity to the native.

		Inputs:
		----------
		pose_dict: dict containing the coordinates for the native
			chains and the rigid transformed poses.
		ca_mask: bool mask for Ca-atoms in each chain.

		Returns:
		----------
		pose_dict: dict containing the poses after validation.
		ignore: if true, no pose survived the validation steps.
		"""
		pose_dict = self.exclude_with_clashes(
			pose_dict = pose_dict,
			ca_mask = ca_mask,
		)
		pose_dict, interface_overlap = self.exclude_overlapping_interfaces(
			pose_dict = pose_dict,
			ca_mask = ca_mask,
		)

		# select the pose with the least interface overlap with the native structure.
		min_v = self.overlap_tolerace
		min_k = ""
		for k, v in interface_overlap.items():
			if v < min_v:
				min_v = v
				min_k = k

		if min_k == "":
			ignore = True
			print( f"\033[1mNo alternate state found...\033[0m" )
		else:
			ignore = False

		print( f"Selected pose: {min_k} \t {min_v}" )
		return pose_dict, ignore

	################################################################################
	################################################################################
	def get_two_states(
		self,
		sys_name: str,
		pose_dict: Dict[str, Dict[str, Any]]
	) -> Dict[str, Dict]:
		"""
		Given the selected pose, get coordinates for the two states:
			native state
			rigid transformed state

		Inputs:
		----------
		pose_dict: dict containing the coordinates of the native
			struct and selected rigid transformed pose.

		returns:
		----------
		two_states: dict containing chain_ids and coordinates for the two states.
		"""
		chain_id1 = pose_dict["fixed"]["chain_id"]
		chain_id2 = pose_dict["moving"]["chain_id"]
		# Select the 1st pose when multiple are present.
		pose_coords = list( pose_dict["pose"]["coords"].values() )[0]

		two_states = {
			f"{sys_name}_S1": {
				f"{chain_id1}": pose_dict["fixed"]["coords"],
				f"{chain_id2}": pose_dict["moving"]["coords"]
			},
			f"{sys_name}_S2": {
				f"{chain_id1}": pose_dict["fixed"]["coords"],
				f"{chain_id2}": pose_coords
			}
		}
		return two_states


	def save_multi_state(
		self,
		two_states: Dict[str, Dict],
		orig_pdb_file: str
	) -> Dict[str, str]:
		"""
		Save the multiple states on disk as a .pdb file.
		Save the structures in the data dir for the system.

		Inputs:
		----------
		two_states: dict containing chain_ids and coordinates for
			the two states.
	
		Returns:
		----------
		two_state_file: dict containing the file path for the two states.
		"""
		base, ext = os.path.splitext( orig_pdb_file )
		state_num = 1
		two_state_file = {}
		for state in two_states:
			state_pdb_file = f"{base}_S{state_num}{ext}"
			two_state_file[state] = state_pdb_file
			self.save_coords_to_pdb(
				pdb_file = orig_pdb_file,
				coords_dict = two_states[state],
				new_pdb_file = state_pdb_file
			)
			state_num += 1
		return two_state_file

	################################################################################
	################################################################################
	def get_xls_for_multistate(
		self,
		sys_name: str,
		two_state_file: Dict[str, str]
	) -> bool:
		"""
		Simulate XLs using JWalk for multiple states.
		Randomly select a subset fo 8 XLs. No FP XL included.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		two_state_file: dict containing the file path for the two states.

		Returns:
		----------
		ignore: if True, indicates that either of the states had
			no XLs post removal of overlapping XLs.
		"""
		print( f"\nSimulating XLs using JWalk..." )
		meta_dir = get_meta_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name
		)
		sys_data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name
		)
		jwalk_dir = os.path.join( meta_dir, "jwalk_multistate" )
		os.makedirs( jwalk_dir, exist_ok = True )

		xls_dict = self.xl_simulator(
			states = two_state_file,
			jwalk_dir = jwalk_dir,
			pdb_struct_dir = sys_data_dir
		)
		xls_s1, xls_s2 = self.select_xls_for_multistate(
			xls_dict = xls_dict,
			num_xls = 8
		)
		if xls_s1.shape == 0 or xls_s2.shape[0] == 0:
			ignore = True
		else:
			ignore = False
			xls_s1, xls_s2 = self.remap_xl_res_numbering(
				sys_name = sys_name,
				meta_dir = meta_dir,
				xls_s1 = xls_s1,
				xls_s2 = xls_s2
			)
			xls_s1, xls_s2 = self.map_chains_to_prot(
				sys_name = sys_name,
				xls_s1 = xls_s1,
				xls_s2 = xls_s2
			)
			# Xls from both states together.
			xls_s1_2 = pd.concat(
				[xls_s1, xls_s2], axis = 0
			)
			xls_s1_2 = xls_s1_2.reset_index( drop = True )
			self.save_ultistate_xls(
				xls_s1_df = xls_s1,
				xls_s2_df = xls_s2,
				xls_s1_2_df = xls_s1_2,
				data_dir = sys_data_dir
			)
		return ignore

	################################################################################
	def xl_simulator(
		self,
		states: List[str],
		jwalk_dir: str,
		pdb_struct_dir: str
	) -> Dict[str, pd.DataFrame]:
		"""
		Simulate Xls using JWalk.
		We simulate XLs involving the following aa: K,R,D,E,N,Q,S,T,Y
			We set the num_inter_xl to 1 so as to select all XLs per aa.
		No FP XL included.
		Merge all TP XLs obtained from all aa.

		Inputs:
		----------
		states: dict containing the file path for the two states.
		jwalk_dir: dir path to store the JWalk output.
		pdb_struct_dir: path to the dir containing the PDB file for the system.

		Returns:
		----------
		xls_dict: dict containing the JWalk simulated XLs as a pd.DataFrame
			for each state.
			{
				entry_id: pd.dataFrame -> prot1,res1,prot2,res2
			}
			entry_id: {sys_name}_S{state_num}
		"""
		xls_dict = {}
		entry_ids = list( states.keys() )
		print( entry_ids )

		for aa in ["LYS", "ARG", "ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR"]:
			print( f"Computing XLs for aa: {aa}" )
			sim_obj = SimulateCrosslinks(
				jwalk_exec = self.config_dict.benchmark.jwalk.jwalk_exec,
				pdb_ids_list = entry_ids,
				jwalk_dir = jwalk_dir,
				pdb_struct_dir = pdb_struct_dir,
				struct_format = "pdb",
				short_linker = self.config_dict.benchmark.jwalk.short_linker,
				long_linker = self.config_dict.benchmark.jwalk.long_linker,
				num_inter_xls = 1,
				cores = self.cpu_cores,
				aa1 = aa,
				aa2 = aa
			)
			sim_obj.reinit_logs = True
			sim_obj.forward()
			xls_dict_aa = sim_obj.xls_dict

			# Merge XLs from all the specified XLs.
			for entry_id in xls_dict_aa:
				if entry_id in xls_dict:
					xls_dict[entry_id] = pd.concat(
						[
							xls_dict[entry_id],
							xls_dict_aa[entry_id]["short_xls"]
						],
						axis = 0
					)
				else:
					xls_dict[entry_id] = xls_dict_aa[entry_id]["short_xls"]
			xls_dict[entry_id] = xls_dict[entry_id].reset_index( drop = True )
		print( f"Total XLs obtained for {entry_id}: {xls_dict[entry_id].shape}" )
		return xls_dict

	################################################################################
	def sample_xls(
		self,
		xls_df: pd.DataFrame,
		num_xls: int
	) -> pd.DataFrame:
		"""
		Randomly select a subset of num_xls XLs.

		Inputs:
		----------
		xls_df: pd.DataFrame containing XLs in the format,
			prot1,res1,prot2,res2

		Returns:
		----------
		xls: pd.DataFrame containing subsampled XLs.
		"""
		np.random.seed( self.config_dict.prng_seed )
		if xls_df.shape[0] <= num_xls:
			xls = xls_df
		else:
			indexes = list( xls_df.index )
			# Sample a subset of XLs without replacement.
			
			sampled_idx = np.random.choice(
				a = indexes,
				size = num_xls,
				replace = False
			)
			xls = xls_df.iloc[sampled_idx]
		return xls


	def select_xls_for_multistate(
		self,
		xls_dict: Dict[str, pd.DataFrame],
		num_xls: int
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Select Xls for the two states.
			Segregte the XLs into three categories:
				State1 XLs only
				State2 XLs only
				Common XLs
			Randomly select a subset fo 8 XLs for each state.
			The two states must have no overlapping XL.
				Ignore the common XLs.

		Inputs:
		----------
		xls_dict: dict containing the JWalk simulated XLs as a pd.DataFrame
			for each state.
		num_xls: no. of XLs to be selected.

		Returns:
		----------
		xls_s1: pd.DataFrame containing XLs for state1.
		xls_s2: pd.DataFrame containing XLs for state2.
		"""
		state1, state2 = list( xls_dict.keys() )
		xls_s1_all = xls_dict[state1]
		xls_s2_all = xls_dict[state2]

		xls_s1 = self.sample_xls(
			xls_df = xls_s1_all,
			num_xls = num_xls
		)

		# Find common XLs based on "res1", "res2".
		# This adds an extra column "_merge" on the right-end.
		common_pairs = pd.merge(
			xls_s1_all[["res1", "res2"]],
			xls_s2_all[["res1", "res2"]],
			on = ["res1", "res2"]
		).drop_duplicates()

		# Remove common XLs from state1 and state2 Xls.
		# Also remove the _merge column.
		xls_s1 = (
			xls_s1_all.merge(
				common_pairs,
				on = ["res1", "res2"],
				how = "left",
				indicator = True
			).query( '_merge == "left_only"' ).drop( columns = "_merge" )
		)
		xls_s1 = xls_s1.reset_index( drop = True )
		xls_s2 = (
			xls_s2_all.merge(
				common_pairs,
				on = ["res1", "res2"],
				how = "left",
				indicator = True
			).query( '_merge == "left_only"' ).drop( columns = "_merge" )
		)
		xls_s2 = xls_s2.reset_index( drop = True )

		# Sample a subset of XLs.
		xls_s1 = self.sample_xls(
			xls_df = xls_s1,
			num_xls = self.config_dict.benchmark.jwalk.num_inter_xls
		)
		xls_s2 = self.sample_xls(
			xls_df = xls_s2,
			num_xls = self.config_dict.benchmark.jwalk.num_inter_xls
		)
		xls_s1 = xls_s1.reset_index( drop = True )
		xls_s2 = xls_s2.reset_index( drop = True )

		# Add a "label" column; indicating TP/FP.
		labels = [1]*xls_s1.shape[0]
		xls_s1["label"] = labels
		labels = [1]*xls_s2.shape[0]
		xls_s2["label"] = labels

		print( f"State1 XLs: {xls_s1.shape[0]} \t State2 XLs: {xls_s2.shape[0]}" )
		return xls_s1, xls_s2

	################################################################################
	def remap_xl_res_numbering(
		self,
		sys_name: str,
		meta_dir: str,
		xls_s1: pd.DataFrame,
		xls_s2: pd.DataFrame
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Map the PDb residue numbering for XLs to the seq_id as in the MMCIF file.
		For the crosslink benchmark, the mapping for all systems has been saved
			on disk previously, just reuse it.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		meta_dir_path: path to the metadata dir.
		xls_s1: pd.DataFrame containing XLs for state1.
		xls_s2: pd.DataFrame containing XLs for state2.

		Returns:
		----------
		xls_s1: pd.DataFrame containing XLs for state1 with residue
			numbering as per the seq_id.
		xls_s2: pd.DataFrame containing XLs for state2 with residue
			numbering as per the seq_id.
		"""
		mapping_file = os.path.join( meta_dir, "pdb_num_seq_id_map.npy" )
		pdb_num_seq_id_map = np.load( mapping_file, allow_pickle = True ).item()

		xls_s1 = map_xls_to_seq_id(
			xl_df = xls_s1,
			pdb_num_seq_id_map = pdb_num_seq_id_map[sys_name]
		)
		xls_s2 = map_xls_to_seq_id(
			xl_df = xls_s2,
			pdb_num_seq_id_map = pdb_num_seq_id_map[sys_name]
		)

		return xls_s1, xls_s2

	################################################################################
	def map_chains_to_prot(
		self,
		sys_name: str,
		xls_s1: pd.DataFrame,
		xls_s2: pd.DataFrame
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		The prot1/2 contains chain_ids labels need to be mapped to protein labels.
		e.g. A -> {sys_name}_1; B -> {sys_name}_2

		Inputs:
		----------
		xls_s1: pd.DataFrame containing XLs for state1.
		xls_s2: pd.DataFrame containing XLs for state2.

		Returns:
		----------
		xls_s1: pd.DataFrame containing XLs for state1.
		xls_s2: pd.DataFrame containing XLs for state2.
		"""
		for i in xls_s1.index:
			xls_s1.loc[i, "prot1"] = f"{sys_name}_1"
			xls_s1.loc[i, "prot2"] = f"{sys_name}_2"
		for i in xls_s2.index:
			xls_s2.loc[i, "prot1"] = f"{sys_name}_1"
			xls_s2.loc[i, "prot2"] = f"{sys_name}_2"
		return xls_s1, xls_s2

	################################################################################
	def save_ultistate_xls(
		self,
		data_dir: str,
		xls_s1_df: pd.DataFrame,
		xls_s2_df: pd.DataFrame,
		xls_s1_2_df: pd.DataFrame
	):
		"""
		Save the XLs for state1/2 on disk in the system data dir.

		Inputs:
		----------
		data_dir: path to the system specific data dir.
		xls_s1: pd.DataFrame containing XLs for state1.
		xls_s2: pd.DataFrame containing XLs for state2.
		"""
		state1_file = os.path.join( data_dir, "interprotein_xls_S1.csv" )
		state2_file = os.path.join( data_dir, "interprotein_xls_S2.csv" )
		state1_2_file = os.path.join( data_dir, "interprotein_xls_S1_2.csv" )
		xls_s1_df.to_csv( state1_file, index = False )
		xls_s2_df.to_csv( state2_file, index = False )
		xls_s1_2_df.to_csv( state1_2_file, index = False )

################################################################################
################################################################################
if __name__ == "__main__":
	MultiStateGenerator().forward()
	print( "May the Force be with you..." )