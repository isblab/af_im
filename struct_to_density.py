"""
Contains wrappers for converting an ensemble of
	structures to a localization probability density map.
"""
from typing import Dict
import os, shutil
import numpy as np
from multiprocessing import Pool
import tqdm
from functools import partial
import mrcfile
import ml_collections as mlc

from Bio.PDB import Superimposer, PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure

from utils.utils import open_file_handler
from utils.pdb_utils import Parser
from utils.tools import eman2_pdb2density


class LocalizationDensity():
	"""
	Convert an ensemble of models to localization density map.
	A localization density map represents the probability of a voxel 
		being occupied by a residues in a set of superposed models.
	"""
	def __init__( self,
		analysis_dir: str,
		struct_format: str,
		model_ids: int,
		molpobity_dict: Dict,
		ld_config: mlc.ConfigDict ):
		self.analysis_dir = analysis_dir
		self.struct_format = struct_format
		self.model_ids = model_ids
		self.molpobity_dict = molpobity_dict

		self.cpu_cores = ld_config["cpu_cores"]
		self.apix = ld_config["apix"]
		self.padding = ld_config["padding"]
		self.target_prob = ld_config["target_prob"]
		self.clean_up = ld_config["clean_up"]


	def forward( self ):
		"""
		Given an ensemble of models:
			Superpose all models.
			Convert each model to density.
		Compute the localization probability density.
		Write a script to view the localization probability density in ChimeraX.
		"""
		self.create_required_files()
		os.makedirs( self.tmp_dir, exist_ok = True )

		print( f"Get a set of superposed models..." )
		self.get_superposed_models()
		print( "Create localization probability density map..." )
		# self.convert_xtruxt_to_density_map()
		lpd = self.create_localization_probability_density()
		level = level = self.get_contour_level( density_map = lpd )
		print( f"Contour level = {level}" )
		self.write_chimeraX_script( contour_level = level )

		if self.clean_up:
			shutil.rmtree( self.tmp_dir )


	def create_required_files( self ):
		"""
		Create the required file paths.
		"""
		self.ld_file = os.path.join( self.analysis_dir, "localization_density.mrc" )
		self.ld_chimerax_script = os.path.join( self.analysis_dir, "ld_chimerax.py" )
		self.tmp_dir = os.path.join( self.analysis_dir, f"ld_tmp" )


	################################################################################
	################################################################################
	def get_model_file( self, model_id: int ) -> str:
		"""
		Return the path to the relaxed model for the given model_id.
		"""
		return os.path.join(
				self.analysis_dir,
				"relaxed_models",
				f"model_{model_id}.{self.struct_format}" )


	def get_superposed_model_file( self, model_id: int ) -> str:
		"""
		Return the path to the superposed model file for the given model_id.
		"""
		superposed_struct_file = os.path.join(
			self.tmp_dir,
			# f"superposed_{self.sys_name}_{model_id}.pdb"
			f"superposed_{model_id}.pdb"
		)
		return superposed_struct_file


	def get_density_map_file( self, model_id: int ) -> str:
		"""
		Return the path to the density map for the given model_id.
		"""
		density_map_file = os.path.join(
			self.tmp_dir,
			# f"density_{self.sys_name}_{model_id}.mrc"
			f"density_{model_id}.mrc"
		)
		return density_map_file

	def parse_mrcfile( self, mrc_file: str ) -> str:
		"""
		Parse the input mrcfile and return the density map.
		"""
		with mrcfile.open( mrc_file, "r" ) as f:
			density_map = f.data
			header = f.header
		return density_map


	def save_to_mrc( self,
		density_map: np.ndarray,
		apix: float,
		origin: np.ndarray,
		ld_file:str ):
		"""
		Save the input density map into an mrc file.
		"""
		with mrcfile.new( ld_file, overwrite = True ) as mrc:
			mrc.set_data( density_map.astype( "float32" ) )
			mrc.voxel_size = ( apix, apix, apix )
			mrc.header.origin = origin
			mrc.update_header_from_data()
			mrc.flush()

	################################################################################
	################################################################################
	def get_atoms_from_model( self, structure: Structure, return_coords = False ):
		"""
		Parse the structure and return the Biopython Atom object.
		"""
		atoms = []
		for atom in structure.get_atoms():
			if return_coords:
				atoms.append( atom.get_coord() )
			else:
				atoms.append( atom )
		return atoms


	# def get_chains_from_atom( self, atoms: List[Atom] ) -> np.ndarray:
	# 	"""
	# 	Given a list of Biopython Atom objects, return the chain IDs (asym_ids).
	# 	"""
	# 	asym_ids = []
	# 	for atom in atoms:
	# 		residue = atom.get_parent()
	# 		chain = residue.get_parent()
	# 		asym_ids.append( chain.id )
	# 	return np.array( asym_ids )


	# def get_atom_coordinates( self, atoms: List[Atom] ) -> np.ndarray:
	# 	"""
	# 	Given a list of Biopython Atom objects, return the coordinates.
	# 	"""
	# 	coords = []
	# 	for atom in atoms:
	# 		coords.apend( atom.get_coord() )
	# 	return np.array( coords ).reshape( -1, 3 )

	################################################################################
	################################################################################
	def superpose_model( self, fixed_model_id: int, moving_model_id: int ):
		"""
		Superpose the moving_model onto the fixed_model and save the superposed moving model on disk.
		"""
		fixed_model_file = self.get_model_file( model_id = fixed_model_id )
		moving_model_file = self.get_model_file( model_id = moving_model_id )

		p = Parser( fixed_model_file )
		fixed_model = p.get_structure()
		fixed_atoms = self.get_atoms_from_model( structure = fixed_model )
		p = Parser( moving_model_file )
		moving_model = p.get_structure()
		moving_atoms = self.get_atoms_from_model( structure = moving_model )

		superpose = Superimposer()
		superpose.set_atoms( fixed = fixed_atoms, moving = moving_atoms )
		superpose.apply( moving_model.get_atoms() )

		io = PDBIO()
		io.set_structure( moving_model )
		superposed_model_file = self.get_superposed_model_file( model_id = moving_model_id )
		io.save( superposed_model_file )


	def get_superposed_models( self ):
		"""
		Obtain a set of superposed models.
		Will superpose all models onto the first model.
		"""
		fixed_model_id = self.model_ids[0]
		fixed_model_file = self.get_model_file( model_id = fixed_model_id )
		p = Parser( fixed_model_file )
		fixed_model = p.get_structure()
		fixed_atoms = self.get_atoms_from_model( structure = fixed_model )

		io = PDBIO()
		io.set_structure( fixed_model )
		superposed_model_file = self.get_superposed_model_file( model_id = fixed_model_id )
		io.save( superposed_model_file )

		partial_func = partial( self.superpose_model, fixed_model_id )
		with Pool( self.cpu_cores ) as p:
			for result in tqdm.tqdm(
				p.imap_unordered( partial_func, self.model_ids[1:] ),
				total = len( self.model_ids[1:] )
			):
				pass

	################################################################################
	################################################################################
	def get_bounding_box( self ):
		"""
		Obtain the bounding box for the density map.
		Compute the min and max XYZ coordinates across all models.
		"""
		max_coords, min_coords = [], []

		for model_id in self.model_ids:
			model_file = self.get_model_file( model_id = model_id )
			p = Parser( model_file )
			model = p.get_structure()
			atoms = self.get_atoms_from_model( structure = model, return_coords = True )

			max_coords.append( np.max( atoms, axis = 0 ) )
			min_coords.append( np.min( atoms, axis = 0 ) )
		
		max_coords = np.max( np.stack( max_coords ), axis = 0 )
		min_coords = np.min( np.stack( min_coords ), axis = 0 )

		# padding = int( 2.0*resolution/self.apix )

		# EMAN2 expects box size in voxel counts not Angstorm.
		# 	voxenl count = ( max - min )/apix
		box = ( ( max_coords - min_coords )/self.apix + 2*self.padding ).astype( int )
		origin = ( min_coords - self.padding ).astype( int )
		return box, tuple( origin )


	def normalize_density_map( self, density_map: np.ndarray ):
		"""
		Normalize the input unnormalized denisty map.
		Divide each voxel by the sum across all voxels.
		"""
		norm_density_map = density_map/np.sum( density_map )
		return norm_density_map


	def get_contour_level( self, density_map: np.ndarray ):
		"""
		Given a normalized density map, find the optimal contour
			level for visualizing the map in ChimeraX.
		Specifically, I want to determine the contour level at which the isosurface
			contains 90% of the total probability.
		"""
		flatten = density_map.flatten()
		# Sort in descending order.
		sort_idx = np.argsort( flatten )[::-1]
		sorted_density = flatten[sort_idx]

		cumsum = np.cumsum( sorted_density )
		idx = np.searchsorted( cumsum, self.target_prob )
		level = sorted_density[idx]
		return level


	def create_localization_probability_density( self ):
		"""
		Obtain density maps for the set of all superposed models.
		Convert each model to a density map.
			Using the molprobity_score as the resolution.
		Compute the localization probability map.
		Identify the contour level for viewing the denisty map in ChimeraX.
		"""
		# warnings.filterwarnings( "ignore" )
		box, origin = self.get_bounding_box()
		superposed_density = []
		for model_id in self.model_ids:
			superposed_model_file = self.get_superposed_model_file( model_id = model_id )
			density_map_file = self.get_density_map_file( model_id = model_id )
			resolution = self.molpobity_dict[model_id]["relaxed"]["MolProbity score"]

			eman2_pdb2density(
				modeL_file = superposed_model_file,
				density_file = density_map_file,
				apix = self.apix,
				res = resolution,
				box = box )
			density_map = self.parse_mrcfile( mrc_file = density_map_file )
			# density_map = self.normalize_density_map( density_map = density_map )
			superposed_density.append( density_map )
		superposed_density = np.stack( superposed_density )
		# Get the localization density across a set of superposed models.
		ld = np.sum( superposed_density, axis = 0  )
		# Get the localization probability.
		lpd = self.normalize_density_map( density_map = ld )

		# Save the density map on disk.
		self.save_to_mrc(
			density_map = lpd,
			apix = self.apix,
			origin = origin,
			ld_file = self.ld_file )

		return lpd

	################################################################################
	################################################################################
	def write_chimeraX_script( self, contour_level: float ):
		"""
		Write a python script for viewing the
			localization density mpa in ChimeraX.
		Save the localization density as a .png file.
		"""
		w = open_file_handler( self.ld_chimerax_script, "w" )
		w.writelines( "from chimerax.core.commands import run\n\n" )
		w.writelines( f"run( session, f\'open {self.ld_file}\' )\n" )
		w.writelines( f"run( session, \'volume #1 level {contour_level}\' )\n" )
		w.writelines( "run( session, \'preset silhouettes\' )\n" )
		w.writelines( "run( session, \'view #1\' )\n" )
		w.writelines( "run( session, \'save ld.png height 1080 width 1080\' )\n" )
		w.writelines( "run( session, \'close session\' )\n" )
		w.close()

		print( "To run the ChimeraX script in headless mode: \n" +
			"\t\tchimerax --offscreen --exit --script script.py" )


if __name__ == "__main__":
	analysis_dict = np.load(
		"/home/kartik/Documents/IMP_Rewired/imp_dl/benchmark/xlmerged_modeling/8wtd/version_5/analysis/analysis_dict.npy", allow_pickle = True ).item()
	print( analysis_dict.keys() )
	LocalizationDensity(
		analysis_dir = "/home/kartik/Documents/IMP_Rewired/imp_dl/benchmark/xlmerged_modeling/8wtd/version_5/analysis",
		struct_format = "pdb",
		model_ids = analysis_dict["selected_good_models"],
		molpobity_dict = analysis_dict["molprob"],
		ld_config = {"cpu_cores": 10, "apix": 1.0, "padding": 10.0, "target_prob": 0.9, "clean_up": False}
	).forward()
