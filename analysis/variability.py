"""
Contains module to determine the ensemble variability given a set of models.
"""
from typing import List, Tuple, Dict
import os, shutil
import numpy as np
from functools import partial
from multiprocessing import Pool
import tqdm

from Bio.PDB import Superimposer, PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

from utils.pdb_utils import Parser



class EnsembleVariability():
	"""
	Compute the per-residue RMSF within a set of models.

	Inputs:
	----------
	model_ids: a list of integer identifiers for all models.
	model_files: a list of structure file paths for all experiemntal/predicted models.
	tmp_dir_path: path for a temporary dir to store intermediate files.
	cpu_cores: no. of CPU coress to be used for parallelizing DockQ computation.

	"""
	def __init__(
		self,
		model_ids: List[int],
		model_files: List[str],
		tmp_dir_path: str,
		cpu_cores: int
	):
		self.model_ids = model_ids
		self.model_files = model_files
		self.tmp_dir_path = tmp_dir_path

		self.cpu_cores = cpu_cores

		self.per_res_rmsd_dict = {}


	def forward( self ):
		"""
		Superpose all models to a reference.
			Using the 1st model as reference.
		Compute per-residue RMSF.
		Parse the per-residue pLDDT from the predicted structure.
		"""
		self.create_tmp_dir()

		self.superpose_models()

		rmsf_per_residue, plddt = self.compute_rmsd()
		self.per_res_rmsd_dict = {
			"rmsf_per_residue": rmsf_per_residue,
			"plddt": plddt
		}
		self.remove_tmp_dir()
		return self.per_res_rmsd_dict


	def create_tmp_dir( self ):
		"""
		Temporary directory is used for storing the intermediate files
			from running Molprobity validation.
		"""
		os.makedirs( self.tmp_dir_path, exist_ok = True )


	def remove_tmp_dir( self ):
		shutil.rmtree( self.tmp_dir_path )


	def get_superposed_model_file( self, model_id: int ) -> str:
		"""
		Return the path to the superposed model file for the given model_id.
		"""
		superposed_struct_file = os.path.join(
			self.tmp_dir_path,
			f"superposed_{model_id}.pdb"
		)
		return superposed_struct_file

	################################################################################
	################################################################################
	def get_atoms_from_model(
		self,
		structure: Structure,
		return_coords = False
		) -> List[Atom | np.ndarray]:
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


	def superpose_model(
		self,
		fixed_atoms: List[Atom | np.ndarray],
		moving_model_id: int
		):
		"""
		Superpose the moving_model onto the fixed_model
			and save the superposed moving model on disk.
		"""
		moving_model_file = self.model_files[moving_model_id]

		p = Parser( moving_model_file )
		moving_struct = p.get_structure()
		moving_atoms = self.get_atoms_from_model( structure = moving_struct )

		superpose = Superimposer()
		superpose.set_atoms( fixed = fixed_atoms, moving = moving_atoms )
		superpose.apply( moving_struct.get_atoms() )

		io = PDBIO()
		io.set_structure( moving_struct )
		superposed_model_file = self.get_superposed_model_file( model_id = moving_model_id )
		io.save( superposed_model_file )


	def superpose_models( self ):
		"""
		Obtain a set of superposed models.
		Will superpose all models onto the first model.
		"""
		fixed_model_file = self.model_files[0]
		p = Parser( fixed_model_file )
		fixed_model = p.get_structure()
		fixed_atoms = self.get_atoms_from_model( structure = fixed_model )

		partial_func = partial( self.superpose_model, fixed_atoms )
		with Pool( self.cpu_cores ) as p:
			for result in tqdm.tqdm(
				p.imap_unordered( partial_func, self.model_ids[1:] ),
				desc = "Superpose",
				total = len( self.model_ids[1:] )
			):
				pass

	################################################################################
	################################################################################
	def parse_structure(
		self,
		model_id: int,
		use_superposed_struct: bool = False
		) -> Dict[str, np.ndarray]:
		"""
		Parse the structure file, and extract the following for all residues:
			Ca-coordinates
			Ca-pLDDT
		By construction, all residues have a Ca atom.
		"""
		if use_superposed_struct:
			model_file = self.get_superposed_model_file( model_id = model_id )
		else:
			model_file = self.model_files[model_id]
		p = Parser( model_file )
		data = {k:[] for k in ["coords", "plddt"]}
		model = p.structure[0]
		for chain in model:
			for residue in chain:
				data["coords"].append( residue["CA"].get_coord() )
				data["plddt"].append( residue["CA"].get_bfactor() )
		data["coords"] = np.array( data["coords"] ).reshape( -1, 3 )
		data["plddt"] = np.array( data["plddt"] ).reshape( -1, 1 )
		return data
				

	def get_coords_n_plddt( self ) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Extract the ca coordinates and plddt for all models.

		Returns:
		----------
		coords: [M, N, 3]; Ca-coordinates for the specified model.
		plddt: [M, N, 1]; Ca-pLDDT for the specified model.
			where, M -> no. of models; N -> no. of resdiues.
		"""
		plddt = []
		coords = []
		use_superposed_struct = False
		for model_id in self.model_ids:
			if model_id != 0:
				use_superposed_struct = True
			data = self.parse_structure(
				model_id = model_id,
			use_superposed_struct = use_superposed_struct
			)
			coords.append( data["coords"] )
			plddt.append( data["plddt"] )
		coords = np.stack( coords )
		plddt = np.stack( plddt )
		return coords, plddt


	def compute_rmsd( self ) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Given the set of superposed models, compute per-residue RMSF for all
			models wrt the reference model (centroid).

		Returns:
		----------
		rmsf_per_residue: [N]; per-residue RMSF for all models wrt the centroid model.
		plddt: [M, N, 1]; Ca-plddt for all models.
		"""
		coords, plddt = self.get_coords_n_plddt()

		# Using the 1st model as a reference for computing RMSD.
		# [1, N, 3]
		centroid = coords.mean( axis = 0, keepdims = True )

		diff = coords - centroid
		sq_diff = ( diff**2 ).sum( axis = -1 )
		# [N]
		rmsf_per_residue = np.sqrt( sq_diff.mean( axis = 0 ) )

		return rmsf_per_residue, plddt


