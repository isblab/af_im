"""
Contains functions to perform AMBER relaxation.
Taken from OpenFold.
"""
from typing import List, Tuple, Dict, Any
import os, time, warnings
from ml_collections import ConfigDict
import numpy as np
from multiprocessing import Pool
import tqdm

from utils.utils import read_json, write_json
from utils.pdb_utils import SaveModels

from openfold.np import protein
from openfold.np.relax import amber_minimize, utils


class AmberRelaxation():
	"""
	Contains methods for performing AMBER relaxation.
	"""
	def __init__( self,
					sys_name: str,
					amber_config: ConfigDict,
					model_ids: List[int],
					protein_obj_dict: Dict[int, protein.Protein],
					analysis_dir: str ):
		self.sys_name = sys_name
		self.amber_config = amber_config
		self.model_ids = model_ids
		self.protein_obj_dict = protein_obj_dict
		self.analysis_dir = analysis_dir

		# Dict to store relaxed models and relaxation metadata.
		self.relax_dict = {}


	def forward( self ):
		"""
		Get the model_id's for all models to perform AMBER relaxation on.
		Run relaxation serially or parallelize it.
		Save the relaxed model
		"""
		self.check_input_type()
		self.create_required_dir()
		self.load_relax_dict()
		self.relax()
		self.save_relaxed_struct()
		print( "\n" )


	def check_input_type( self ):
		"""
		The input must be an instance of List.
		Both must have the same no. of elements.
		"""
		print( "Validating inputs for relaxation" )
		if not isinstance( self.model_ids, np.ndarray ):
			raise ValueError( f"model_ids must be a numpy.ndarray. " +
				f"Provided {type( self.model_ids )}..." )

		if not isinstance( self.protein_obj_dict, Dict ):
			raise ValueError( "protein_obj_dict must be a Dict. " +
				f"Provided {type( self.protein_obj_dict )}..." )

		if len( self.model_ids ) != len( self.protein_obj_dict ):
			raise ValueError( f"Missing model in model_ids ({len( self.model_ids )}) or " +
				f"protein_obj_dict ({len( self.protein_obj_dict )})..." )


	def create_required_dir( self ):
		"""
		Create the required directories.
		"""
		relaxed_model_dir = self.amber_config.relaxed_model_dir

		self.relaxed_model_dir = os.path.join( self.analysis_dir, relaxed_model_dir )
		os.makedirs( self.relaxed_model_dir, exist_ok = True )


	def load_relax_dict( self ):
		"""
		Create the relax_dict file path.
		Load the relaxation dict, if it exists.
		"""
		self.relax_dict_file = os.path.join(
			self.analysis_dir,
			f"{self.amber_config.amber_logs_file}.npy" )

		if os.path.exists( self.relax_dict_file ):
			self.relax_dict = np.load( self.relax_dict_file, allow_pickle = True ).item()


	def save_relax_dict( self ):
		"""
		Save relax_dict on disk.
		"""
		np.save( self.relax_dict_file, self.relax_dict, allow_pickle = True )


	################################################################################
	################################################################################
	def relax( self ):
		"""
		Run relaxation pipeline either in series or in parallel.
		"""
		# print( "\n\n" + "-"*70 )
		# print( "Running AMBER relaxation" )
		# print( "-"*70 + "\n" )
		if self.amber_config.parallelize:
			self.relax_in_parallel()
		else:
			self.relax_in_series()


	def relax_in_series( self ):
		"""
		Serially run the relaxation pipeline.
		"""
		for model_id in self.model_ids:
			print( f"Performing AMBER relaxation for model {model_id}" )
			if model_id in self.relax_dict:
				continue
			else:
				ts = time.perf_counter()
				_, min_pdb, debug_dict = self.run_relaxation_pipeline(
					relax_input = ( model_id, self.protein_obj_dict[model_id] )
					)
				self.relax_dict[model_id] = {
					"prot": min_pdb,
					"metadata": debug_dict
				}
				time_taken = time.perf_counter() - ts
				print( f"Time taken = {time_taken/60} minutes" )
				self.save_relax_dict()


	def models_to_relax_in_parallel( self ):
		"""
		Select the models to relaxed in parallel.
		"""
		subset_model_ids = []
		subset_protein_obj = []
		total = 0
		for model_id in self.model_ids:
			if model_id in self.relax_dict:
				continue
			else:
				subset_model_ids.append( model_id )
				subset_protein_obj.append( self.protein_obj_dict[model_id] )
				total += 1
		return total, zip( subset_model_ids, subset_protein_obj )


	def relax_in_parallel( self ):
		"""
		Parallelize relaxation pipeline.
		"""
		total, parallel_input = self.models_to_relax_in_parallel()
		with Pool( self.amber_config.cpu_cores ) as p:
			for result in tqdm.tqdm( 
				p.imap_unordered( self.run_relaxation_pipeline, parallel_input ),
				total = total
				):
				model_id, min_pdb, debug_dict = result

				self.relax_dict[model_id] = {
					"prot": min_pdb,
					"metadata": debug_dict
				}
				self.save_relax_dict()


	################################################################################
	################################################################################
	def run_relaxation_pipeline( self, relax_input: Tuple[int, Dict] ):
		"""
		Run AMBER relaxation.
		Post-process the relaxation output.
		Save output as a model to a .pdb/.cif file on disk.
		"""
		# model_id, prot = prot.items()
		model_id, prot = relax_input
		out, pdb_str_prior_min = self.amber_relax( prot = prot )
		min_pdb, debug_dict = self.post_process_relax_output(
				prot = prot,
				out = out,
				pdb_str_prior_min = pdb_str_prior_min )

		return model_id, min_pdb, debug_dict


	def amber_relax( self, prot: protein.Protein ) -> Dict[str, Any]:
		"""
		Taken from openfold/utils/np/relax/relax.py
		Modified implementation of AmberRelaxation.process().
		Runs Amber relax on a prediction, adds hydrogens, returns PDB string.
		"""
		# get pdb_str before minimization.
		pdb_str_prior_min = amber_minimize.clean_protein( prot )
		out = amber_minimize.run_pipeline(
			prot = prot,
			max_iterations = self.amber_config.max_iterations,
			tolerance = self.amber_config.tolerance,
			stiffness = self.amber_config.stiffness,
			exclude_residues = self.amber_config.exclude_residues,
			max_outer_iterations = self.amber_config.max_outer_iterations,
			use_gpu = self.amber_config.use_gpu,
		)
		return out, pdb_str_prior_min


	def post_process_relax_output( self, prot: protein.Protein,
									out: Dict[str, Any], pdb_str_prior_min: str
									) -> Tuple[protein.Protein, Dict[str, float]]:
		"""
		Post processing of the AMBER relaxation output.
		Compute accessory metrics, including initial_energy,
			final_energy, no. of attempts, and RMSD between
			initial and final structure.
		Add missing atoms, bfactor, and convert to pdb string.
		Compute structural violations.
    	"""
		min_pos = out["pos"]
		start_pos = out["posinit"]
		rmsd = np.sqrt( np.sum( ( start_pos - min_pos ) ** 2 ) / start_pos.shape[0] )
		debug_data = {
			"initial_energy": out["einit"],
			"final_energy": out["efinal"],
			"attempts": out["min_attempts"],
			"rmsd": rmsd,
			"violations": out["structural_violations"][
			"total_per_residue_violations_mask"]
		}
		pdb_str = amber_minimize.clean_protein( prot )
		if min_pos.shape[0] != pdb_str.count("\nATOM"):
			warnings.warn( "The number of positions must match the number of atoms. " +
							f"min_pos = {min_pos.shape[0]} \t pdb_str = " + str( pdb_str.count('\nATOM') ) )
			# pdb_str = pdb_str_prior_min
			pdb_str = out["min_pdb"]
			print( pdb_str.count('\nATOM') )

		# Adds missing atoms to Protein instance.
		min_pdb = utils.overwrite_pdb_coordinates( pdb_str, min_pos )
		min_pdb = utils.overwrite_b_factors( min_pdb, prot.b_factors )
		utils.assert_equal_nonterminal_atom_types(
		    protein.from_pdb_string( min_pdb ).atom_mask, prot.atom_mask
		)

		min_pdb = protein.add_pdb_headers( prot, min_pdb )

		return min_pdb, debug_data


	################################################################################
	################################################################################
	def save_relaxed_struct( self ):
		"""
		Save the relaxed structures as separate models in the PDB/CIF file.
		"""
		output_format = self.amber_config.output_format
		save_single_model = self.amber_config.save_single_model

		save_model_obj = SaveModels( title = self.sys_name,
									output_format = output_format,
									ensemble_dir = self.relaxed_model_dir,
									save_single_model = save_single_model )
		# Initialize the System object.
		save_model_obj.initialize_system()

		for model_id in self.model_ids:
			min_pdb = self.relax_dict[model_id]["prot"]
			final_prot = protein.from_pdb_string( min_pdb )

			save_model_obj.add_model( prot = final_prot, model_id = model_id )

		relaxed_ensemble_file = os.path.join(
			self.analysis_dir, f"{self.sys_name}_relaxed_ensemble"
			)
		save_model_obj.save(
			system = save_model_obj.system,
			output_path = relaxed_ensemble_file )
