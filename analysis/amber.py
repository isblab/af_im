"""
Contains functions to perform AMBER relaxation.
Taken from OpenFold.
"""
from typing import Optional
from typing import Tuple, Dict, Any, Sequence, Optional
from ml_collections import ConfigDict
import numpy as np
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
		self.load_relax_logs()
		self.relax()


	def check_input_type( self ):
		"""
		The input must be an instance of List.
		Both must have the same no. of elements.
		"""
		print( "\n\n" + "-"*70 )
		print( "\033[1mValidating inputd for relaxation\033[0m" )
		print( "-"*70 + "\n" )
		if not isinstance( self.model_ids, List ):
			raise ValueError( "model_ids must be a List. " +
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

    	if save_single_model:
    		os.makedirs( os.path.join( self.analysis_dir, relaxed_model_dir ) )


	def load_relax_dict( self ):
		"""
		Create the logs dict file path.
		Load the relaxation logs dict, if it exists.
		"""
		self.relax_dict_file = os.path.join(
			self.analysis_dir,
			f"{self.amber_config.amber_logs_file}.json" )

		self.relax_dict = read_json( self.relax_dict_file )


	################################################################################
	################################################################################
	def relax( self ):
		"""
		Run relaxation pipeline either in series or in parallel.
		"""
		print( "\n\n" + "-"*70 )
		print( "\033[1mRunning AMBER relaxation\033[0m" )
		print( "-"*70 + "\n" )
		if self.amber_config.parallelize:
			self.relax_in_parallel()
		else:
			self.relax_in_series()


	def relax_in_series( self ):
		"""
		Serially run the relaxation pipeline.
		"""
		for i in range( self.model_ids ):
			print( f"Performing AMBER relaxation for model {self.model_ids[i]}" )
			if model_id in self.relax_dict:
				continue
			else:
				model_id, min_pdb, debug_dict = self.run_relaxation_pipeline(
					model_id = self.model_ids[i],
					prot = self.protein_obj_dict[i]
					)
				self.relax_dict[model_id] = {
					"prot": min_pdb,
					"metadata": debug_dict
				}
				write_json( self.relax_dict, self.relax_dict_file )


	def models_to_relax_in_parallel( self ):
		"""
		Select the models to relaxed in parallel.
		"""
		subset_model_ids = []
		subset_protein_obj = {}
		for model_id in self.model_ids:
			if model_id in self.relax_dict:
				continue
			else:
				subset_model_ids.append( model_id )
				subset_protein_obj = self.protein_obj_dict[model_id]
		return zip( subset_model_ids, subset_protein_obj )


	def relax_in_parallel( self ):
		"""
		Parallelize relaxation pipeline.
		"""
		parallel_input = self.models_to_relax_in_parallel()
		with Pool( self.amber_config.cpu_cores ) as p:
			for result in tqdm( 
				p.imp( self.run_relaxation_pipeline, parallel_input ),
				total = len( parallel_input )
				):
				model_id, min_pdb, debug_dict = result

				self.relax_dict[model_id] = {
					"prot": min_pdb,
					"metadata": debug_dict
				}
				write_json( self.relax_dict, self.relax_dict_file )


	################################################################################
	################################################################################
	def run_relaxation_pipeline( self, model_id: int, prot: protein.Protein ):
		"""
		Run AMBER relaxation.
		Post-process the relaxation output.
		Save output as a model to a .pdb/.cif file on disk.
		"""
		out = self.amber_relax( prot = prot )
		min_pdb, debug_dict = self.post_process_relax_output( out = out )

		return model_id, min_pdb, debug_dict


	def amber_relax( self, prot: protein.Protein ) -> Dict[str, Any]:
		"""
		Taken from openfold/utils/np/relax/relax.py
		Modified implementation of AmberRelaxation.process().
		Runs Amber relax on a prediction, adds hydrogens, returns PDB string.
		"""
        out = amber_minimize.run_pipeline(
            prot = prot,
            max_iterations = self.amber_config.max_iterations,
            tolerance = self.amber_config.tolerance,
            stiffness = self.amber_config.stiffness,
            exclude_residues = self.amber_config.exclude_residues,
            max_outer_iterations = self.amber_config.max_outer_iterations,
            use_gpu = self.amber_config.use_gpu,
        )
        return out


    def post_process_relax_output( self, out: Dict[str, Any] ) -> protein.Protein:
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
        # Adds missing atoms to Protein instance.
        pdb_str = amber_minimize.clean_protein( prot )
        min_pdb = utils.overwrite_pdb_coordinates( pdb_str, min_pos )
        min_pdb = utils.overwrite_b_factors( min_pdb, prot.b_factors )
        utils.assert_equal_nonterminal_atom_types(
            protein.from_pdb_string( min_pdb ).atom_mask, prot.atom_mask
        )

        min_pdb = protein.add_pdb_headers( prot, min_pdb )

        return min_pdb, debug_dict


	################################################################################
	################################################################################
    def save_relaxed_struct( self ):
    	"""
    	Save the relaxed structures as separate models in the PDB/CIF file.
    	"""
    	output_format = self.amber_config.output_format
    	relaxed_model_dir = self.amber_config.relaxed_model_dir
    	save_single_model = self.amber_config.save_all_models

		save_model_obj = SaveModels( title = self.sys_name,
									output_format = output_format,
									ensemble_dir = relaxed_model_dir,
									save_single_model = save_single_model )
		# Initialize the System object.
		save_model_obj.initialize_system()

    	for model_id in self.model_ids:
    		min_pdb = self.relax_dict[model_id]["prot"]
    		final_prot = protein.from_pdb_string( min_pdb )

    		save_model_obj.add_model( final_prot )

		relaxed_ensemble_file = os.path.join(
			self.analysis_dir, f"{self.sys_name}_relaxed_ensemble"
			)
		save_model_obj.save(
			system = save_model_obj.system,
			output_path = relaxed_ensemble_file )
