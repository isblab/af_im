"""
Contains moduole for computing the DOckQ score between two sets of models.
"""
from typing import List, Tuple, Dict, Iterable
import os, shutil
from multiprocessing import Pool
import tqdm

from utils.tools import dockq, dockq_cli
from utils.pdb_utils import remap_chains_cif, remap_chains_pdb


class DockQ():
	"""
	Contains methods for computing DockQ.
	Given two sets of models, we compute all-v-all DockQ.

	Inputs:
	----------
	model_ids1: a list of integer identifiers for all modls in set 1 (model1).
	model_files1: a list of structure file paths for all experiemntal/predicted
		models in set 1 (model1).
	model_ids2: a list of integer identifiers for all modls in set 2 (model2).
	model_files2: a list of structure file paths for all experiemntal/predicted
		models in set 2 (model2).
	native_sys_chain_map: dict containing mapping between the native
		and system chain IDs.
	use_native_chains_for_model2: If True, sets the use_native_chains arg for
		remap_chains_cif() to True, else False.
		This stands on the assumption that model1's would always be the predicted
			model and the model2's could be predicted models or the native structure.
		For predicted models, the chain IDs may not be the native chain IDs.
	tmp_dir_path: path for a temporary dir to store intermediate files.
	cpu_cores: no. of CPU coress to be used for parallelizing DockQ computation.
	"""
	def __init__(
		self,
		model_ids1: List[int],
		model_files1: List[str],
		model_ids2: List[int],
		model_files2: List[str],
		# native_sys_chain_map: Dict[str, str],
		# use_native_chains_for_model2: bool,
		tmp_dir_path: str,
		cpu_cores: int,
	):
		self.model_ids1 = model_ids1
		self.model_files1 = model_files1
		self.model_ids2 = model_ids2
		self.model_files2 = model_files2
		self.tmp_dir_path = tmp_dir_path

		# self.use_native_chains_for_model2 = use_native_chains_for_model2
		# self.native_sys_chain_map = native_sys_chain_map

		self.cpu_cores = cpu_cores

		# Dict to store remapped file paths.
		self.remapped_dict = {}
		# Dict to store the structural similarity.
		self.dockq_dict = {}


	def forward( self ):
		"""
		"""
		self.create_tmp_dir()
		# self.run_chain_remapping()

		self.compute_dockq_parallel()

		self.remove_tmp_dir()
		return self.dockq_dict


	def create_tmp_dir( self ):
		"""
		Create a temporary dir for storing intermediate
			files for dockq computation.
		"""
		os.makedirs( self.tmp_dir_path, exist_ok = True )


	def remove_tmp_dir( self ):
		shutil.rmtree( self.tmp_dir_path )

	################################################################################
	################################################################################
	def run_chain_remapping( self ):
		"""
		For the given models 1 and 2, remap the chains
			to system chai IDs.
			Ssytem chain IDs start from A.
		The remapped files would be stored in the tmp dir.

		self.remapped_dict: {
			model_files1: List[str],
			model_files2: List[str]
		}
		"""
		remapped_files1 = self.remap_chains_in_struct(
			model_ids = self.model_ids1,
			model_files = self.model_files1,
			remap_prefix = "model1",
			use_native_chains = False
		)
		self.remapped_dict["model_files1"] = remapped_files1

		remapped_files2 = self.remap_chains_in_struct(
			model_ids = self.model_ids2,
			model_files = self.model_files2,
			remap_prefix = "model2",
			use_native_chains = self.use_native_chains_for_model2
		)
		self.remapped_dict["model_files2"] = remapped_files2


	def remap_chains_in_struct(
		self,
		model_ids: List[int],
		model_files: List[str],
		remap_prefix: str,
		use_native_chains: bool
	) -> List[str]:
		"""
		For running DockQ, the chain IDs in the native
			and predicted structures must be the same
			(model1 and model2 here).

		Inputs:
		----------
		model_ids: a list of integer identifiers for a model.
		model_files: a list of file paths for the predicted model.
		remap_prefix: prefix for the remapped file.
		use_native_chains: if True, uses the native chain ID mapping
			in the given map_dict.
			This is required because the native chains are defined based
			on the experimental structure, which may not be the same for
			the predicted structures.
			Hence, for predicted structures we redefine native chains based
			on the auth_asym_ids.

		Returns:
		----------
		remapped_files: a list of all the file paths for the
			remapped structure.
		"""
		remapped_files = []
		for i, model_id in enumerate( model_ids ):
			model_file = model_files[i]
			base, ext = os.path.splitext( model_file )
			remapped_file = os.path.join(
				self.tmp_dir_path,
				f"{remap_prefix}_model{model_id}_remapped{ext}"
			)
			if ext == ".cif":
				remap_chains_cif(
					struct_file = model_file,
					map_dict = self.native_sys_chain_map,
					remapped_file = remapped_file,
					use_native_chains = use_native_chains
				)
			elif ext == ".pdb":
				remap_chains_pdb(
					struct_file = model_file,
					map_dict = self.native_sys_chain_map,
					remapped_file = remapped_file
				)
			else:
				raise ValueError(
					f"Incorrect format for the structure file for model = {model_id}..."
					)
			remapped_files.append( remapped_file )
		return remapped_files

	################################################################################
	################################################################################
	def prep_input_for_parallel( self ) -> Tuple[List, int]:
		"""
		Create a list of all-v-all combinations of model1-model2 pairs
			for parallel processing.
		
		Returns:
		----------
		parallel_input: a list of tuples containing the model_id and file path for
			the structures to compute the DockQ.
			model_id1, model_file1, model_id2, model_file2
		total: total no. of model1-model2 pairs.
		"""
		parallel_input = []
		total = 0
		# for m1, f1 in zip( self.model_ids1, self.remapped_dict["model_files1"] ):
		# 	for m2, f2 in zip( self.model_ids2, self.remapped_dict["model_files2"] ):
		for m1, f1 in zip( self.model_ids1, self.model_files1 ):
			for m2, f2 in zip( self.model_ids2, self.model_files2 ):
				out_file = os.path.join(
					self.tmp_dir_path,
					f"dockq_out_{m1}-{m2}.txt"
					)
				err_file = os.path.join(
					self.tmp_dir_path,
					f"dockq_error_{m1}-{m2}.txt"
					)
				parallel_input.append(
					( m1, f1, m2, f2, out_file, err_file )
				)
				total += 1
		return parallel_input, total


	def compute_dockq_parallel( self ):
		"""
		We parallelize computing DockQ across pairs of model1-model2.

		self.dockq_dict: {
			model_id1: {
				model_id2: dockq
			}
		}
		"""
		parallel_input, total = self.prep_input_for_parallel()

		with Pool( self.cpu_cores ) as p:
			for result in tqdm.tqdm(
				p.imap_unordered(
					self.compute_dockq_sequential,
					parallel_input,
					chunksize = self.cpu_cores//2
				),
				desc = "DockQ",
				total = total
			):
				model_id1, model_id2, d = result
				if not model_id1 in self.dockq_dict:
					self.dockq_dict[model_id1] = {}
				self.dockq_dict[model_id1][model_id2] = d


	def compute_dockq_sequential(
		self,
		input_pair: Tuple,
		) -> Tuple[int, Dict[int, float]]:
		"""
		For the given model1, compute DockQ wrt all model2's.

		Inputs:
		----------
		input_pair: a tuple containing the model_id and file path for
			the structures to compute the DockQ.
			model_id1, model_file1, model_id2, model_file2

		Returns:
		----------
		model_ids1: integer identifiers for the model in set 1 (model1).
		model_ids2: integer identifiers for the model in set 2 (model2).
		dockq score: DockQ score for the given models.
		"""
		model_id1, model_file1, model_id2, model_file2, out_file, err_file = input_pair
		# d = dockq(
		# 	native_file = model_file1,
		# 	model_file = model_file2
		# )
		d = dockq_cli(
			native_file = model_file1,
			model_file = model_file2,
			out_file = out_file,
			err_file = err_file
		)
		return model_id1, model_id2, d
		
