"""
Contains wrapper for RMSD computation using USalign.
"""
from typing import List, Tuple, Dict, Iterable
import os, copy, time, warnings
from ml_collections import ConfigDict
from multiprocessing import Pool
import tqdm
import numpy as np

from utils.utils import run_subprocess
from utils.tools import ( load_ensemble, compute_rmsd, usalign, get_alignment_score )

class StructuralSimilarity():
	"""
	Contains methods for computing RMSD.
	Given two sets of models, we compute all-v-all similarity.
	"""
	def __init__(
		self,
		model_ids1: List[int],
		model_files1: List[str],
		model_ids2: List[int],
		model_files2: List[str],
		tmp_dir_path: str,
		mol: str = "prot",
		mm: int = 1,
		ter: int = 1
	):
		self.model_ids1 = model_ids1
		self.model_files1 = model_files1
		self.model_ids2 = model_ids2
		self.model_files2 = model_files2
		self.tmp_dir_path = tmp_dir_path

		# USalign parameters
		self.mol = mol
		self.mm = mm
		self.ter = ter

		# Dict to store the structural similarity.
		self.similarity_dict = {}


	def forward( self ):
		"""
		"""
		self.struct_models_exist()
		self.create_tmp_dir()

		# Sanity check.
		if self.rmsd_config.metric not in ["rmsd", "tm"]:
			raise ValueError( "Incorrect structural similarity metric specified. Use rmsd/tm..." )
		if self.rmsd_config.similarity_cutoff < 0.0:
			raise ValueError( f"Cannot use negative similarity cutoff..." )
		if self.rmsd_config.metric == "tm":
			if self.rmsd_config.similarity_cutoff > 1.0:
				raise ValueError( "Similarity cutoff for TM-score cannot be >1.0..." )

		self.compute_structural_similarity_parallel()

		if self.rmsd_config.clean_up:
			self.remove_tmp_dir()
		return self.similarity_dict


	def create_tmp_dir( self ):
		"""
		Create a temporary dir for storing intermediate
			files for rmsd computation.
		"""
		self.tmp_dir = os.path.join( self.analysis_dir, "rmsd_tmp" )
		os.makedirs( self.tmp_dir, exist_ok = True )


	def remove_tmp_dir( self ):
		cmd = ["rm", "-r", f"{self.tmp_dir}"]
		run_subprocess( cmd )


	def struct_models_exist( self ):
		"""
		Check if the predicted structure files exist on disk.
		"""
		for model_file in self.model_files1:
			if not os.path.exists( self.model_file ):
				raise FileNotFoundError( 
					f"Model1: {model_file} does not exist..."
				 )
		for model_file in self.model_files2:
			if not os.path.exists( self.model_file ):
				raise FileNotFoundError( 
					f"Model2: {model_file} does not exist..."
				 )

	################################################################################
	################################################################################
	def compute_structural_similarity_parallel( self ):
		"""
		Parallelize similarity computation across all models in set1 (model_is1).
		self.similarity_dict
		{
			model_id1: {
				model_id2: {
					rmsd: float,
					tm: float,
				}
			}
		}
		"""
		with Pool( self.cpu_cores ) as p:
			for result in tqdm.tqdm(
				p.imap_unordered(
					self.compute_structural_similarity_sequentially(
							zip( self.model_ids1, self.model_files1 )
						),
					chunksize = self.cpu_cores//2
				),
				total = len( self.model_files1 )
			):
				model_id1, sim_dict = result
				self.similarity_dict[model_id1] = sim_dict


	def compute_structural_similarity_sequentially( self, model1: Iterable[Tuple[int, str]] ):
		"""
		Compute the similarity of the model1 with all models in set 2 (model_ids2).
		This is done sequentially.
		"""
		sim_dict = {}
		model_id1, model_file1 = list( model1 )
		for i, model_id2 in enumerate( self.model_ids2 ):
			model_file2 = self.model_files2[i]
			rmsd, tm = self.get_similarity_from_usalign(
				model_id1 = model_id1,
				model_file1 = model_file1,
				model_id2 = model_id2,
				model_file2 = model_file2
			)
			sim_dict[model_id2] = {
				"rmsd": rmsd,
				"tm": tm
			}
		return model_id1, sim_dict


	def get_similarity_from_usalign(
		self,
		model_id1: int,
		model_id2: int,
		model_file1: str,
		model_file2: str
		):
		"""
		Run USalign and return the RMSD and TM-score.
		"""
		stdout_file = usalign(
			usalign_script = self.rmsd_config.usalign_script,
			model_id1 = model_id1,
			model1_file = model_file1,
			model_id2 = model_id2,
			model2_file = model_file2,
			tmp_dir = self.tmp_dir,
			mol = self.mol,
			mm = self.mm,
			ter = self.ter,
			)
		rmsd, tm = get_alignment_score( stdout_file )
		return rmsd, tm

