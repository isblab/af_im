"""
Contains wrapper for RMSD computation using USalign.
"""
from typing import List, Dict
import os, copy, time, warnings
from ml_collections import ConfigDict
import numpy as np

from utils.utils import run_subprocess
from utils.tools import ( load_ensemble, compute_rmsd )

class StructuralSimilarity():
	"""
	Contains methods for computing RMSD.
	"""
	def __init__( self,
					sys_name: str,
					rmsd_config: ConfigDict,
					model_ids: List[int],
					struct_format: str,
					analysis_dir: str,
					ensemble_dir: str ):
		self.sys_name = sys_name
		self.rmsd_config = rmsd_config
		self.model_ids = model_ids
		# Predicted structure saved as .pdb/.cif
		self.struct_format = struct_format
		self.analysis_dir = analysis_dir
		# Dir containing predicted structures.
		self.ensemble_dir = ensemble_dir

		# Dict to store relaxed models and relaxation metadata.
		self.rmsd_dict = {}
		self.selected_model_index = np.array( [] )


	def forward( self ):
		"""
		"""
		self.struct_models_exist()
		self.create_tmp_dir()
		models = self.rmsd_pipeline_mdanalysis()
		self.selected_model_index = models

		if self.rmsd_config.clean_up:
			self.remove_tmp_dir()


	def struct_models_exist( self ):
		"""
		Check if the predicted structure files exist on disk.
		"""
		for model_id in self.model_ids:
			struct_file = self.get_struct_file( model_id = model_id )
			if not os.path.exists( struct_file ):
				raise FileNotFoundError( 
					f"{struct_file} does not exist..."
				 )


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


	def get_struct_file( self, model_id: int ):
		"""
		Return the path to the predicted structure
			file for the given model_id.
		"""
		struct_file = os.path.join(
			self.ensemble_dir,
			f"model_{model_id}.{self.struct_format}" )
		return struct_file

	################################################################################
	################################################################################
	def rmsd_pipeline_mdanalysis( self ):
		"""
		Using MDAnalysis for computing RMSD.
		For each model compute RMSD against all other models and
			remove structurally similar models (RMSD <= similarity_cutoff).
		Returhn model IDs for all selected models.
		"""
		warnings.filterwarnings( "ignore" )
		selected_models = []
		ignore_models = []

		if len( self.model_ids ) == 1:
			selected_models = self.model_ids
		else:
			for i in range( len( self.model_ids ) ):
				model_id1 = self.model_ids[i]
				if model_id1 in ignore_models:
					continue
				u1 = load_ensemble(
					ensemble_file = os.path.join( self.ensemble_dir, f"model_{model_id1}.pdb" )
					)
				for j in range( i, len( self.model_ids ) ):
					model_id2 = self.model_ids[j]
					if model_id1 == model_id2 or model_id2 in ignore_models:
						continue

					u2 = load_ensemble(
						ensemble_file = os.path.join( self.ensemble_dir, f"model_{model_id2}.pdb" )
						)

					rmsd = compute_rmsd( to_align = u2, ref = u1, ref_frame = 0 )
					if rmsd[0, -1] <= 1.0:
						ignore_models.append( model_id2 )
					else:
						if model_id1 not in selected_models:
							selected_models.append( model_id1 )
							self.rmsd_dict[f"model_{model_id1}_{model_id2}"] = {
								"rmsd": rmsd[0, -1], "tm": None}
			# selected_model_index = np.where(
			# 	np.isin(
			# 		np.array( self.model_ids ), np.array( selected_models )
			# 		)
			# 	)

		return selected_models

	################################################################################
	################################################################################
	def rmsd_pipeline_usalign( self ):
		"""
		For all good-scoring models, compute all-vs-all RMSD.
		Remove structurally similar models (RMSD <= similarity_cutoff).
		Using USalign for computing RMSD.
		"""
		selected_model_index = []
		ignore_models = []
		if len( self.model_ids ) == 1:
			selected_model_index = copy.copy( self.model_ids )
		else:
			total_models = len( self.model_ids )
			for i in range( total_models ):
				model_id1 = self.model_ids[i]
				if model_id1 in ignore_models:
					continue
				for j in range( i, total_models ):
					model_id2 = self.model_ids[j]
					if model_id1 == model_id2 or model_id2 in ignore_models:
						continue
					
					stdout_file = self.usalign( model_id1 = model_id1, model_id2 = model_id2 )
					rmsd, tm = self.get_alignment_score( stdout_file )

					self.rmsd_dict[f"model_{model_id1}_{model_id2}"] = {
						"rmsd": rmsd, "tm": tm}

					if rmsd <= self.rmsd_config.similarity_cutoff:
						ignore_models.append( model_id2 )

				if model_id1 not in selected_model_index:
					selected_model_index.append( i )
		return np.array( selected_model_index )


	def usalign( self, model_id1: int, model_id2: int ):
		"""
		Use USalign for computing the TM-score
			and RMSD for the given models.
		Assuming model_id1 to be the reference.

		mol --> molecule type [auto, prot, RNA.
		mm --> multimeric laignment option.
			0: (default) alignment of two monomeric structures.
			1: alignment of two multi-chain oligomeric structures.
			2: alignment of individual chains to an oligomeric structure.
			Look at USalign -h option for more details.
		ter --> #chains to align.
			0: align all chains from all models.
			1: align all chains of the first model.
			2: (default) only align the first chain.

		USalign model1.pdb model2.pdb -ter 0 -mm 1 -mol prot
		"""
		model1 = self.get_struct_file( model_id = model_id1 )
		model2 = self.get_struct_file( model_id = model_id2 )

		stdout_file = os.path.join( self.tmp_dir, f"model_{model_id1}_{model_id2}.txt" )
		stderr_file = os.path.join( self.tmp_dir, f"error_{model_id1}_{model_id2}.txt" )

		cmd = [f"./{self.rmsd_config.usalign_script}", 
				f"{model1}",
				f"{model2}",
				"-mol", "prot",
				"-mm", f"{self.rmsd_config.mm}",
				"-ter", f"{self.rmsd_config.ter}"]

		run_subprocess(
			command = cmd,
			stdout_file = stdout_file,
			stderr_file = stderr_file
		)
		return stdout_file


	def get_alignment_score( self, stdout_file: str ):
		"""
		Return the TM-score and RMSD.
		Read the MMalign/USalign output stored in a txt file.
			Line14: Aligned length= 572, RMSD=   0.76, Seq_ID=n_identical/n_aligned= 1.000
			Line15: TM-score= 0.XXXXX (if normalized by length of Chain_1, i.e., LN=XX, d0=X.XX)
			Line16: TM-score= 0.XXXXX (if normalized by length of Chain_2, i.e., LN=XXX, d0=X.XX)
		"""
		with open( stdout_file, "r" ) as f:
			output = f.readlines()

		rmsd_line = output[14]
		tm_line = output[15]

		rmsd = float( rmsd_line.split( "," )[1].split( "RMSD=" )[1] )
		tm = float( tm_line.split( " " )[1] )
		return rmsd, tm

