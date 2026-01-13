"""
Contains wrapper for RMSD computation using USalign.
"""
from typing import List, Dict
import os, copy, time, warnings
from ml_collections import ConfigDict
import numpy as np

from utils.utils import run_subprocess
from utils.tools import ( load_ensemble, compute_rmsd, usalign, get_alignment_score )

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
					ensemble_dir: str,
					ref_model: int = None ):
		self.sys_name = sys_name
		self.rmsd_config = rmsd_config
		self.model_ids = model_ids
		# Predicted structure saved as .pdb/.cif
		self.struct_format = struct_format
		self.analysis_dir = analysis_dir
		# Dir containing predicted structures.
		self.ensemble_dir = ensemble_dir
		# Model with which to start structural similarity comparison.
		self.ref_model = ref_model

		# Dict to store relaxed models and relaxation metadata.
		self.rmsd_dict = {}
		self.selected_model_index = np.array( [] )


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

		if self.rmsd_config.tool == "mdanalysis":
			models = self.rmsd_pipeline_mdanalysis()
		elif self.rmsd_config.tool == "usalign":
			models = self.rmsd_pipeline_usalign()
		else:
			raise ValueError( f"Incorrect tool specified - {self.rmsd_config.tool} - for RMSd computation..." )
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
		Given model_id1 and model_id2, we remove all model_id2's which are structurally similar.
		Returhn model IDs for all selected models.
		"""
		warnings.filterwarnings( "ignore" )
		selected_models = []
		ignore_models = []

		if len( self.model_ids ) == 1:
			selected_models = self.model_ids
		else:
			for i in range( len( self.model_ids ) ):
				if i == 0 and self.ref_model is not None:
					model_id1 = self.ref_model
				else:
					model_id1 = self.model_ids[i]
				if model_id1 in ignore_models:
					continue
				u1 = load_ensemble(
					struct_file = os.path.join( self.ensemble_dir, f"model_{model_id1}.pdb" )
					)
				for j in range( i, len( self.model_ids ) ):
					model_id2 = self.model_ids[j]
					if model_id1 == model_id2 or model_id2 in ignore_models:
						continue

					u2 = load_ensemble(
						struct_file = os.path.join( self.ensemble_dir, f"model_{model_id2}.pdb" )
						)

					rmsd = compute_rmsd( to_align = u2, ref = u1, ref_frame = 0 )
					if rmsd[0, -1] <= self.rmsd_config.similarity_cutoff:
						ignore_models.append( model_id2 )
					else:
						self.rmsd_dict[f"model_{model_id1}_{model_id2}"] = {
							"rmsd": rmsd[0, -1], "tm": None}

				if model_id1 not in selected_models:
					selected_models.append( model_id1 )

		return np.array( selected_models )

	################################################################################
	################################################################################
	def rmsd_pipeline_usalign( self ):
		"""
		For all good-scoring models, compute all-vs-all RMSD.
		Remove structurally similar models (RMSD <= similarity_cutoff).
		Using USalign for computing RMSD.
		"""
		metric = self.rmsd_config.metric

		selected_models = []
		ignore_models = []
		if len( self.model_ids ) == 1:
			selected_models = copy.copy( self.model_ids )
		else:
			total_models = len( self.model_ids )
			for i in range( total_models ):
				if i == 0 and self.ref_model is not None:
					model_id1 = self.ref_model
				else:
					model_id1 = self.model_ids[i]
				if model_id1 in ignore_models:
					continue
				for j in range( i, total_models ):
					model_id2 = self.model_ids[j]
					if model_id1 == model_id2 or model_id2 in ignore_models:
						continue

					rmsd, tm = self.get_tm_from_usalign(
						model_id1 = model_id1,
						model_id2 = model_id2 )
					similarity = rmsd if metric == "rmsd" else tm

					if metric == "rmsd" and similarity <= self.rmsd_config.similarity_cutoff:
						ignore_models.append( model_id2 )
					elif metric == "tm" and similarity >= self.rmsd_config.similarity_cutoff:
						ignore_models.append( model_id2 )
					else:
						self.rmsd_dict[f"model_{model_id1}_{model_id2}"] = {
							"rmsd": rmsd, "tm": tm}

				if model_id1 not in selected_models:
					selected_models.append( model_id1 )
		return np.array( selected_models )


	def get_tm_from_usalign( self, model_id1: int, model_id2: int ):
		"""
		Run USalign and return the RMSD and TM-score.
		"""
		stdout_file = usalign(
			usalign_script = self.rmsd_config.usalign_script,
			model_id1 = model_id1,
			model1_file = self.get_struct_file( model_id = model_id1 ),
			model_id2 = model_id2,
			model2_file = self.get_struct_file( model_id = model_id2 ),
			tmp_dir = self.tmp_dir,
			mol = self.rmsd_config.mol,
			mm = self.rmsd_config.mm,
			ter = self.rmsd_config.ter,
			)
		rmsd, tm = get_alignment_score( stdout_file )
		return rmsd, tm
