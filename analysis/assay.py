"""
Script to perform analysis for the set of models obtained after fine-tuning.
Select models based on their loss.
	Use HDBSCAN for clustering.
	Select the largest cluster.
	Alternatively, 
"""
import os
import glob
from typing import List, Dict, Iterator
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
import hdbscan
from Bio.PDB import PDBParser, PDBIO, Structure, Model

from utils.utils import open_file_handler, run_subprocess
from utils.pdb_utils import Parser
from utils.create_plots import create_plot_from_dict



class Assay():
	"""
	Perform analysis and validation of the predicted ensemble of models.
		1. Segregate models into good and bad models.
		2. Assess data satisfaction for good models.
		3. Perfrom AMBER relaxation.
		4. Perform MolProbity validation.
	"""
	def __init__( self, sys_name: str,
						modeling_output_dir: str,
						seed_worker,
						cores: int, prec: int ):
		seed_worker()
		# Name of the system to be modeled.
		self.sys_name = sys_name
		# Output modeling dir for the system.
		self.modeling_output_dir = modeling_output_dir
		# No. of CPU cores to use.
		self.cores = 5
		# precision for floats.
		self.prec = 4


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.create_required_dir()
		self.load_stats_file()



	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		# JSON file to save model_num for good and bad models.
		self.clustering_output = os.path.join( self.analysis_dir,
										f"{self.sys_name}_clustered_models.json" )
		self.stats_file = os.path.join( self.modeling_output_dir, "Stats.npy" )


	def create_required_dir( self ):
		"""
		Check if required dir exist.
		Create the required directories if not already existing.
		"""
		if not os.path.exists( self.modeling_output_dir ):
			raise FileNotFoundError( "Required modeling dir: " +
								f"{self.modeling_output_dir} does not exist..." )

		os.makedirs( self.analysis_dir, exist_ok = True )


	def load_stats_file( self ):
		"""
		Load the stats file.
		"""
		self.stats_dict = np.load( self.stats_file, allow_pickle = True ).item()


	def get_assessment_metrics( self ):
		"""
		Select the required metrics for downstream assessment.
		Also select the model_ids.
		"""
		input_dict = {
			"model_id": np.array( self.stats_dict["model_id"] ),
			"violations": np.array( self.stats_dict["loss"]["violation"] ),
			"ccom": np.array( self.stats_dict["chain_center_of_mass"] ),
		}

		for name in self.stats_dict["metrics"]:
			metric = self.stats_dict["metrics"][name]
			input_dict[f"{name}_metric"] = np.array( metric )

		return input_dict


	def select_good_scoring_models( self ):
		"""
		Given the ensemble of predicted structures, separate good and bad models.
		For this we use the violation loss, ccom loss, and xl satisfaction.
		Bad models have: more violations, chain clashes, and low data satisfaction.
		"""
		input_dict = self.get_assessment_metrics()

		clust = Clustering()




	def assess_data_satisfaction( self ):



	def run_amber_relaxation( self ):


	def perform_molprobity_validation( self ):




	def create_analysis_dir( self ):
		"""
		Create a dir to store analysis results.
		"""
		self.analysis_dir = os.path.join( self.output_dir, "analysis" )
		if not os.path.exists( self.analysis_dir ):
			os.makedirs( self.analysis_dir )

	# 	self.tmp_dir = os.path.join( self.analysis_dir, "tmp" )
	# 	if not os.path.exists( self.tmp_dir ):
	# 		os.makedirs( self.tmp_dir )



	def prep_data( self ) -> np.array:
		"""
		Given the dict containing loss and metrics, do
			Select the following:
				Violation loss, CCOM loss, restraint loss.
				Restraint metrics
			Stack together in an array.
		"""
		# Convert all metric values to -- (1 - metric).
		# 	Lower (1 - metric) the better -- same as loss.
		metrics = {}
		for k in self.metrics_dict:
			metrics[k] = 1 - np.array( self.metrics_dict )

		data = np.stack( 
				[
					self.loss_dict["violation"],
					self.loss_dict["chain_center_of_mass"],
					self.loss_dict["xlr"],
					metrics
				],
				axis = 1
		)

		return data


	def get_scaled_data( self, data: np.array ) -> np.array:
		"""
		Perform standard scaling.
		"""
		scaler = StandardScaler()
		scaled_data = scaler.fit_transform( data )

		return scaled_data


	def get_hdbscan_clusters( self, data: np.array ) -> Dict[int, List[int]]:
		"""
		Run HDBSCAN clustering the return dict for all clusters.
		"""
		hdb = hdbscan.HDBSCAN( min_cluster_size = self.min_cluster_size,
								min_samples = self.min_samples )
		hdb.fit( scaled_data )

		# Get cluster labels for each data point.
		labels = hdb.labels_

		hdb_clusters = {}
		for label in np.unique(labels):
			# Get indices for all data points having a label.
			hdb_clusters[label] = np.array( 
									np.where( labels == label )[0]
									)

		return hdb_clusters


	def model_filtering( self ) -> List:
		"""
		Remove outlier models from the given set of models.
		Use HDBSCAN to identify such noise models (unclustered).
			Consider the all except the unclustered models for further analysis.
			Use violation loss, CCOM loss, restraint loss, and 
				restraint metrics for clustering.
		"""
		data = self.prep_data()
		# The model at epoch 0 will be kept.
		scaled_data = self.get_scaled_data( data[1:, :] )

		clusters = self.hdb_clusters( scaled_data )

		# Add the epoch 0 model.
		selected_models = [0]
		# Add 1 to all cluster indices to correct for the 
		# 	numbering due to removing the epoch 0 model.
		for label in clusters:
			clusters[label] += 1

			if label == -1:
				unclustered_models = clusters[label]
			else:
				# Ignore the unclustered models.
				selected_models.extend( clusters[label] )

		selected_models = sorted( selected_models )
		return unclustered_models, selected_models


	def save_loss_metrics( self, model_ids: List[int], file_name: str ):
		"""
		Select the loss and metric values for the given model IDs and save on disk.
		"""
		df_dict = {"model_id": model_ids}
		for k in self.loss_dict:
			v = np.array( self.loss_dict[k] )
			df_dict[k] = v[model_ids]

		for k in self.metrics_dict:
			v = np.array( self.metrics_dict[k] )
			df_dict[f"{k}_metric"] = np.round( 
											np.mean( v[model_ids]), 
											self.prec
											)
		file_name = os.path.join( self.analysis_dir, f"summary_{file_name}.csv" )
		df.to_csv( file_name, index = False )


if __name__ == "__main__":
	Assay().forward()

