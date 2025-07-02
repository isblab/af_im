"""
The following module runs the IntegrativeLearning pipeline
	to obtain initial predictions for the specified benchmark.
Filters out complexes that:
	Have >cutoff data satisfaction.
"""
from typing import Dict
import os, glob, time, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.spatial import distance_matrix
from openfold_wrapper import IntegrativeLearning

from topology import topology_dict
from utils.utils import ( open_file_handler )


class InitPrediction():
	def __init__( self ):
		self.benchmark_name = "afu"
		# Define the modeling objective.
		self.modeling_objective = "Obtaining initial prediction."
		self.modeling_version = 0

		self.base_dir = os.path.join( "./benchmark/" )
		# Name for the dir to store modeling output for all systems.
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"

		self.benchmark = pd.read_csv( os.path.abspath(
							f"./benchmark/{self.benchmark_name}_benchmark.csv" )
							)

		# self.xl_benchmark = ["8gtj", '8i2f', "7qot", "8dwl","7xad",
		# 					"8phv", "8cxj", "8g9p", "8pfc", "8gt0",
		# 					"8gtk", "8bzr", "8h8a", "7wr6", "7ymf",
		# 					"7xvk", "8b3s", "7xvo", "8odr", "8gxe",
		# 					"8t1c", "8wtd"]
		self.num_systems = self.benchmark.shape[0]

		self.selected_benchmark_file = os.path.join(
										self.base_dir,
										f"selected_{self.benchmark_name}_benchmark.csv"
										)


	def forward( self ):
		"""
		"""
		# self.create_dirs()
		self.run_modeling_for_benchmark()

		self.filter_benchmark()


	# def create_dirs( self ):
	# 	"""
	# 	Craete the required directories.
	# 	"""
	# 	os.makedirs( self.modeling_dir, exist_ok = True )


	def get_sys_path( self, sys_name: str ):
		# sys_path = os.path.abspath( f"./benchmark/modeling/{sys_name}/" )
		sys_path = os.path.join( 
					os.path.abspath( f"{self.base_dir}/{self.modeling_dir_name}/{sys_name}" )
			)
		return sys_path


	def get_stat_file_path( self, sys_path: str ) -> pd.DataFrame:
		"""
		Return the path to the stats file for the given system.
		"""
		stat_file_path = os.path.join( sys_path, f"version_{self.modeling_version}/Stats.npy" )
		return stat_file_path


	def load_stat_file( self, sys_path: str ) -> Dict[str, Dict]:
		"""
		Load the stat file on memory.
		"""
		stat_file_path = self.get_stat_file_path( sys_path )
		stats_dict = np.load( stat_file_path, allow_pickle = True ).item()
		return stats_dict


	def run_modeling_for_benchmark( self ):
		"""
		Run the Integrative modeling pipeline for the entire benchmark.
		"""
		print( "\033[1mRunning modeling for the benchmark...\033[0m" )
		curr_dir = os.getcwd()
		for idx, sys_name in enumerate( self.benchmark["pdb_id"] ):
			print( "\n------------------------------------------------------------" )
			print( "------------------------------------------------------------" )
			print( f"{idx}/{self.num_systems} --> {sys_name}" )
			print( "------------------------------------------------------------" )
			print( "------------------------------------------------------------\n" )
			sys_path = self.get_sys_path( sys_name )
			stat_file_path = self.get_stat_file_path( sys_path )

			if not os.path.exists( stat_file_path ):
				self.run_modeling_for_system( sys_name )
			else:
				print( f"Summary file already present for {sys_name}..." )

			# Return to base_dir.
			os.chdir( curr_dir )


	def run_modeling_for_system( self, sys_name: str ):
		"""
		Run the Integrative modeling pipeline for a give system.
		If summary file exists fo a run, do not run again.
		Empty the CUDA cache after each run.
		"""

		# Directory containing input data for the modeled system.
		data_dir = os.path.join( self.base_dir,
								f"{self.benchmark_name}_benchmark/{sys_name}/" )
		# fasta_path = "./benchmark/"

		topo_dict = topology_dict()
		topo_dict.objective = f"{sys_name} {self.modeling_objective}"
		topo_dict.train.version = self.modeling_version

		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				# fasta_path = fasta_path,
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.disable_overwrite_warning = True
		il_obj.forward()
		torch.cuda.empty_cache()


	def get_summary_dict( self ):
		"""
		Return an empty summary dict containing all required keys.
		"""
		empty_dict = {k:[] for k in ["pdb_id", "xlr_metric_0", "xlr_metric_last",
									"viol0", "viol_last", "ccom0", "ccom_last", "total_length",
									"Interprotein-XLs", "auth_asym_ids"]}
		return empty_dict


	def filter_benchmark( self ):
		"""
		Remove complexes that have >cutoff data satisfaction for
			the initial predicted structure.
		"""
		print( "\033[1mFiltering complexes based on data satisfaction...\033[0m" )

		summary_dict = self.get_summary_dict()

		for sys_name in self.benchmark["pdb_id"]:
			sys_dict = self.get_system_data( sys_name = sys_name )

			if sys_dict is None:
				continue

			for k in summary_dict:
				summary_dict[k].append( sys_dict[k] )

		df = pd.DataFrame( summary_dict )
		df.to_csv( self.selected_benchmark_file, index = False )



	def get_system_data( self, sys_name: str ):
		"""
		Get the following info for all complexes:
			PDB ID (sys_name)
			XL sstisfaction at epoch 0.
			XL sstisfaction at last 0.
			Total length of the system.
			No. of Inter-protein XLs.
			Auth asym IDs.
		"""
		sys_path = self.get_summary_file_path( sys_name )
		stats_dict = self.load_stat_file( sys_path )

		epoch0_xlr = stats_dict["metric"]["xlr"][0]
		last_epoch_xlr = stats_dict["metric"]["xlr"][-1]

		if epoch0_xlr > 0.75:
			return None
		else:
			sys_dict = self.get_summary_dict()
			sys_dict["pdb_id"] = sys_name

			sys_dict["xlr_metric_0"] = epoch0_xlr
			sys_dict["xlr_metric_0"] = last_epoch_xlr

			idx = self.benchmark.index[self.benchmark["pdb_id"] == k].tolist()
			aa_id = self.benchmark.loc[idx, "auth_asym_ids"].tolist()[0]
			length = self.benchmark.loc[idx, "total_length"].tolist()[0]
			xls = self.benchmark.loc[idx, "Interprotein-XLs"].tolist()[0]

			sys_dict["total_length"] = length
			sys_dict["Interprotein-XLs"] = xls
			sys_dict["auth_asym_ids"] = aa_id

			return sys_dict


if __name__ == "__main__":
	InitPrediction().forward()
