"""
The following module runs the IntegrativeLearning pipeline
	to obtain initial predictions for the specified benchmark.
Filters out complexes that:
	Have >cutoff data satisfaction.
"""
from typing import Dict
import os, glob, time, subprocess, time
import ml_collections as mlc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from openfold_wrapper import IntegrativeLearning

from topology import topology_dict
from utils.utils import ( open_file_handler,
						read_json,
						write_json,
						run_subprocess )


class InitPrediction():
	def __init__( self ):
		self.benchmark_name = "afu"  # afu/ sabdab
		# Define the modeling objective.
		self.modeling_objective = f"({self.benchmark_name}) Obtaining initial prediction."
		self.modeling_version = 0

		self.logs = {}



	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.initialize_logs_dict()
		self.load_benchamrk()
		self.run_modeling_for_benchmark()

		self.filter_benchmark()

		write_json( self.logs, self.logs_file )


	## ------------------------------------------------------ ##
	## ------------------------------------------------------ ##
	def initialize_logs_dict( self ):
		"""
		Log the following info:
			Time taken by each system.
			Memory consumed by each system.
		"""
		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )
		else:
			self.logs = {k:{} for k in ["time", "memory"]}


	def create_required_paths( self ):
		"""
		Create all the required paths.
		"""
		self.base_dir = os.path.join( os.path.abspath( "./benchmark/" ) )
		self.meta_dir = os.path.join( self.base_dir,
							f"{self.benchmark_name}_metadata" )
		# Name for the dir to store modeling output for all systems.
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"

		self.benchmark_file = os.path.join( self.meta_dir,
						f"{self.benchmark_name}_benchmark.csv" )

		self.selected_benchmark_file = os.path.join( self.meta_dir,
						f"selected_{self.benchmark_name}_benchmark.csv" )
		self.logs_file = os.path.join( self.meta_dir,
						f"Logs_init_model_{self.benchmark_name}.json" )


	## ------------------------------------------------------ ##
	## ------------------------------------------------------ ##
	def load_benchamrk( self ):
		"""
		Load the benchmak .csv file.
		Note the total no. of systems to be modeled.
		"""
		self.benchmark = pd.read_csv( self.benchmark_file )
		self.num_systems = self.benchmark.shape[0]


	def get_sys_path( self, sys_name: str ):
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


	## ------------------------------------------------------ ##
	## ------------------------------------------------------ ##
	def log_memory_usage( self, sys_name: str, device: str ):
		"""
		Log the device memory used during modeling.
		device must be in the following formar: cuda[0]
		Reset the CUDA memory stats after logging.
		"""
		if device == "cpu":
			raise valueError( "Pytorch does not provide " +
					"built-in functions to check CPU memory stats. " +
					"Change device to cuda[0/1]..." )
		else:
			device_num = int( device[-1] )
		# get peak memory since last reset.
		max_allocated = torch.cuda.max_memory_reserved( device_num )
		max_reserved = torch.cuda.max_memory_reserved( device_num )
		self.logs["memory_allocated"][sys_name] = max_allocated
		self.logs["memory_reserved"][sys_name] = max_reserved

		torch.cuda.reset_peak_memory_stats( device_num )


	def modify_topology( self, sys_name: str ):
		"""
		Add modeling objective and version to topology for each system.
		"""
		topo_dict = topology_dict()
		topo_dict.objective = f"{sys_name} {self.modeling_objective}"
		topo_dict.train.version = self.modeling_version

		return topo_dict


	## ------------------------------------------------------ ##
	## ------------------------------------------------------ ##
	def run_modeling_for_benchmark( self ):
		"""
		Run the Integrative modeling pipeline for the entire benchmark.
		"""
		print( "\033[1mRunning modeling for the benchmark...\033[0m" )
		curr_dir = os.getcwd()
		for idx, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( "\n" + "-"*70 + "\n" + "-"*70 )
			print( "-"*25 + f" {idx}/{self.num_systems} --> {sys_name} " + "-"*25 )
			print( "\n" + "-"*70 + "\n" + "-"*70 )

			# 7xpc -> Error in res_idx_map[chain1]
			# if sys_name in ["7ui8", "7qot"]:  # "7xpc"
				# continue
			sys_path = self.get_sys_path( sys_name )
			stat_file_path = self.get_stat_file_path( sys_path )

			if not os.path.exists( stat_file_path ):
				tic = time.time()
				topo_dict = self.modify_topology( sys_name = sys_name )
				device = topo_dict.train.device
				self.run_modeling_for_system( sys_name = sys_name,
												topo_dict = topo_dict )
				toc = time.time()
				self.logs["time"][sys_name] = toc-tic
				self.log_memory_usage()

			else:
				print( f"Summary file already present for {sys_name}..." )

			write_json( self.logs, self.logs_file )

			# Return to base_dir.
			os.chdir( curr_dir )
			fasta_path = f"./benchmark/{sys_name}_fasta/"
			if os.path.exists( fasta_path ):
				run_subprocess( ["rm", f"{fasta_path}"] )


	def run_modeling_for_system( self, sys_name: str, topo_dict: mlc.ConfigDict ):
		"""
		Run the Integrative modeling pipeline for a give system.
		If summary file exists fo a run, do not run again.
		Empty the CUDA cache after each run.
		"""

		# Directory containing input data for the modeled system.
		data_dir = os.path.join( self.base_dir,
								f"{self.benchmark_name}_benchmark/{sys_name}/" )
		# fasta_path = "./benchmark/"



		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				# fasta_path = fasta_path,
				sys_config_file =  f"sys_config_tp_{sys_name}.json",
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.disable_overwrite_warning = True
		il_obj.forward()
		torch.cuda.empty_cache()


	## ------------------------------------------------------ ##
	## ------------------------------------------------------ ##
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



	def get_system_data( self, sys_name: str ) -> Dict:
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
