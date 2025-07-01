"""
Run the IntegrativeLearning module on the entire benchmark.
Check the distribution of violation loss, ccom loss, xl_restraint.
The configs for the simulation must be specified in the topology file.
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


class ModelingBenchmark():
	def __init__( self ):
		# Define the modeling objective.
		self.modeling_objective = "Not using FAPE and Supervised_chi loss."
		self.modeling_version = 0

		self.base_dir = os.path.join( "./benchmark/" )
		# Name for the dir to store modeling output for all systems.
		self.modeling_dir_name = "modeling"
		# Output dir for storing each benchmark run results.
		self.benchmark_output_dir = os.path.join( self.base_dir, "benchmark_results/" )
		self.modeling_version_dir = os.path.join( self.benchmark_output_dir,
												f"version_{self.modeling_version}/" )

		# self.benchmark = pd.read_csv( os.path.abspath( "./benchmark/benchmark.csv" ) )

		self.xl_benchmark = ["8gtj", '8i2f', "7qot", "8dwl","7xad",
							"8phv", "8cxj", "8g9p", "8pfc", "8gt0",
							"8gtk", "8bzr", "8h8a", "7wr6", "7ymf",
							"7xvk", "8b3s", "7xvo", "8odr", "8gxe",
							"8t1c", "8wtd"]
		self.num_systems = len( self.xl_benchmark )

		# File to write system name, data and time taken.
		self.misc_file = os.path.join( self.modeling_version_dir, "Time_taken.txt" )



	def forward( self ):
		"""
		"""
		tic = time.time()
		self.create_dirs()
		self.run_modeling_for_benchmark()
		self.plot_modeling_results()
		toc = time.time()

		self.write_misc_details( toc-tic )


	def create_dirs( self ):
		"""
		Craete the required directories.
		"""
		os.makedirs( self.benchmark_output_dir, exist_ok = True )
		os.makedirs( self.modeling_version_dir, exist_ok = True )


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
		for idx, sys_name in enumerate( self.xl_benchmark ):
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
								f"imp_dl_benchmark/{sys_name}/" )

		topo_dict = topology_dict()
		topo_dict.objective = f"{sys_name} {self.modeling_objective}"
		topo_dict.train.version = self.modeling_version

		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.disable_overwrite_warning = True
		il_obj.forward()
		torch.cuda.empty_cache()



	def plot_modeling_results( self ):
		"""
		For violation loss, ccom loss, and XL satisfaction,
			Plot distribution of per epoch values for each system.
			Plot distribution of avg values across systems.
		"""
		print( "Creating plots..." )
		self.plot_per_epoch_distribution()
		self.plot_avg_distribution()


	def plot_per_epoch_distribution( self ):
		"""
		Plot the distribution of per epoch values for
			 loss, ccom loss, and xl satisfaction for each systems.
		 Create separate plots for each term.
		"""
		plt.rcParams["font.family"] = "sans"
		for qty in ["violation", "chain_center_of_mass", "xlr"]:
			sys_data= []
			categories = []
			_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )
			for sys_name in self.xl_benchmark:
				categories.append( sys_name )

				sys_path = self.get_sys_path( sys_name )
				stats_dict = self.load_stat_file( sys_path )

				if qty == "xlr":
					# metrics_dict = self.get_metrics_dict( sys_path )
					metrics_dict = stats_dict["metrics"]
					sys_data.append( metrics_dict[qty] )
				else:
					# loss_dict = self.get_loss_dict( sys_path )
					loss_dict = stats_dict["loss"]
					sys_data.append( loss_dict[qty] )

			ax.violinplot( sys_data )
			ax.tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )

			plt.xticks( np.arange(1, len( categories ) + 1 ), categories )

			path = os.path.join( self.modeling_version_dir, f"{qty}_per_epoch.png" )
			plt.savefig( path, dpi = 300 )
			plt.close()


	def plot_avg_distribution( self ):
		"""
		Plot the distribution of avg values across modeling run for,
			 loss, ccom loss, and xl satisfaction for each systems.
		 Create separate plots for each term.
		"""
		plt.rcParams["font.family"] = "sans"
		for qty in ["violation", "chain_center_of_mass", "xlr"]:
			init_sys_data = []
			avg_sys_data = []
			categories = ["Initial", "Average"]
			_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )
			for sys_name in self.xl_benchmark:
				sys_path = self.get_sys_path( sys_name )
				stats_dict = self.load_stat_file( sys_path )

				if qty == "xlr":
					# metrics_dict = self.get_metrics_dict( sys_path )
					metrics_dict = stats_dict["metrics"]
					init_sys_data.append( np.round( metrics_dict[qty][0], 3 ) )
					avg_sys_data.append( np.round( np.mean( metrics_dict[qty] ), 3 ) )
				else:
					# loss_dict = self.get_loss_dict( sys_path )
					loss_dict = stats_dict["loss"]
					init_sys_data.append( np.round( loss_dict[qty][0], 3 ) )
					avg_sys_data.append( np.round( np.mean( loss_dict[qty] ), 3 ) )

			ax.violinplot( [init_sys_data, avg_sys_data] )
			ax.tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )

			plt.xticks( np.arange(1, len( categories ) + 1 ), categories )

			path = os.path.join( self.modeling_version_dir, f"{qty}_avg.png" )
			plt.savefig( path, dpi = 300 )
			plt.close()


	def write_misc_details( self, time_taken: float ):
		"""
		Write down the system used, date, and objective of the simulation.
		Assumes time is provided in seconds
		"""
		w = open_file_handler( self.misc_file, "w" )
		with subprocess.Popen( "hostname", shell = True, stdout = subprocess.PIPE ) as proc:
			system = proc.communicate()[0]
		with subprocess.Popen( "date", shell = True, stdout = subprocess.PIPE ) as proc:
			sys_date = proc.communicate()[0]
		w.writelines( f"System = {system} \t Date = {sys_date}\n" )
		w.writelines( f"{time_taken/60} minutes OR {time_taken/3600} hours." )
		w.close()



if __name__ == "__main__":
	ModelingBenchmark().forward()


