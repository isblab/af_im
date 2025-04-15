"""
Run the IntegrativeLearning module and get 
	XL satisfaction for all PDB IDs.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from openfold_wrapper import IntegrativeLearning

from topology import topology_dict
from utils.utils import write_json


class EyeDrop():
	def __init__( self ):
		self.benchmark = pd.read_csv( os.path.abspath( "./benchmark/benchmark.csv" ) )


	def forward( self ):
		"""
		Run IntegrativeLearning module for the given
			PDB IDs and check XL staisfaction for the 
			OpenFold predicted structure.
		"""
		sys_xl_satisfaction = {}
		curr_dir = os.getcwd()
		# problem -- 7xpc
		for sys_name in ["8ct8", "8gtj", "8i2f", "7qot", "8dwl",
						"7xad", "7y4a", "8phv", "8cxj", "8jwj",
						"8bs9", "8g9p", "8pfc", "8gt0",
						"7umb", "8cqz", "8hbe", "7ysp", "8u45",
						"8p98", "8gtm", "8bos", "8pn6", "8gtk",
						"8iy6", "8f2p", "8hbf", "7ytu", "8dc0",
						"8hbn", "8i5w", "8cyi", "8bzr", "8enf",
						"8h8a", "8brt", "7zt6", "7wr6", "7ymf",
						"7ui8", "8b06", "8gp6", "7rb3", "8ov4",
						"8efw", "7xvk", "8hhj", "8h0n", "8t3k",
						"7uj3", "8gtx", "8t7v", "8eoj", "8hi7",
						"8igc", "8ipl", "8b3s", "7wge", "8sah",
						"7wko", "7zjv", "8k5r", "8gt5", "8ain",
						"7zcm", "8i9q", "8tj3", "8ozc", "8j64",
						"8pfd", "7xky", "8ey4", "8bj8", "7xvo",
						"8odr", "8ey0", "8h5b", "8jyg", "8ezs",
						"8gxe", "7zch", "8oof"]:
			print( "\n------------------------------------------------------------" )
			print( "------------------------------------------------------------" )
			print( sys_name)
			print( "------------------------------------------------------------" )
			print( "------------------------------------------------------------\n" )
			# # Heteromers not run.
			# 			"8jyg", "8ezs", "8gxe", "7zch", "8oof",
			# 			"8p81", "8t1c", "8alk", "7wqu", "7yui",
			# 			"8skk", "8ba1", "8jmr", "8pwb", "8wtd",
			# 			"8alm", "8sg7"]
			sys_path = self.get_sys_path( sys_name )
			summary_file_path = self.summary_file_path( sys_path )

			if not os.path.exists( summary_file_path ):
				topo_dict = topology_dict()
				topo_dict.objective = f"{sys_name} with rmse_xlr. Just checking XL satisfaction."
				topo_dict.train.version = 0
				IntegrativeLearning( sys_name, topo_dict ).forward()
			else:
				print( f"Summary file already present for {sys_name}..." )

			epoch0_xlr_metric, viol0, ccom0 = self.get_epoch0_metrics( summary_file_path )

			sys_xl_satisfaction[sys_name] = [epoch0_xlr_metric, viol0, ccom0]

			os.chdir( curr_dir )

		self.save_sys_xl_satisfaction( sys_xl_satisfaction )


	def get_sys_path( self, sys_name: str ):
		sys_path = os.path.abspath( f"./benchmark/imp_dl_benchmark/{sys_name}/" )
		return sys_path


	def summary_file_path( self, sys_path: str ):
		summary_file_path = os.path.join( sys_path, f"test/version_0.0/Summary.csv" )
		return summary_file_path


	def get_epoch0_metrics( self, summary_file_path: str ):
		"""
		Parse the "Summary.csv" file in the output directory
			and get the xlr_metric value at epoch 0.
		"""
		df = pd.read_csv( summary_file_path )

		epoch0_xlr_metric = df.loc[0, "xlr_metric"]
		viol0 = df.loc[0, "violation"]
		ccom0 = df.loc[0, "chain_center_of_mass"]
		return epoch0_xlr_metric, viol0, ccom0


	def save_sys_xl_satisfaction( self, sys_xl_satisfaction ):
		"""
		Save the Xl satisfaction metric for all systems as a .csv file.
		"""
		flat_dict = {k:[] for k in ["pdb_id", "xlr_metric_epoch0", "viol0",
									"ccom0", "total_length",
									"Interprotein-XLs", "auth_asym_ids"]}

		for k, v in sys_xl_satisfaction.items():
			idx = self.benchmark.index[self.benchmark["pdb_id"] == k].tolist()
			aa_id = self.benchmark.loc[idx, "auth_asym_ids"].tolist()[0]
			length = self.benchmark.loc[idx, "total_length"].tolist()[0]
			xls = self.benchmark.loc[idx, "Interprotein-XLs"].tolist()[0]
			flat_dict["pdb_id"].append( k )
			flat_dict["xlr_metric_epoch0"].append( v[0] )
			flat_dict["viol0"].append( v[1] )
			flat_dict["ccom0"].append( v[2] )
			flat_dict["total_length"].append( length )
			flat_dict["Interprotein-XLs"].append( xls )
			flat_dict["auth_asym_ids"].append( aa_id )

		df = pd.DataFrame( flat_dict )
		df.to_csv( 
				os.path.join( "./benchmark/benchmark_xl_satisfaction.csv" ),
				index = False
				 )

		# Save only the required PDBs into another .csv file.
		# Remove entries with al XLs satisfied.
		selected_df = df[df["xlr_metric_epoch0"] < 1.0]
		selected_df = selected_df.reset_index( drop = True )
		selected_df.to_csv( "./benchmark/selected_benchmark_xl_satisfaction.csv", index = False )


if __name__ == "__main__":
	EyeDrop().forward()
