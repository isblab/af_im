"""
A wrapper script to obtain the initial data satisfcation for selected complexes.
Will consider complexes for which the data satisfaction is < a cutoff.
"""
from typing import Dict
import os, time, subprocess, traceback
from datetime import datetime
import ml_collections as mlc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from openfold.config import model_config

from openfold_wrapper import IntegrativeLearning
from loss import LossFunction
from metrics import Metrics
from topology import topology_dict
from utils.utils import ( open_file_handler,
						read_json,
						write_json )
from topology import topology_dict

from utils.paths import get_sys_data_dir_path
from utils.utils import parse_nested_dict

class InitPrediction():
	def __init__( self ):
		self.benchmark_name = "xlmerged"  # xlsim/abag/oreilly/xlmerged
		# Define the modeling objective.
		self.modeling_objective = f"({self.benchmark_name}) Obtaining initial prediction."
		self.modeling_dir_name = "xlmerged_modeling"
		self.modeling_version = 0
		self.device = "cuda:0"
		self.data_sat_cutoff = 1.0 if self.benchmark_name == "oreilly" else 0.75

		self.init_pred_metrics = {}
		self.logs = {}



	def forward( self ):
		"""
		"""
		print( f"Using device = {self.device}" )
		time.sleep( 1 )
		self.create_required_paths()
		self.initialize_logs_dict()
		self.load_benchamrk()
		self.predict_for_benchmark()
		self.filter_complexes()

		# Load PDB ID to benchmark mapping.
		self.pdb_benchmark_map = read_json( self.pdb_benchmark_map_file )
		self.write_data_for_selected_complexes()

		write_json( self.logs, self.logs_file )


	################################################################################
	################################################################################
	def initialize_logs_dict( self ):
		"""
		Log the following info:
			Time taken by each system.
			Memory consumed by each system.
		"""
		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )
		else:
			self.logs = {k:{} for k in ["time",
										"memory_allocated",
										"memory_reserved",
										"errored"]}


	def create_required_paths( self ):
		"""
		Create all the required paths.
		"""
		self.base_dir = os.path.join( os.path.abspath( "../benchmark/" ) )
		self.meta_dir = os.path.join( self.base_dir,
							f"{self.benchmark_name}_metadata" )

		self.benchmark_file = os.path.join( self.meta_dir,
						f"{self.benchmark_name}_benchmark.csv" )
		# Dict mapping PDB ID to its respective benchmark.
		self.pdb_benchmark_map_file = os.path.join( self.meta_dir, "pdb_benchmark_mapping.json" )

		self.init_pred_metrics_file = os.path.join( self.meta_dir, "init_pred_metrics.json" )

		self.selected_benchmark_file = os.path.join( self.meta_dir,
						f"selected_{self.benchmark_name}_benchmark.csv" )
		self.logs_file = os.path.join( self.meta_dir,
						f"Logs_init_model_{self.benchmark_name}.json" )


	################################################################################
	################################################################################
	def load_benchamrk( self ):
		"""
		Load the benchmak .csv file.
		Note the total no. of systems to be modeled.
		"""
		self.benchmark = pd.read_csv( self.benchmark_file )
		self.num_systems = self.benchmark.shape[0]


	def get_sys_path( self, sys_name: str ):
		sys_path = os.path.join( 
					os.path.abspath(
						f"{self.base_dir}/{self.modeling_dir_name}/{sys_name}"
						)
			)
		return sys_path


	################################################################################
	################################################################################
	def log_error( self, sys_name: str ):
		"""
		If an error occurs while modeling,
			Log the errorneous entry_id in the dataset sepcific metadata dir.
			log the traceback in the system dir.
		"""
		self.logs["errored"][sys_name] = None

		sys_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		current_datetime = datetime.now()
		timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
		error_file = os.path.join( sys_dir, f"error_init_pred_{timestamp}.txt" )
		w = open_file_handler( error_file, "w" )
		w.write( traceback.format_exc() )
		w.close()

		print( f"An error occured for system: {sys_name}. " +
				f"Check error log in {error_file}...\n" )


	def log_memory_usage( self, sys_name: str ):
		"""
		Log the device memory used during modeling.
		device must be in the following formar: cuda[0]
		Reset the CUDA memory stats after logging.
		"""
		if self.device == "cpu":
			raise ValueError( "Pytorch does not provide " +
					"built-in functions to check CPU memory stats. " +
					"Change device to cuda[0/1]..." )
		else:
			device_num = int( self.device[-1] )
		# Get peak memory since last reset.
		max_allocated = torch.cuda.max_memory_reserved( device_num )
		max_reserved = torch.cuda.max_memory_reserved( device_num )
		self.logs["memory_allocated"][sys_name] = max_allocated
		self.logs["memory_reserved"][sys_name] = max_reserved


	def predict_for_benchmark( self ):
		"""
		Get initial prediction for all complexes in the benchmark.
		Obtain the violation loss and XL satisfaction.
		Prediction failed for the following due to H-chain mapped to Titin.
			["1kcs", "1f58", "2b1h", "5dmi", "1uj3",
			"4i3r", "2qhr", "6aq7", "1osp", "3ujj",
			"3sge", "4m1d", "5dmi", "5u3j", "6db7",
			"6u6u", "6jep", "6q18", "7n4j", "7tp3",
			"8x0t", "8fdo", "6xq0", "8yor"]
		5e8e_B has a non-standard aa PCA.
		7xpc contains non-standard aa MSE.
		"""
		if os.path.exists( self.init_pred_metrics_file ):
			self.init_pred_metrics = read_json( self.init_pred_metrics_file )
		else:
			for sys_name in self.benchmark["PDB ID"]:
				print( f"\n----------- \033[1m{sys_name}\033[0m" )
				if sys_name in self.logs["errored"]:
					print( f"{sys_name} errored in a previous run..." )
					continue
				try:
					tic = time.perf_counter()
					violation, xl_metric = self.run_per_system_prediction( sys_name = sys_name )
					self.init_pred_metrics[sys_name] = {
						"violation": violation.item(),
						"xl_satisfaction": xl_metric.item() }
					toc = time.perf_counter()
					if not sys_name in self.logs["time"]:
						self.logs["time"][sys_name] = toc-tic
						write_json( self.logs, self.logs_file )
					write_json( self.init_pred_metrics, self.init_pred_metrics_file )
				except:
					self.log_error( sys_name = sys_name )

				torch.cuda.empty_cache()		


	def run_per_system_prediction( self, sys_name: str ):
		"""
		Run OpenFold prediction for a given complex (system).
		Initialize the IntergrativeLearning class.
		Obtain an initial prediction.
		Compute the violation loss and XL metric.
		"""
		sys_conf_suff = "_tpfp"
		# Initialize the topology dict.
		topo_dict = topology_dict()

		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				sys_config_file =  f"sys_config_{sys_name}{sys_conf_suff}.json",
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.create_required_paths_dirs()

		print( "\n" + "-"*70 + "\n" +"-"*27 +
			" \033[1mData gathering\033[0m " +
			"-"*27 + "\n" + "-"*70 + "\n" )
		restraint_features = il_obj.run_data_gathering()

		print( "\n" + "-"*70 + "\n" +"-"*24 +
			" \033[1mSystem representation\033[0m " +
			"-"*23 + "\n" + "-"*70 + "\n" )
		( _, _, gt_feature_dict, init_pred_dict ) = il_obj.run_system_representation()

		self.log_memory_usage( sys_name = sys_name )

		topo_dict.loss.violation.enabled = True
		topo_dict.loss.violation.add_penalty = True
		topo_dict.loss.xlr.enabled = False
		topo_dict.loss.xlr.add_penalty = False

		print( "\nComputing violation loss and data satisfaction..." )
		# Convert all to torch tensor.
		gt_feature_dict = parse_nested_dict( gt_feature_dict, "to_tensor" )
		init_pred_dict = parse_nested_dict( init_pred_dict, "to_tensor" )
		gt_feature_dict["residue_index"] = gt_feature_dict["residue_index"].to( torch.int64 )
		# Add batch dim.
		gt_feature_dict = parse_nested_dict( gt_feature_dict, "add_dim" )
		init_pred_dict = parse_nested_dict( init_pred_dict, "add_dim" )

		loss_func = LossFunction( topo_dict["loss"], self.device )
		_, losses = loss_func.forward( init_pred_dict,
			gt_feature_dict,
			restraint_features )
		metrics_func = Metrics( topo_dict["metrics"], restraint_features )
		metrics_dict = metrics_func.forward(
			out = init_pred_dict, last_epoch = False  )

		violation = losses["violation"]
		xl_metric = metrics_dict["xlr"]
		print( f"{sys_name} -- Violation = {violation} \t XL_metric = {xl_metric}" )
		return violation, xl_metric


	def filter_complexes( self ):
		"""
		Remove complexes for which data satisfcation is greater than the specified cutoff.
		"""
		all_keys = list( self.init_pred_metrics.keys() )
		for sys_name in all_keys:
			if self.init_pred_metrics[sys_name]["xl_satisfaction"] > self.data_sat_cutoff:
				self.init_pred_metrics.pop( sys_name )
		print( f"Selected complexes = {len( self.init_pred_metrics )}\n" )


	################################################################################
	################################################################################
	def write_data_for_selected_complexes( self ):
		"""
		Fetch the required details for all complexes (system)
			in the benchmark and save on disk.
		"""
		benchmark_dict = {}
		for sys_name in self.init_pred_metrics:
			sys_dict = self.get_system_data( sys_name = sys_name )
			for k in sys_dict:
				if k in benchmark_dict:
					benchmark_dict[k].append( sys_dict[k] )
				else:
					benchmark_dict[k] = [sys_dict[k]]
		df = pd.DataFrame( benchmark_dict )
		df.to_csv( self.selected_benchmark_file, index = False )


	def get_system_data( self, sys_name: str ) -> Dict:
		"""
		Get the following info for all complexes:
			PDB ID (sys_name)
			Violations in the initial structure.
			XL sstisfaction for initial structure.
			Total length of the system.
			No. of TP Inter-protein XLs.
			No. of FP Inter-protein XLs.
			Auth asym IDs.
		"""
		idx = self.benchmark.index[self.benchmark["PDB ID"] == sys_name].tolist()
		sys_dict = {
			"PDB ID": sys_name,
			"violation": self.init_pred_metrics[sys_name]["violation"],
			"xl_satisfaction": self.init_pred_metrics[sys_name]["xl_satisfaction"],
			"Total length": self.benchmark.loc[idx, "Total length"].tolist()[0],
			"TP XLs": self.benchmark.loc[idx, "Selected TP XLs"].tolist()[0],
			"FP XLs": self.benchmark.loc[idx, "Selected FP XLs"].tolist()[0],
			"Auth Asym ID": self.benchmark.loc[idx, "Auth Asym ID"].tolist()[0],
			"benchmark": self.pdb_benchmark_map[sys_name]
		}

		return sys_dict

	# def log_memory_usage( self, sys_name: str ):
	# 	"""
	# 	Log the device memory used during modeling.
	# 	device must be in the following formar: cuda[0]
	# 	Reset the CUDA memory stats after logging.
	# 	"""
	# 	if self.device == "cpu":
	# 		raise valueError( "Pytorch does not provide " +
	# 				"built-in functions to check CPU memory stats. " +
	# 				"Change device to cuda[0/1]..." )
	# 	else:
	# 		device_num = int( self.device[-1] )
	# 	# Get peak memory since last reset.
	# 	max_allocated = torch.cuda.max_memory_reserved( device_num )
	# 	max_reserved = torch.cuda.max_memory_reserved( device_num )
	# 	self.logs["memory_allocated"][sys_name] = max_allocated
	# 	self.logs["memory_reserved"][sys_name] = max_reserved


	# def modify_topology( self, sys_name: str ):
	# 	"""
	# 	Add modeling objective and version to topology for each system.
	# 	"""
	# 	topo_dict = topology_dict()
	# 	topo_dict.objective = f"{sys_name} {self.modeling_objective}"
	# 	topo_dict.train.version = self.modeling_version
	# 	topo_dict.train.device = self.device

	# 	# if self.benchmark_name == "sabdab":
	# 	# 	topo_dict.db_preset = "reduced_dbs"

	# 	return topo_dict


	# ## ------------------------------------------------------ ##
	# ## ------------------------------------------------------ ##
	# def run_modeling_for_benchmark( self ):
	# 	"""
	# 	Run the Integrative modeling pipeline for the entire benchmark.
	# 	"""
	# 	print( "\033[1mRunning modeling for the benchmark...\033[0m" )
	# 	curr_dir = os.getcwd()
	# 	for idx, sys_name in enumerate( self.benchmark["PDB ID"] ):
	# 		print( "\n" + "-"*70 + "\n" + "-"*70 )
	# 		print( "-"*25 + f" {idx}/{self.num_systems} --> {sys_name} " + "-"*25 )
	# 		print( "\n" + "-"*70 + "\n" + "-"*70 )

	# 		# 5e8e_B has a non-standard aa PCA.
	# 		# All these failed due to H-chain mapped to Titin.
	# 		if sys_name in self.logs["errored"]:
	# 			continue
	# 		# if sys_name in ["1kcs", "1f58", "2b1h", "5dmi", "1uj3",
	# 		# 				"4i3r", "2qhr", "6aq7", "1osp", "3ujj",
	# 		# 				"3sge", "4m1d", "5dmi", "5u3j", "6db7",
	# 		# 				"6u6u", "6jep", "6q18", "7n4j", "7tp3",
	# 		# 				"8x0t", "8fdo", "6xq0", "8yor"]:
	# 		#             print( f"Skipping PDB ID: {sys_name}...\n" )

	# 	    # 7xpc -> contains non-standard aa MSE.
	# 	    # if sys_name in ["7xvk", "8st9"]:
	# 	    #         continue

	# 		sys_path = self.get_sys_path( sys_name )
	# 		stat_file_path = self.get_stat_file_path( sys_name )

	# 		if not os.path.exists( stat_file_path ):
	# 			tic = time.time()
	# 			topo_dict = self.modify_topology( sys_name = sys_name )
	# 			device = topo_dict.train.device

	# 			# torch.cuda.reset_peak_memory_stats( device = torch.device( device ) )
	# 			self.run_modeling_for_system( sys_name = sys_name,
	# 											topo_dict = topo_dict )
	# 			toc = time.time()
	# 			self.logs["time"][sys_name] = toc-tic
	# 			self.log_memory_usage( sys_name = sys_name )

	# 			write_json( self.logs, self.logs_file )
	# 		else:
	# 			print( f"Summary file already present for {sys_name}..." )

	# 		# Return to base_dir.
	# 		os.chdir( curr_dir )
	# 		fasta_path = f"./benchmark/{sys_name}_fasta/"
	# 		if os.path.exists( fasta_path ):
	# 			run_subprocess( ["rm", f"{fasta_path}"] )


	# def log_error( self, sys_name: str ):
	# 	"""
	# 	If an error occurs while modeling,
	# 		Log the errorneous entry_id in the dataset sepcific metadata dir.
	# 		log the traceback in the system dir.
	# 	"""
	# 	self.logs["errored"][sys_name] = None

	# 	ver_path = self.get_modeling_version_path( sys_name )
	# 	current_datetime = datetime.now()
	# 	timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
	# 	error_file = os.path.join( ver_path, f"error_init_pred_{timestamp}.txt" )
	# 	w = open_file_handler( error_file, "w" )
	# 	w.write( traceback.format_exc() )
	# 	w.close()

	# 	print( f"An error occured for system: {sys_name}. " +
	# 			f"Check error log in {error_file}...\n" )



	# def run_modeling_for_system( self, sys_name: str, topo_dict: mlc.ConfigDict ):
	# 	"""
	# 	Run the Integrative modeling pipeline for a give system.
	# 	If summary file exists fo a run, do not run again.
	# 	If an error occurs, log it to the modeling dir.
	# 	Empty the CUDA cache after each run.
	# 	"""

	# 	# Directory containing input data for the modeled system.
	# 	data_dir = os.path.join( self.base_dir,
	# 							f"{self.benchmark_name}_benchmark/{sys_name}/" )
	# 	# fasta_path = "./benchmark/"
	# 	try:
	# 		il_obj = IntegrativeLearning( 
	# 				sys_name = sys_name,
	# 				base_dir = self.base_dir,
	# 				data_dir = data_dir,
	# 				# fasta_path = fasta_path,
	# 				sys_config_file =  f"sys_config_{sys_name}.json",
	# 				modeling_dir_name = self.modeling_dir_name,
	# 				topology_dict = topo_dict,
	# 				)
	# 		il_obj.disable_overwrite_warning = True
	# 		il_obj.forward()
	# 	except:
	# 		self.log_error( sys_name = sys_name )

	# 	torch.cuda.empty_cache()


	# ## ------------------------------------------------------ ##
	# ## ------------------------------------------------------ ##
	# def get_summary_dict( self ):
	# 	"""
	# 	Return an empty summary dict containing all required keys.
	# 	"""
	# 	empty_dict = {k:[] for k in ["PDB ID", "XLR metric 0", "XLR metric last",
	# 								"Viol0", "CCOM0", "Total length",
	# 								"TP XLs", "FP XLs", "Auth Asym ID"]}
	# 	return empty_dict


	# def filter_benchmark( self ):
	# 	"""
	# 	Remove complexes that have >cutoff data satisfaction for
	# 		the initial predicted structure.
	# 	"""
	# 	print( "\n\n" + "-"*70 )
	# 	print( "\033[1mFiltering complexes based on data satisfaction...\033[0m" )
	# 	print( "-"*70 + "\n" )

	# 	# summary_dict = self.get_summary_dict()
	# 	summary_dict = {k:[] for k in ["PDB ID", "XLR metric 0", "XLR metric last",
	# 								"Viol0", "CCOM0", "Total length",
	# 								"TP XLs", "FP XLs", "Auth Asym ID"]}

	# 	for sys_name in self.benchmark["PDB ID"]:

	# 		if sys_name in self.logs["errored"]:
	# 			continue
	# 		sys_dict = self.get_system_data( sys_name = sys_name )
	# 		if sys_dict is None:
	# 			continue

	# 		for k in summary_dict:
	# 			summary_dict[k].append( sys_dict[k] )

	# 	df = pd.DataFrame( summary_dict )
	# 	df.to_csv( self.selected_benchmark_file, index = False )



	# def get_system_data( self, sys_name: str ) -> Dict:
	# 	"""
	# 	Get the following info for all complexes:
	# 		PDB ID (sys_name)
	# 		XL sstisfaction at epoch 0.
	# 		XL sstisfaction at last 0.
	# 		Total length of the system.
	# 		No. of TP Inter-protein XLs.
	# 		No. of FP Inter-protein XLs.
	# 		Auth asym IDs.
	# 	"""
	# 	stats_dict = self.load_stat_file( sys_name )

	# 	epoch0_xlr = stats_dict["metrics"]["xlr"][0]
	# 	last_epoch_xlr = stats_dict["metrics"]["xlr"][-1]

	# 	if epoch0_xlr > self.data_sat_cutoff:
	# 		return None
	# 	else:
	# 		idx = self.benchmark.index[self.benchmark["PDB ID"] == sys_name].tolist()
	# 		sys_dict = {
	# 			"PDB ID": sys_name,
	# 			"XLR metric 0": epoch0_xlr,
	# 			"XLR metric last": last_epoch_xlr,
	# 			"Viol0": stats_dict["loss"]["violation"][0],
	# 			"CCOM0": stats_dict["loss"]["chain_center_of_mass"][0],
	# 			"Total length": self.benchmark.loc[idx, "Total length"].tolist()[0],
	# 			"TP XLs": self.benchmark.loc[idx, "Selected TP XLs"].tolist()[0],
	# 			"FP XLs": self.benchmark.loc[idx, "Selected FP XLs"].tolist()[0],
	# 			"Auth Asym ID": self.benchmark.loc[idx, "Auth Asym ID"].tolist()[0]
	# 		}

	# 		return sys_dict


if __name__ == "__main__":
	InitPrediction().forward()


