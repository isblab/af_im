"""
A wrapper script to obtain the initial data satisfcation for selected complexes.
Will consider complexes for which the data satisfaction is <= a cutoff.
"""
from typing import Tuple, Dict
import os, sys, time, traceback, argparse
import signal, pickle as pkl
from datetime import datetime
import numpy as np
import pandas as pd
from multiprocessing import Pool
import tqdm
import torch

from openfold_wrapper import IntegrativeLearning
from system_representation import SystemRepresentation
from loss import LossFunction
from metrics import Metrics
from topology import topology_dict
from utils.utils import ( open_file_handler,
						read_json,
						write_json )
from topology import topology_dict

from utils.paths import get_sys_data_dir_path, get_init_pred_file
from utils.utils import parse_nested_dict

# "8a67", "8evd", "8ij9", "8oij" --> Removed jus for testing.

PROBLEMATIC = [
	# Prediction failed for the following due to H-chain mapped to Titin.
	# This is not an exhaustive list; just some entries that I found.
	"1kcs", "1f58", "2b1h", "5dmi", "1uj3",
	"4i3r", "2qhr", "6aq7", "1osp", "3ujj",
	"3sge", "4m1d", "5u3j", "6db7", "6u6u",
	"6jep", "6q18", "7n4j", "7tp3", "8x0t",
	"8fdo", "6xq0", "8yor", "3qa3", "7yds",
	"5e8e",    # 5e8e_B has a non-standard aa PCA.
	"7xpc",    # contains non-standard aa MSE.
	# Some error in map_residue_to_index.
	"6m4v", "5e8e", "7xpc"
	]


class InitPrediction():
	def __init__( self,
		benchmark_name: str,
		modeling_version: int,
		cpu_cores: int = 5,
		device: str = "cpu" ):
		self.benchmark_name = benchmark_name # "xlmerged"  # xlsim/abag/oreilly/xlmerged
		# Define the modeling objective.
		self.modeling_objective = f"({self.benchmark_name}) Obtaining initial prediction."
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"
		self.modeling_version = modeling_version
		self.cpu_cores = cpu_cores
		self.device = device  # "cuda:0"
		self.data_sat_cutoff = 1.0 if self.benchmark_name == "oreilly" else 0.75
		self.sys_conf_suff = "_tpfp"

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
		# self.prep_msa_for_benchmark()
		self.predict_for_benchmark()
		self.filter_complexes()

		# # Load PDB ID to benchmark mapping.
		# self.pdb_benchmark_map = read_json( self.pdb_benchmark_map_file )
		self.write_data_for_selected_complexes()

		write_json( self.logs, self.logs_file )

	################################################################################
	################################################################################
	def initialize_logs_dict( self ):
		"""
		Log the following info:
			Time taken by each system.
			Memory consumed by each system.
			Entries for which an error occured.
		"""
		log_keys = ["time", "msa_time",
			"msa_created", "memory_allocated",
			"memory_reserved", "errored"]
		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )
			# Initialize keys added in later versions of the script.
			for k in log_keys:
				if k not in self.logs:
					self.logs[k] = {}
		else:
			self.logs = {k:{} for k in log_keys}


	def create_required_paths( self ):
		"""
		Create all the required paths.
		"""
		self.base_dir = os.path.join( os.path.abspath( "../benchmark/" ) )
		self.meta_dir = os.path.join( self.base_dir,
							f"{self.benchmark_name}_metadata" )

		self.benchmark_file = os.path.join( self.meta_dir,
						f"{self.benchmark_name}_benchmark{self.sys_conf_suff}.csv" )
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


	################################################################################
	################################################################################
	def log_error( self, sys_name: str ):
		"""
		If an error occurs while modeling,
			Log the errorneous entry_id in the dataset sepcific metadata dir.
			log the traceback in the system dir.
		
		Input:
		----------
		sys_name -> name of the complex modeled. For the benchmark, it's the PDB ID.
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

		Input:
		----------
		sys_name -> name of the complex modeled. For the benchmark, it's the PDB ID.
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


	def init_integrative_learning_module(
			self, sys_name: str ) -> IntegrativeLearning:
		"""
		Initialize the IntegrativeLearning module.

		Input:
		----------
		sys_name -> name of the complex modeled. For the benchmark, it's the PDB ID.

		Returns:
		----------
		il_obj -> an instance of IntegrativeLearning().
		topo_dict -> dict specifying the configs for modeling.
		"""
		# Initialize the topology dict.
		topo_dict = topology_dict()
		# Not using XL tolerance while computing metric.
		topo_dict.metrics.xlr.allow_xl_tolerance = False

		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				sys_config_file =  f"sys_config_{sys_name}{self.sys_conf_suff}.json",
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.create_required_paths_dirs()
		return il_obj, topo_dict


	################################################################################
	################################################################################
	def prep_msa_for_benchmark( self ):
		"""
		Prepare MSAs for all complexes in the benchmark, in parallel.
			MSA creation can be parallelized and to cut down runtime.
		"""
		if os.path.exists( self.init_pred_metrics_file ):
			self.init_pred_metrics = read_json( self.init_pred_metrics_file )

		try:
			with Pool( self.cpu_cores ) as p:
				for result in tqdm.tqdm(
					p.imap_unordered(
					self.prep_msa_per_system, self.benchmark["PDB ID"] ),
					total = len( self.benchmark["PDB ID"] )
					):
					sys_name, success, time_taken = result

					if success:
						self.logs["msa_created"][sys_name] = None
						self.logs["msa_time"][sys_name] = time_taken
		except KeyboardInterrupt:
			print( "\nInterrupted by user. Terminating pool..." )
			# Kill all processes.
			p.terminate()
			p.join()
			raise


	def prep_msa_per_system( self, sys_name: str ):
		"""
		For the given system, create input MSAs.

		Input:
		----------
		sys_name -> name of the complex modeled. For the benchmark, it's the PDB ID.

		Returns:
		----------
		sys_name -> same as above.
		success -> bool identifier indicating if MSA creation completed successfully.
		time_taken -> time taken for MSA creation.
		"""
		if sys_name in PROBLEMATIC:
			success = False
			time_taken = None
		elif sys_name in self.logs["errored"]:
			success = False
			time_taken = None

		if sys_name in self.logs["msa_created"]:
			success = True
			time_taken = self.logs["msa_time"][sys_name]
		else:
			# try:
			ts = time.perf_counter()
			il_obj, topo_dict = self.init_integrative_learning_module(
				sys_name = sys_name )
			sys_rep_obj = SystemRepresentation(
				sys_name = sys_name,
				is_multimer = True,
				sys_rep_config = topo_dict.system_representation,
				fasta_dir = il_obj.fasta_dir,
				alignment_dir = il_obj.alignment_dir,
				ofold_output_dir = il_obj.ofold_output_dir,
				seed_worker = il_obj.seed_worker
			)
			# Only run the input creation part.
			sys_rep_obj.create_required_paths()
			sys_rep_obj.init_ofold_config()
			sys_rep_obj.init_feature_processor()
			sys_rep_obj.prepare_input()
			success = True
			te = time.perf_counter()
			time_taken = te - ts
			# except KeyboardInterrupt:
			# 	print( "KeyBoard Interrupt..." )
			# 	raise
			# except:
			# 	# self.log_error( sys_name = sys_name )
			# 	print( "Here" )
			# 	success = False
			# 	time_taken = None
		return sys_name, success, time_taken


	################################################################################
	################################################################################
	def predict_for_benchmark( self ):
		"""
		Get initial prediction for all complexes in the benchmark.
		Obtain the violation loss and XL satisfaction.
		"""
		if os.path.exists( self.init_pred_metrics_file ):
			self.init_pred_metrics = read_json( self.init_pred_metrics_file )
		# else:
		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			if sys_name in PROBLEMATIC:
				continue

			print( f"\n----------- \033[1m {i}. {sys_name}\033[0m" )
			if sys_name in self.logs["errored"]:
				print( f"{sys_name} errored in a previous run..." )
				continue
			elif sys_name in self.init_pred_metrics:
				print( f"Already completed for {sys_name}..." )
				continue

			tic = time.perf_counter()
			violation, xl_metric, ptm, iptm = self.run_prediction_per_system( sys_name = sys_name )
			self.init_pred_metrics[sys_name] = {
				"violation": violation.item(),
				"xl_satisfaction": xl_metric.item(),
				"ptm": ptm,
				"iptm": iptm }
			toc = time.perf_counter()
			if not sys_name in self.logs["time"]:
				self.logs["time"][sys_name] = toc-tic
				write_json( self.logs, self.logs_file )
			write_json( self.init_pred_metrics, self.init_pred_metrics_file )



	def run_prediction_per_system(
		self, sys_name: str
		) -> Tuple[float, float, float, float]:
		"""
		Run OpenFold prediction for a given complex (system).
		Initialize the IntergrativeLearning class.
		Obtain an initial prediction.
		Compute the violation loss and XL metric.

		Input:
		----------
		sys_name -> name of the complex modeled. For the benchmark, it's the PDB ID.
		"""
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.synchronize()
		il_obj, topo_dict = self.init_integrative_learning_module(
			sys_name = sys_name )

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

		init_pred_file = get_init_pred_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		f = open_file_handler( init_pred_file, "rb" )
		out = pkl.load( f )
		f.close()
		ptm = float( out["ptm_score"] )
		iptm = float( out["iptm_score"] )

		del il_obj
		torch.cuda.empty_cache()
		return violation, xl_metric, ptm, iptm


	def filter_complexes( self ):
		"""
		Remove complexes for which data satisfcation is
			greater than the specified cutoff.
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

		Input:
		----------
		sys_name -> name of the complex modeled. For the benchmark, it's the PDB ID.

		Returns:
		----------
		sys_dict -> dict containing the relevant info as mentioned above for the system.
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
			"pTM": round( self.init_pred_metrics[sys_name]["ptm"], 3 ),
			"ipTM": round( self.init_pred_metrics[sys_name]["iptm"], 3 )
		}

		return sys_dict


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description = "obtain openfold predictions for the specified benchmark."
	)
	parser.add_argument(
		"-b", "--benchmark_name",
		type = str, required = True,
		help = "name of the benchmark. Allowed xlmerged/experiment..." )
	parser.add_argument(
		"-v", "--modeling_version",
		type = int, required = False, default = 0,
		 help = "an integer identifier for the mdeling run. Not strictly needed here." )
	parser.add_argument(
		"-c", "--cpu_cores", type = int,
		required = False, default = 4,
		 help = "no. of cpu cores to use for parallelizing MSA creation." )
	parser.add_argument(
		"-d", "--device",
		type = str, required = False, default = "cuda:0",
		 help = "device to be used for running OpenFold prediction." )

	args = parser.parse_args()
	InitPrediction(
		benchmark_name = args.benchmark_name,
		modeling_version = args.modeling_version,
		device = args.device
	).forward()
