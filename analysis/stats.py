"""
Given the predicted structures for the benchmark,
	compute summary statistics.
"""
from typing import List, Dict
import os
import numpy as np
import pandas as pd

from utils.analysis_utils import return_metric

from config import get_config_dict

from utils.analysis_utils import return_metric
from utils.paths import (
	get_benchmark_analysis_dir_path
)

CONFIG_DICT = get_config_dict()
BASE_DIR = os.path.join(
	os.path.abspath( CONFIG_DICT.models.base_dir )
	)
BENCHMARK_NAME = CONFIG_DICT.benchmark.globals.benchmark_name
ANALYSIS_DIR = get_benchmark_analysis_dir_path(
	base_dir = BASE_DIR,
	benchmark_name = BENCHMARK_NAME
)
STATS_DIR = os.path.join(
	ANALYSIS_DIR, "stats"
)

for d in [STATS_DIR]:
	os.makedirs( d, exist_ok = True )

################################################################################
################################################################################
def compute_data_satisfactIon_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the following stats
		related to data satisfaction,
	We categrorize the max data satisfcation into:
		Low (<0.25)
		Medium (0.25-0.75)
		High (>0.75)
	"""
	records = return_metric(
		metric = "xl_sat",
		config_name = config_name,
		models = [model]
	)
	stats = {}
	# for model in MODELS:
	max_xl_sat = np.array( records[model]["max_xl_sat"] )
	total = max_xl_sat.shape[0]
	low = np.sum( np.where( max_xl_sat <= 0.25, 1, 0 ) )
	high = np.sum( np.where( max_xl_sat >= 0.75, 1, 0 ) )
	medium = total - ( low + high )
	stats["low"] = low
	stats["medium"] = medium
	stats["high"] = high

	return stats

################################################################################
def compute_tm_score_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the following stats
		related to TM-score,
	We categrorize the TM-score into:
		Low (<0.33)
		Medium (0.33-0.7)
		High (>0.7)
	"""
	records = return_metric(
		metric = "tm",
		config_name = config_name,
		models = [model]
	)
	stats = {}
	# for model in MODELS:
	max_tm = np.array( records[model]["max_tm"] )
	total = max_tm.shape[0]
	low = np.sum( np.where( max_tm <= 0.33, 1, 0 ) )
	high = np.sum( np.where( max_tm >= 0.7, 1, 0 ) )
	medium = total - ( low + high )
	stats["low"] = low
	stats["medium"] = medium
	stats["high"] = high

	return stats

################################################################################
def compute_dockq_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the following stats
		related to DockQ,
	We categrorize the DockQ into:
		Low (<0.23)
		Medium (0.23-0.8)
		High (>0.8)
	"""
	records = return_metric(
		metric = "dockq",
		config_name = config_name,
		models = [model]
	)
	stats = {}
	# for model in MODELS:
	max_dockq = np.array( records[model]["max_dockq"] )
	total = max_dockq.shape[0]
	low = np.sum( np.where( max_dockq <= 0.33, 1, 0 ) )
	high = np.sum( np.where( max_dockq >= 0.7, 1, 0 ) )
	medium = total - ( low + high )
	stats["low"] = low
	stats["medium"] = medium
	stats["high"] = high

	return stats

################################################################################
def compute_unique_struct_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the following stats
		related to no. of unique structures,
	We categrorize the DockQ into:
		Low (1)
		Medium (2-10)
		High (>10)
	"""
	records = return_metric(
		metric = "unique_struct",
		config_name = config_name,
		models = [model]
	)
	stats = {}
	# for model in MODELS:
	num_unique_struct = np.array( records[model]["unique_struct"] )
	total = num_unique_struct.shape[0]
	low = np.sum( np.where( num_unique_struct == 1, 1, 0 ) )
	high = np.sum( np.where( num_unique_struct >= 10, 1, 0 ) )
	medium = total - ( low + high )
	stats["low"] = low
	stats["medium"] = medium
	stats["high"] = high

	return stats

################################################################################
def compute_unique_interface_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the following stats
		related to no. of structures with unique interface,
	We categrorize the DockQ into:
		Low (1)
		Medium (2-10)
		High (>10)
	"""
	records = return_metric(
		metric = "unique_interface",
		config_name = config_name,
		models = [model]
	)
	stats = {}
	# for model in MODELS:
	num_unique_interface = np.array( records[model]["unique_interface"] )
	total = num_unique_interface.shape[0]
	low = np.sum( np.where( num_unique_interface == 1, 1, 0 ) )
	high = np.sum( np.where( num_unique_interface >= 10, 1, 0 ) )
	medium = total - ( low + high )
	stats["low"] = low
	stats["medium"] = medium
	stats["high"] = high

	return stats

################################################################################
def return_stats_for_metric(
	model: str,
	config_name: str,
	metric: str
	):
	"""
	"""
	if metric == "xl_sat":
		stats = compute_data_satisfactIon_stats(
			model = model,
			config_name = config_name
		)
	elif metric == "tm":
		stats = compute_tm_score_stats(
			model = model,
			config_name = config_name
		)
	elif metric == "dockq":
		stats = compute_dockq_stats(
			model = model,
			config_name = config_name
		)
	else:
		raise ValueError( f"Unsupported metric specified: {metric}..." )

	return stats


################################################################################
def create_summary_file_per_config(
	configs: Dict[str, List]
	):
	"""
	Given a list of configs, create a summary file containing
		statistics for the benchmark.
	"""
	idx = 0
	for model in configs:
		flat_dict = {k: [] for k in ["config", "metric", "low", "medium", "high"]}
		for config_name in configs[model]:
			print( f"{idx}. {model}: {config_name}" )
			for metric in ["xl_sat", "tm", "dockq"]:
				stats = return_stats_for_metric(
					model = model,
					config_name = config_name,
					metric = metric
				)
				flat_dict["config"].append( config_name )
				flat_dict["metric"].append( metric )
				for level in ["low", "medium", "high"]:
					flat_dict[level].append( stats[level] )
			idx += 1

		summary_file = os.path.join(
			STATS_DIR, f"{model}_stats.csv"
		)
		df = pd.DataFrame( flat_dict )
		df.to_csv( summary_file, index = False )

################################################################################
################################################################################
if __name__ == "__main__":
	MODELS = ["alphalink2", "boltz2", "grasp"]
	configs = {
		"alphalink2": ["beta1"],
		"boltz2": [
			"alpha", "alpha2", "beta1", "beta2", "beta3", "beta4",
			"gamma", "delta1", "delta2", "delta3",
			"epsilon1", "epsilon2", "epsilon3", "epsilon4",
			"zeta1", "theta1"
		],
		"grasp": ["alpha", "beta1", "beta2", "beta3"]
	}
	create_summary_file_per_config( configs = configs )




