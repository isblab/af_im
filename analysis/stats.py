"""
Given the predicted structures for the benchmark,
	compute summary statistics.
"""
from typing import List, Dict
import os
import numpy as np
import pandas as pd

from config import get_config_dict

from utils.analysis_utils import (
	compute_tp_fp_xl_sat,
	return_metric
	)
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

THRESHOLD = {
	"xl_sat": 0.75,
	"tm": 0.7,
	"dockq": 0.23,
	"unique_struct": 1,
	"unique_interface": 1,
}
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
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	stats = {}
	max_xl_sat = np.array( records[model]["max_xl_sat"] )
	low = np.sum( np.where( max_xl_sat < THRESHOLD["xl_sat"], 1, 0 ) )
	high = np.sum( np.where( max_xl_sat >= THRESHOLD["xl_sat"], 1, 0 ) )
	stats["low"] = low
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
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	stats = {}
	max_tm = np.array( records[model]["max_tm"] )
	low = np.sum( np.where( max_tm < THRESHOLD["tm"], 1, 0 ) )
	high = np.sum( np.where( max_tm >= THRESHOLD["tm"], 1, 0 ) )
	stats["low"] = low
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
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	stats = {}
	max_dockq = np.array( records[model]["max_dockq"] )
	low = np.sum( np.where( max_dockq < THRESHOLD["dockq"], 1, 0 ) )
	high = np.sum( np.where( max_dockq >= THRESHOLD["dockq"], 1, 0 ) )
	stats["low"] = low
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
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	stats = {}
	num_unique_struct = np.array( records[model]["unique_struct"] )
	low = np.sum( np.where( num_unique_struct == 1, 1, 0 ) )
	high = np.sum( np.where( num_unique_struct > 1, 1, 0 ) )
	stats["low"] = low
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
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	stats = {}
	num_unique_interface = np.array( records[model]["unique_interface"] )
	low = np.sum( np.where( num_unique_interface == 1, 1, 0 ) )
	high = np.sum( np.where( num_unique_interface > 1, 1, 0 ) )
	stats["low"] = low
	stats["high"] = high

	return stats

################################################################################
def compute_molprob_score_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the no. of structures with mean
		MolProbity score < experimental resolution.
	We categrorize into:
		Low (<resolution)
		High (>resolution)
	"""
	records = return_metric(
		metric = "molprob",
		config_name = config_name,
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	stats = {}
	molprob_score = np.array( records[model]["mean_molprob"] ).reshape( -1 )
	resolution = np.array( records[model]["resolution"] ).reshape( -1 )
	diff = molprob_score-resolution
	low = np.sum( np.where( diff <= 0, 1, 0 ) )
	high = np.sum( np.where( diff > 0, 1, 0 ) )
	stats["low"] = low
	stats["high"] = high

	return stats

################################################################################
def compute_clashscore_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the no. of structures with mean the
		clashscore lower than that of the experiemntal structure.
	We categrorize into:
		Low (<resolution)
		High (>resolution)
	"""
	records = return_metric(
		metric = "molprob",
		config_name = config_name,
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	records_native = return_metric(
		metric = "molprob",
		config_name = config_name,
		benchmark_name = BENCHMARK_NAME,
		models = [model]
	)
	stats = {}
	pred_clash = np.array( records[model]["mean_clash"] )
	native_clash = np.array( records_native[model]["mean_clash"] )
	diff = pred_clash-native_clash
	low = np.sum( np.where( diff <= 0, 1, 0 ) )
	high = np.sum( np.where( diff > 0, 1, 0 ) )
	stats["low"] = low
	stats["high"] = high

	return stats

################################################################################
def compute_fp_xl_sat_stats( model: str, config_name: str ):
	"""
	For the given config, obtain the no. of complexes for which a method
		satisfies FP XLs.
	We categrorize into:
		Low (no FP XL satisfied)
		High (>0 FP XLs satisfied)
	"""
	records = return_metric(
		metric = "xl_sat",
		config_name = config_name,
		benchmark_name = BENCHMARK_NAME,
		models = [model],
	)
	stats = {}
	tp_xl_sat, fp_xl_sat, max_tp_sat, max_fp_sat = compute_tp_fp_xl_sat(
		xl_pair_sat = records[model]["xl_pair_sat"],
		labels = records[model]["label"]
	)
	# N -> no. of complexes
	num_complexes = len( records[model]["complex"] )
	# [N, T_f]; T_f -> no. of FP XLs
	fp_xl_sat = np.array( fp_xl_sat ).reshape( num_complexes, -1 )
	# [N]
	per_sys_count = np.count_nonzero( fp_xl_sat, axis = -1 )
	low = np.sum( np.where( per_sys_count == 0, 1, 0 ) )
	high = np.sum( np.where( per_sys_count > 0, 1, 0 ) )
	stats["low"] = low
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
	elif metric == "fp_xl_sat":
		stats = compute_fp_xl_sat_stats(
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
	elif metric == "unique_struct":
		stats = compute_unique_struct_stats(
			model = model,
			config_name = config_name
		)
	elif metric == "unique_interface":
		stats = compute_unique_interface_stats(
			model = model,
			config_name = config_name
		)
	elif metric == "molprob_score":
		stats = compute_molprob_score_stats(
			model = model,
			config_name = config_name
		)
	elif metric == "clash":
		stats = compute_clashscore_stats(
			model = model,
			config_name = config_name
		)
	else:
		raise ValueError( f"Unsupported metric specified: {metric}..." )

	return stats

################################################################################
def supp_table():
	"""
	Create a supplementary table containing the following,
		Mean XL satisfaction
		Max XL satisfaction
		Mean DockQ
		Max DockQ
		No. of unique interfaces
		Mean MolProbity score
		Mean clashscore
	"""
	config_name = "beta1"
	models = ["alphalink2", "boltz2", "grasp"]
	metrics = ["xl_sat", "dockq", "unique_interface", "molprob"]
	metric_labels = [
		"max_xl_sat", "mean_xl_sat",
		"max_dockq", "mean_dockq",
		"unique_interface",
		"mean_molprob", "mean_clash"
	]
	stats = {k:[] for k in ["complex", "method"] + metric_labels}
	# If True, add complex name and method. Don't add again for every metric.
	skip = False
	for metric in metrics:
		records = return_metric(
			metric = metric,
			config_name = config_name,
			benchmark_name = "crosslink",
			models = models
		)
		for i, sys_name in enumerate( records["boltz2"]["complex"] ):
			for model in models:
				if metric in ["xl_sat", "dockq"]:
					stats[f"max_{metric}"].append(
						round( records[model][f"max_{metric}"][i], 3 )
					)
					stats[f"mean_{metric}"].append(
						round( records[model][f"mean_{metric}"][i], 3 )
					)
				elif metric == "unique_interface":
					stats[metric].append(
						round( records[model][metric][i], 3 )
					)
				elif metric == "molprob":
					stats["mean_molprob"].append(
						round( records[model]["mean_molprob"][i], 3 )
					)
					stats["mean_clash"].append(
						round( records[model]["mean_clash"][i], 3 )
					)

				if not skip:
					stats["complex"].append( sys_name )
					stats["method"].append( model )
		skip = True
	for k in stats:
		print( k, " -> ", len( stats[k] ) )
	df = pd.DataFrame( stats )
	df.to_csv(
		os.path.join( STATS_DIR, f"supp_table_beta1.csv" )
	)

################################################################################
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
		flat_dict = {k: [] for k in ["config", "metric", "low", "high"]}
		for config_name in configs[model]:
			print( f"{idx}. {model}: {config_name}" )
			for metric in [
				"xl_sat", "fp_xl_sat", "tm", "dockq",
				"unique_struct", "unique_interface",
				"molprob_score", "clash"
				]:
				stats = return_stats_for_metric(
					model = model,
					config_name = config_name,
					metric = metric
				)
				flat_dict["config"].append( config_name )
				flat_dict["metric"].append( metric )
				for level in ["low", "high"]:
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
		"alphalink2": ["beta1", "beta2", "beta3"],
		"boltz2": [
			"alpha", "alpha2", "beta1", "beta2", "beta3", "beta4",
			"gamma", "delta1", "delta2", "delta3",
			"epsilon1", "epsilon2", "epsilon3", "epsilon4",
			"zeta1", "theta1"
		],
		"grasp": ["alpha", "beta1", "beta2", "beta3"]
	}
	create_summary_file_per_config( configs = configs )
	supp_table()

