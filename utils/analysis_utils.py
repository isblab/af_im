"""
Contains methods for parsing the analysis dict and
	return the desired metric for subsequent analysis.
"""
from typing import List, Tuple, Dict
import os
import numpy as np

from config import get_config_dict

from utils.utils import read_json
from utils.paths import (
	get_meta_dir_path,
	get_model_output_dir_path,
	get_benchmark_analysis_dir_path
)

CONFIG_DICT = get_config_dict()
BASE_DIR = os.path.join(
	os.path.abspath( CONFIG_DICT.models.base_dir )
	)

# Complexes for which prediction failed
IGNORE_SYSTEMS = ["5xct", "6iww", "7agf"]

################################################################################
################################################################################
def load_logs_dict(
	model: str,
	config_name: str,
	benchmark_name: str
	):
	"""
	Load the logs dict for the specified model and config.
	"""
	model_out_dir = get_model_output_dir_path(
		base_dir = BASE_DIR,
		benchmark_name = benchmark_name,
		model = model,
		config_name = config_name
	)
	logs_file = os.path.join(
		model_out_dir, f"Logs_{model}.json"
	)
	logs_dict = read_json( logs_file )
	return logs_dict


def load_analysis_dict(
	model: str,
	config_name: str,
	benchmark_name: str
):
	"""
	For the given model and config, load and
		return  the analysis dict.

	Inputs:
	----------
	model: identifier for the model being used: alphalInk2/grasp/boltz2
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	benchmark_name: 

	Returns:
	----------
	The analysis results for the specified model config.
	"""
	analysis_dir = get_benchmark_analysis_dir_path(
		base_dir = BASE_DIR,
		benchmark_name = benchmark_name
	)
	per_config_logs_file = os.path.join(
		analysis_dir,
		f"Logs_{benchmark_name}_{model}.npy"
	)
	analysis_dict = np.load(
		per_config_logs_file, allow_pickle = True
	 ).item()

	model_key = f"{model}_{config_name}"
	return analysis_dict[model_key]

################################################################################
# Prepare inputs for plotting the the various metrics
################################################################################
def prep_pred_time_input(
	config_name: str,
	benchmark_name: str,
	models: List[str]
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes obtain the time taken for prediction.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.

	Returns:
	----------
	records: dict containing the complexes and their
		respective prediction time.
	"""
	records = {}
	for model in models:
		records[model] = {
			k:[] for k in ["complex", "time"]
			}
		data = load_logs_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)

		for sys_name in data["completed"]:
			if sys_name in IGNORE_SYSTEMS:
				continue
			time_taken = data["time"][sys_name]/ 3600
			records[model]["complex"].append( sys_name )
			records[model]["time"].append( time_taken )
		records[model]["avg_time"] = np.mean( records[model]["time"] )
	return records


def prep_xl_satisfaction_input(
	config_name: str,
	models: List[str],
	benchmark_name: str
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes obtain the following:
		Mean XL satisfaction
		Max XL satisfaction

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.

	Returns:
	----------
	records: 
	"""
	records = {}
	for model in models:
		records[model] = {
			k:[] for k in [
				"complex", "per_model_xl_sat", "mean_xl_sat",
				"max_xl_sat", "xl_pair_sat", "label"
				]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			xl_sat = data[sys_name]["xl_metrics"]["xl_satisfaction"]
			xl_pair_sat = data[sys_name]["xl_metrics"]["xl_pair_satisfaction"]
			label = data[sys_name]["xl_metrics"]["label"]

			records[model]["complex"].append( sys_name )
			records[model]["per_model_xl_sat"].extend( xl_sat )
			records[model]["mean_xl_sat"].append( xl_sat.mean() )
			records[model]["max_xl_sat"].append( xl_sat.max() )
			records[model]["xl_pair_sat"].append( xl_pair_sat )
			records[model]["label"].append( label )

	return records

################################################################################
def compute_tp_fp_xl_sat(
	xl_pair_sat: List[np.ndarray],
	labels: List[np.ndarray]
) -> Tuple[List, List]:
	"""
	Given the per XL satisfaction across all models compute
		the fraction of TP and FP XLs satisfied.
	Given the no. of models satisfying a given XL pair, we
		consider an XL satisfied if even 1 model satifies it.

	Inputs:
	----------
	xl_pair_sat: [T, M] contains count of the no. of models
		satisfying each XL.
		T -> no. of XLs; M -> no. of models.
	labels: binary array indicating whether an XL is TP (1) or FP (0).
	"""
	tp_xl_sat, fp_xl_sat = [], []
	for j in range( len( labels ) ):
		label = labels[j]
		xl_sat = xl_pair_sat[j]

		total_tp = np.count_nonzero( label )
		total_fp = label.shape[0] - total_tp

		# TP XLs
		tp_sat = np.count_nonzero(
			np.where( xl_sat*label > 0, 1, 0 )
			)/total_tp
		# FP XLs -> (1-label) 
		fp_sat = np.count_nonzero(
			np.where( xl_sat*( 1-label ) > 0, 1, 0 )
			)/total_fp

		tp_xl_sat.append( tp_sat )
		fp_xl_sat.append( fp_sat )
	return tp_xl_sat, fp_xl_sat

################################################################################
def prep_native_tm_input(
	config_name: str,
	benchmark_name: str,
	models: List[str],
	native_model_id: int = 1000,
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes obtain the following:
		Mean TM-score wrt native
		Max TM-score wrt native
	TM-score is stored as a nested dict with,
		model_id1: {model_id2: tm}

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	models: list of models for which to gather data.

	Returns:
	----------
	records: 
	"""
	records = {}
	for model in models:
		records[model] = {
			k:[] for k in ["complex", "per_model_tm", "mean_tm", "max_tm"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			tm = []
			sorted_model_ids = sorted( list( data[sys_name]["tm"].keys() ) )
			for model_id1 in sorted_model_ids:
					tm.append(
						data[sys_name]["tm"][model_id1][native_model_id]["tm"]
					)
			tm = np.array( tm )

			records[model]["complex"].append( sys_name )
			records[model]["per_model_tm"].extend( tm )
			records[model]["mean_tm"].append( tm.mean() )
			records[model]["max_tm"].append( tm.max() )
	return records

################################################################################
def prep_native_dockq_input(
	config_name: str,
	benchmark_name: str,
	models: List[str],
	native_model_id: int = 1000,
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes obtain the following:
		Mean DockQ wrt native
		Max DockQ wrt native
	DockQ is stored as a nested dict with,
		model_id1: {model_id2: dockq}

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	models: list of models for which to gather data.

	Returns:
	----------
	records: 
	"""
	records = {}
	for model in models:
		records[model] = {
			k:[] for k in ["complex", "per_model_dockq", "mean_dockq", "max_dockq"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			dockq = []
			sorted_model_ids = sorted( list( data[sys_name]["dockq"].keys() ) )
			for model_id1 in sorted_model_ids:
				d = data[sys_name]["dockq"][model_id1][native_model_id]
				# for k, v in data[sys_name]["dockq"][model_id1].items():
				dockq.append( d )
			dockq = np.array( dockq )

			records[model]["complex"].append( sys_name )
			records[model]["per_model_dockq"].extend( dockq )
			records[model]["mean_dockq"].append( dockq.mean() )
			records[model]["max_dockq"].append( dockq.max() )
	return records

################################################################################
def prep_unique_models_input(
	config_name: str,
	benchmark_name: str,
	similarity_metric: str,
	models: List[str]
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes plot bar plot for the no. of unique structures.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.

	Returns:
	----------
	records: 
	"""
	records = {}
	if similarity_metric not in ["unique_struct", "unique_interface"]:
		raise ValueError(
			"Incorrect similarity metric specified" +
			" for selecting unique models..."
		)
	for model in models:
		records[model] = {
			k:[] for k in ["complex", similarity_metric]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)
		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			unique_models = len(
				data[sys_name][similarity_metric]["model_id"]
			)

			records[model]["complex"].append( sys_name )
			records[model][similarity_metric].append( unique_models )
	return records

################################################################################
def return_molprobity_metric(
	molprob_dict: Dict,
	molprob_metric: str,
	):
	"""
	Given a dict containing per-model Molprobity metrics, return the following:
		Per-model metric values
		Mean metric value
		Max/Min metric value
	"""
	sorted_model_ids = sorted( list( molprob_dict.keys() ) )
	if molprob_metric == "molprob":
		per_model_metric = np.array(
			[molprob_dict[k]["MolProbity score"] for k in sorted_model_ids]
			)
		mean_metric = per_model_metric.mean()
	elif molprob_metric == "clash":
		per_model_metric = np.array(
			[molprob_dict[k]["Clashscore"] for k in sorted_model_ids]
			)
		mean_metric = per_model_metric.mean()
	# Ramachandran favored
	elif molprob_metric == "favored":
		per_model_metric = np.array(
			[molprob_dict[k]["favored"] for k in sorted_model_ids]
			)
		mean_metric = per_model_metric.mean()
	# Rotamer outliers
	elif molprob_metric == "rotamer":
		per_model_metric = np.array(
			[molprob_dict[k]["Rotamer outliers"] for k in sorted_model_ids]
			)
		mean_metric = per_model_metric.mean()
	# C-beta deviation
	elif molprob_metric == "C_beta":
		per_model_metric = np.array(
			[molprob_dict[k]["C-beta deviations"] for k in sorted_model_ids]
			)
		mean_metric = per_model_metric.mean()
	else:
		ValueError( f"Unsupported molprobity metric specified: {molprob_metric}..." )
	return per_model_metric, mean_metric


def prep_molprobity_input(
	config_name: str,
	benchmark_name: str,
	for_native: bool,
	models: List[str]
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes obtain the following:
		Molprobity score for the predicted model
		Resolution of the experimental structure

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.

	Returns:
	----------
	"""
	records = {}

	meta_dir = get_meta_dir_path(
		base_dir = BASE_DIR,
		benchmark_name = BENCHMARK_NAME
	)
	resolution_dict_file = os.path.join( meta_dir, "resolution_dict.json" )
	resolution_dict = read_json( resolution_dict_file )

	for model in models:
		records[model] = {
			k:[] for k in [
				"complex",
				"per_model_molprob", "per_model_clash", "per_model_favored",
				"per_model_rotamer", "per_model_C_beta",
				"mean_molprob", "mean_clash", "mean_favored",
				"mean_rotamer", "mean_C_beta",
				"resolution"
				]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			records[model]["complex"].append( sys_name )
			molprob_key = "molprob_native" if for_native else "molprob"
			for m in ["molprob", "clash", "favored", "rotamer", "C_beta"]:
				per_model_metric, mean_metric = return_molprobity_metric(
					molprob_dict = data[sys_name][molprob_key],
					molprob_metric = m
				)
				records[model][f"per_model_{m}"].extend(
					per_model_metric.reshape( -1 ).tolist()
					)
				records[model][f"mean_{m}"].append( mean_metric )
			records[model]["resolution"].append( resolution_dict[sys_name] )
	return records

################################################################################
def prep_rmsf_input(
	config_name: str,
	benchmark_name: str,
	models: List[str]
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes obtain the following:
		Per-residue RMSF
		pLDDT

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.

	Returns:
	----------
	"""
	records = {}

	for model in models:
		records[model] = {
			k:[] for k in ["complex", "rmsf", "plddt"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			rmsf = data[sys_name]["rmsf"]["rmsf_per_residue"].reshape( -1 )
			plddt = data[sys_name]["rmsf"]["plddt"].mean( axis = 0 ).reshape( -1 )

			records[model]["complex"].append( sys_name )
			records[model]["rmsf"].append( rmsf )
			records[model]["plddt"].append( plddt )
	return records

################################################################################
def prep_confidence_metrics_input(
	config_name: str,
	benchmark_name: str,
	models: List[str],
) -> Dict[str, List]:
	"""
	For all the methods (AlphaLink2, Boltz2, GRASP), across all
		complexes obtain the following:
		Global confidence metrics
			ipTM+pTM for AlphaLink2
			Ranking score for Boltz2
			Score for GRASP

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	models: list of models for which to gather data.

	Returns:
	----------
	records: 
	"""
	records = {}
	for model in models:
		records[model] = {
			k:[] for k in [
				"complex", "per_model_confidence",
				"mean_confidence", "max_confidence"
				]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name,
			benchmark_name = benchmark_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			sorted_model_ids = sorted( list( data[sys_name]["confidence"].keys() ) )
			per_model_conf = []
			for model_id in sorted_model_ids:
				per_model_conf.append(
					data[sys_name]["confidence"][model_id]
				)
			per_model_conf = np.array( per_model_conf )

			if model == "grasp":
				# For GRASp, the score ranges from 0-100.
				per_model_conf = per_model_conf/100

			records[model]["complex"].append( sys_name )
			records[model]["per_model_confidence"].extend( per_model_conf )
			records[model]["mean_confidence"].append( per_model_conf.mean() )
			records[model]["max_confidence"].append( per_model_conf.max() )
	return records

################################################################################
def return_metric(
	config_name: str,
	benchmark_name: str,
	models: List[str],
	metric: str,
	molprob_native: bool = False
) -> Dict[str, List]:
	"""
	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	metric: 
	molprob_native: bool flag for molprobity metrics.
		If true, returns molprobity metrics for the native structure.
	
	Returns:
	----------
	records: 
	"""
	global MODELS, BENCHMARK_NAME, ANALYSIS_DIR
	MODELS = models
	BENCHMARK_NAME = benchmark_name
	ANALYSIS_DIR = get_benchmark_analysis_dir_path(
		base_dir = BASE_DIR,
		benchmark_name = benchmark_name
	)

	if metric == "time":
		records = prep_pred_time_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			models = models
		)
	elif metric == "xl_sat":
		records = prep_xl_satisfaction_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			models = models
		)
	elif metric == "tm":
		records = prep_native_tm_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			models = models
		)
	elif metric == "dockq":
		records = prep_native_dockq_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			models = models
		)
	elif metric == "unique_struct":
		records = prep_unique_models_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			similarity_metric = "unique_struct",
			models = models
		)
	elif metric == "unique_interface":
		records = prep_unique_models_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			similarity_metric = "unique_interface",
			models = models
		)
	elif metric == "molprob":
		records = prep_molprobity_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			for_native = molprob_native,
			models = models
		)
	elif metric == "rmsf":
		records = prep_rmsf_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			models = models
		)
	elif metric == "confidence":
		records = prep_confidence_metrics_input(
			config_name = config_name,
			benchmark_name = benchmark_name,
			models = models
		)
	else:
		raise ValueError( f"Unknown metric specified - {metric}..." )
	return records
