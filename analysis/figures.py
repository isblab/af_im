"""
Given the analysis data across the benchmark for all the models
	create the required plots.
For per-config analysis, plot the following across all complexes:
	- XL satisfaction: unguided, short and long XLs
	- No. of unique structures
	- DockQ wrt native
	- Molprobity vs resolution
"""
from typing import List, Dict, Any
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_config_dict

from utils.utils import read_json
from utils.paths import (
	get_meta_dir_path,
	get_model_output_dir_path,
	get_benchmark_analysis_dir_path
)
warnings.filterwarnings( "ignore" )

################################################################################
# Globals
################################################################################
# Complexes for which prediction failed
IGNORE_SYSTEMS = ["5xct", "6iww", "7agf"]
THRESHOLD = {
	"xl_sat": 0.75,
	"tm": 0.7,
	"dockq": 0.23,
	"unique_struct": 1,
	"unique_interface": 1,
	"molprob": None
}
METHOD_LABELS = {
	"alphalink2": "AlphaLink2",
	"boltz2": "Boltz2",
	"grasp": "GRASP"
}
COLOR = {
	"alphalink2": "blue",
	"boltz2": "green",
	"grasp": "orange"
}
MARKER = {
	"alphalink2": "s",
	"boltz2": "v",
	"grasp": "o"
}
XY_LABEL_SIZE = 20
TITLE_SIZE = 20

CONFIG_DICT = get_config_dict()
BASE_DIR = os.path.join(
	os.path.abspath( CONFIG_DICT.models.base_dir )
	)
BENCHMARK_NAME = CONFIG_DICT.benchmark.globals.benchmark_name
ANALYSIS_DIR = get_benchmark_analysis_dir_path(
	base_dir = BASE_DIR,
	benchmark_name = BENCHMARK_NAME
)
FIG_DIR = os.path.join(
	ANALYSIS_DIR,
	"figures"
)
MEAN_MAX__DIR = os.path.join(
	FIG_DIR, "mean_max_figs"
)
CROSS_CONFIG_DIR = os.path.join(
	FIG_DIR, "cross_config_figs"
)
XL_SAT_DIR = os.path.join(
	FIG_DIR, "xl_sat_figs"
)
TM_DIR = os.path.join(
	FIG_DIR, "tm_figs"
)
DOCKQ_DIR = os.path.join(
	FIG_DIR, "dockq_figs"
)
TM_DOCKQ_DIR = os.path.join(
	FIG_DIR, "tm_dockq_figs"
)
XL_DOCKQ_DIR = os.path.join(
	FIG_DIR, "xl_dockq_figs"
)
UNIQUE_MODEL_DIR = os.path.join(
	FIG_DIR, "unique_models_figs"
)
MEAN_MAX_DIR = os.path.join(
	FIG_DIR, "mean_max_xlsat_dockq_figs"
)
RMSF_DIR = os.path.join(
	FIG_DIR, "rmsf_figs"
)
MOLPROB_DIR = os.path.join(
	FIG_DIR, "molprob_figs"
)
# Create the required directories
for d in [
	FIG_DIR, CROSS_CONFIG_DIR, XL_SAT_DIR,
	TM_DIR, DOCKQ_DIR, TM_DOCKQ_DIR, MOLPROB_DIR,
	MEAN_MAX_DIR, UNIQUE_MODEL_DIR, RMSF_DIR
	]:
	os.makedirs( d, exist_ok = True )

################################################################################
################################################################################
def load_logs_dict( model: str, config_name: str ):
	"""
	Load the logs dict for the specified model and config.
	"""
	model_out_dir = get_model_output_dir_path(
		base_dir = BASE_DIR,
		benchmark_name = BENCHMARK_NAME,
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
	config_name: str
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

	Returns:
	----------
	The analysis results for the specified model config.
	"""
	per_config_logs_file = os.path.join(
		ANALYSIS_DIR,
		f"Logs_{BENCHMARK_NAME}_{model}.npy"
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
	config_name: str
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
	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", "time"]
			}
		data = load_logs_dict(
			model = model,
			config_name = config_name
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
	config_name: str
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
	for model in MODELS:
		records[model] = {
			k:[] for k in [
				"complex", "per_model_xl_sat", "mean_xl_sat",
				"max_xl_sat", "xl_pair_sat"
				]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			xl_sat = data[sys_name]["xl_metrics"]["xl_satisfaction"]
			# print( data[sys_name]["xl_metrics"].keys() )
			xl_pair_sat = data[sys_name]["xl_metrics"]["xl_pair_satisfaction"]
			# print( data[sys_name]["xl_metrics"].keys() )
			# label = data[sys_name]["xl_metrics"]["label"]

			records[model]["complex"].append( sys_name )
			records[model]["per_model_xl_sat"].extend( xl_sat )
			records[model]["mean_xl_sat"].append( xl_sat.mean() )
			records[model]["max_xl_sat"].append( xl_sat.max() )
			records[model]["xl_pair_sat"].append( xl_pair_sat )
			# records[model]["label"].append( label )

	return records

################################################################################
def prep_native_tm_input(
	config_name: str
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
	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", "per_model_tm", "mean_tm", "max_tm"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			tm = []
			for model_id1 in data[sys_name]["tm"]:
				for k, v in data[sys_name]["tm"][model_id1].items():
					tm.append( v["tm"] )
			tm = np.array( tm )

			records[model]["complex"].append( sys_name )
			records[model]["per_model_tm"].extend( tm )
			records[model]["mean_tm"].append( tm.mean() )
			records[model]["max_tm"].append( tm.max() )
	return records

################################################################################
def prep_native_dockq_input(
	config_name: str
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
	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", "per_model_dockq", "mean_dockq", "max_dockq"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			dockq = []
			for model_id1 in data[sys_name]["dockq"]:
				for k, v in data[sys_name]["dockq"][model_id1].items():
					dockq.append( v )
			dockq = np.array( dockq )

			records[model]["complex"].append( sys_name )
			records[model]["per_model_dockq"].append( dockq )
			records[model]["mean_dockq"].append( dockq.mean() )
			records[model]["max_dockq"].append( dockq.max() )
	return records

################################################################################
def prep_unique_models_input(
	config_name: str,
	similarity_metric: str
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
	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", similarity_metric]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
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
	molprob_metric: str
	):
	"""
	Given a dict containing per-model Molprobity metrics, return the following:
		Per-model metric values
		Mean metric value
		Max/Min metric value
	"""
	if molprob_metric == "molprob":
		per_model_metric = np.array(
			[v["MolProbity score"] for k,v in molprob_dict.items()]
			)
		mean_metric = per_model_metric.mean()
	elif molprob_metric == "clash":
		per_model_metric = np.array(
			[v["Clashscore"] for k,v in molprob_dict.items()]
			)
		mean_metric = per_model_metric.mean()
	# Ramachandran favored
	elif molprob_metric == "favored":
		per_model_metric = np.array(
			[v["favored"] for k,v in molprob_dict.items()]
			)
		mean_metric = per_model_metric.mean()
	else:
		ValueError( f"Unsupported molprobity metric specified: {molprob_metric}..." )
	return per_model_metric, mean_metric


def prep_molprobity_input(
	config_name: str,
	for_native: bool
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

	for model in MODELS:
		records[model] = {
			k:[] for k in [
				"complex",
				"per_model_molprob", "per_model_clash", "per_model_favored",
				"mean_molprob", "mean_clash", "mean_favored",
				"resolution"
				]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			records[model]["complex"].append( sys_name )
			molprob_key = "molprob_native" if for_native else "molprob"
			for m in ["molprob", "clash", "favored"]:
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
	config_name: str
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

	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", "rmsf", "plddt"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
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
def return_metric(
	config_name: str,
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
	if metric == "time":
		records = prep_pred_time_input(
			config_name = config_name,
		)
	elif metric == "xl_sat":
		records = prep_xl_satisfaction_input(
			config_name = config_name,
		)
	elif metric == "tm":
		records = prep_native_tm_input(
			config_name = config_name,
		)
	elif metric == "dockq":
		records = prep_native_dockq_input(
			config_name = config_name,
		)
	elif metric == "unique_struct":
		records = prep_unique_models_input(
			config_name = config_name,
			similarity_metric = "unique_struct"
		)
	elif metric == "unique_interface":
		records = prep_unique_models_input(
			config_name = config_name,
			similarity_metric = "unique_interface"
		)
	elif metric == "molprob":
		records = prep_molprobity_input(
			config_name = config_name,
			for_native = molprob_native
		)
	elif metric == "rmsf":
		records = prep_rmsf_input(
			config_name = config_name,
		)
	else:
		raise ValueError( f"Unknown metric specified - {metric}..." )
	return records

################################################################################
# Create plots fo rthe required metrics
################################################################################
def plot_pred_time_taken_across_models(
	config_name: str
):
	"""
	Plot the time taken for prediction for all models
		across all complexes as a bar plot.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records = prep_pred_time_input( config_name = config_name )
	fig, ax = plt.subplots( 1, 1, figsize = ( 7, 7 ) )

	for i, model in enumerate( MODELS ):
		ax.bar(
			i,
			records[model]["avg_time"],
			0.5,
			color = COLOR[model]
		)

	ax.set_xticks( [0, 1, 2] )
	ax.tick_params( axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		["AlphaLink2", "Boltz2", "GRASP"],
		rotation = 0, fontsize = 12
		)
	ax.set_ylabel( f"Average time taken (hours)", fontsize = XY_LABEL_SIZE-5 )
	ax.set_xlabel( "Method", fontsize = XY_LABEL_SIZE-5 )
	ax.tick_params(
		axis = "both",
		labelsize = 20,
		length = 10,
		width = 4
	)

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"time_taken_{config_name}.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def plot_xl_satisfaction_across_models(
	config_name: str
):
	"""
	Plot the mean and max XL satisfaction for all models
		across all complexes as a scatter plot.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records = prep_xl_satisfaction_input( config_name = config_name )

	fig, ax = plt.subplots( 2, 1, figsize = ( 20, 10 ) )
	for i, agg in enumerate( ["mean", "max"] ):
		# Create a scatter plot for each method.
		for j, model in enumerate( MODELS ):
			X = np.arange( 0, len( records[model]["complex"] ), 1 )
			ax[i].scatter(
				X,
				np.array( records[model][f"{agg}_xl_sat"] ), # + 0.01*( i+1 )
				s = 100,
				c = COLOR[model],
				marker = MARKER[model],
				label = METHOD_LABELS[model]
			)

		ax[i].axhline( 0.75, color = "red" )
		ax[i].set_xticks( X )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels(
			records[model]["complex"], rotation = 90, fontsize = 12
			)
		ax[i].set_ylabel( f"{agg.capitalize()} XL satisfaction", fontsize = XY_LABEL_SIZE )
		ax[i].set_xlabel( "Complexes", fontsize = XY_LABEL_SIZE )
		ax[i].set_ylim( -0.1, 1.1 )
		ax[i].legend()
		# ax[i].xticks( rotation = 90 )
		ax[i].tick_params(
			axis = "both",
			labelsize = 20,
			length = 10,
			width = 4
		)
	plt.tight_layout()
	file = os.path.join(
		XL_SAT_DIR, f"xl_satisfaction_{config_name}.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def plot_native_tm_across_models(
	config_name: str
):
	"""
	Plot the mean and max TM-score wrt the native structure for all
		models across all complexes as a scatter plot.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records = prep_native_dockq_input( config_name = config_name )

	fig, ax = plt.subplots( 2, 1, figsize = ( 20, 10 ) )
	for i, agg in enumerate( ["mean", "max"] ):
		# Create a scatter plot for each method.
		for j, model in enumerate( MODELS ):
			X = np.arange( 0, len( records[model]["complex"] ), 1 )
			ax[i].scatter(
				X,
				records[model][f"{agg}_dockq"],
				s = 100,
				c = COLOR[model],
				marker = MARKER[model],
				label = METHOD_LABELS[model]
			)

		ax[i].axhline( THRESHOLD["tm"], color = "red" )
		ax[i].set_xticks( X )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels(
			records[model]["complex"], rotation = 90, fontsize = 12
			)
		ax[i].set_ylabel( f"{agg.capitalize()} TM-score", fontsize = XY_LABEL_SIZE )
		ax[i].set_xlabel( "Complexes", fontsize = XY_LABEL_SIZE )
		ax[i].set_ylim( -0.1, 1.1 )
		ax[i].legend()
		# ax[i].xticks( rotation = 90 )
		ax[i].tick_params(
			axis = "both",
			labelsize = 20,
			length = 10,
			width = 4
		)
	plt.tight_layout()
	file = os.path.join(
		TM_DIR, f"tm_{config_name}.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def plot_native_dockq_across_models(
	config_name: str
):
	"""
	Plot the mean and max DockQ wrt the native structure for all
		models across all complexes as a scatter plot.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records = prep_native_dockq_input( config_name = config_name )

	fig, ax = plt.subplots( 2, 1, figsize = ( 20, 10 ) )
	for i, agg in enumerate( ["mean", "max"] ):
		# Create a scatter plot for each method.
		for j, model in enumerate( MODELS ):
			X = np.arange( 0, len( records[model]["complex"] ), 1 )
			ax[i].scatter(
				X,
				records[model][f"{agg}_dockq"],
				s = 100,
				c = COLOR[model],
				marker = MARKER[model],
				label = METHOD_LABELS[model]
			)

		ax[i].axhline( 0.23, color = "red" )
		ax[i].set_xticks( X )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels(
			records[model]["complex"], rotation = 90, fontsize = 12
			)
		ax[i].set_ylabel( f"{agg.capitalize()} DockQ", fontsize = XY_LABEL_SIZE )
		ax[i].set_xlabel( "Complexes", fontsize = XY_LABEL_SIZE )
		ax[i].set_ylim( -0.1, 1.1 )
		ax[i].legend()
		# ax[i].xticks( rotation = 90 )
		ax[i].tick_params(
			axis = "both",
			labelsize = 20,
			length = 10,
			width = 4
		)

	plt.tight_layout()
	file = os.path.join(
		DOCKQ_DIR, f"dockq_{config_name}.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def plot_unique_models_across_models(
	config_name: str
):
	"""
	Plot the no. of unique predicted structures  for all
		models across all complexes.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records_tm = prep_unique_models_input(
		config_name = config_name,
		similarity_metric = "unique_struct" )
	records_dockq = prep_unique_models_input(
		config_name = config_name,
		similarity_metric = "unique_interface" )

	fig, ax = plt.subplots( 2, 1, figsize = ( 20, 10 ) )
	widths = [-0.25, 0, 0.25]
	# Create a scatter plot for each method.
	labels = ["Unique structure", "Unique interface"]
	for i, model in enumerate( MODELS ):
		for j, metric in enumerate( ["unique_struct", "unique_interface"] ):
			X = np.arange( 0, len( records_tm[model]["complex"] ), 1 )
			if metric == "unique_struct":
				ax[j].bar(
					X+widths[i],
					records_tm[model][metric], 0.1, # widths[i],
					color = COLOR[model],
					label = f"{METHOD_LABELS[model]}"
				)
			else:
				ax[j].bar(
					X+widths[i],
					records_dockq[model][metric], 0.1, # widths[i],
					color = COLOR[model],
					label = f"{METHOD_LABELS[model]}"
				)

			ax[j].axhline( 1, color = "red" )
			ax[j].set_xticks( X )
			ax[j].tick_params( axis = "both", width = 2, length = 5 )
			ax[j].set_xticklabels(
				records_tm[model]["complex"], rotation = 90, fontsize = 16
				)
			ax[j].set_title( f"{labels[j]}", fontsize = TITLE_SIZE )
			ax[j].set_ylabel( f"No. of unique models", fontsize = XY_LABEL_SIZE )
			ax[j].set_xlabel( "Complexes", fontsize = XY_LABEL_SIZE )
			ax[j].set_ylim( 0, 25+1 )
			ax[j].legend()
			# ax[j].xticks( rotation = 90 )
			ax[j].tick_params(
				axis = "both",
				labelsize = 20,
				length = 10,
				width = 4
			)

	plt.tight_layout()
	file = os.path.join(
		UNIQUE_MODEL_DIR, f"unique_models_{config_name}.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def plot_molprobity_metrics_across_models(
	config_name: str
):
	"""
	Plot the Molprobity metrics for all models
		across all complexes.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records_pred = return_metric(
		config_name = config_name,
		metric = "molprob",
		molprob_native = False
		)
	records_native = return_metric(
		config_name = config_name,
		metric = "molprob",
		molprob_native = True
		)
	molprob_score_file = os.path.join(
		MOLPROB_DIR, f"molprobity_score_{config_name}.png"
		)
	molprob_other_metrics_file = os.path.join(
		MOLPROB_DIR, f"other_molprob_{config_name}.png"
		)

	plot_molprobity_score(
		records_pred = records_pred,
		records_native = records_native,
		plot_file = molprob_score_file
	)
	plot_other_molprobity_metrics(
		records_pred = records_pred,
		records_native = records_native,
		plot_file = molprob_other_metrics_file
	)


def plot_molprobity_score(
	records_pred: Dict[str, List],
	records_native: Dict[str, List],
	plot_file: str
):
	"""
	Create the following plots for the MolProbity score
		for all models across all complexes:
		- Mean MolProbity score for predicted and native.
		- Mean MolProbity score for predicted vs 
			resolution

	Inputs:
	----------
	"""
	if len( MODELS ) == 3:
		fig, axes = plt.subplots( 3, 2, figsize = ( 12, 12 ) )
	elif len( MODELS ) == 2:
		fig, axes = plt.subplots( 2, 2, figsize = ( 10, 12 ) )
	elif len( MODELS ) == 1:
		fig, axes = plt.subplots( 1, 2, figsize = ( 8, 12 ) )
	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		xlabels = ["MolProbity Score for native", "Experimental resolution"]
		for j, native_metric in enumerate( ["mean_molprob", "resolution"] ):
			if len( MODELS ) == 1:
				ax = axes[j]
			else:
				ax = axes[i, j]
			ax.scatter(
				records_native[model][native_metric],
				records_pred[model]["mean_molprob"],
				c = COLOR[model], label = METHOD_LABELS[model]
				)

			ax.plot( [0, 5], [0, 5], c = "gray" )
			ax.set_title( METHOD_LABELS[model], fontsize = TITLE_SIZE )
			ax.set_xlabel( xlabels[j], fontsize = XY_LABEL_SIZE )
			ax.set_ylabel( f"MolProbity score", fontsize = XY_LABEL_SIZE )
			ax.tick_params(
				axis = "both",
				labelsize = 20,
				length = 10,
				width = 4
			)
	plt.tight_layout()
	plt.savefig( plot_file, dpi = 300 )
	plt.close()


def plot_other_molprobity_metrics(
	records_pred: Dict[str, List],
	records_native: Dict[str, List],
	plot_file: str
):
	"""
	Plot other MolProbity metrics, including clashscore,
		ramachandran favored for all models across all
		complexes as a scatter plot.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	if len( MODELS ) == 3:
		fig, axes = plt.subplots( 3, 2, figsize = ( 15, 15 ) )
	elif len( MODELS ) == 2:
		fig, axes = plt.subplots( 2, 2, figsize = ( 12, 15 ) )
	elif len( MODELS ) == 1:
		fig, axes = plt.subplots( 1, 2, figsize = ( 8, 15 ) )
	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		# Plot histograms for clashscore and Ramachandran favored
		for j, metric in enumerate( ["clash", "favored"] ):
			if len( MODELS ) == 1:
				ax = axes[j]
			else:
				ax = axes[i, j]
			ax.scatter(
				records_native[model][f"mean_{metric}"],
				records_pred[model][f"mean_{metric}"]
			)

			ax.set_title( METHOD_LABELS[model], fontsize = TITLE_SIZE )
			if metric == "clash":
				ax.set_xlim( 0 )
				ax.set_ylim( 0 )
				ax.axhline( 5, c = "red" )
				ax.axvline( 5, c = "red" )
				ax.set_xlabel( f"Clash score (native)", fontsize = XY_LABEL_SIZE )
				ax.set_ylabel( f"Clash score", fontsize = XY_LABEL_SIZE )
			elif metric == "favored":
				ax.set_xlim( 0, 110 )
				ax.set_ylim( 0, 110 )
				ax.axhline( 98, c = "red" )
				ax.axvline( 98, c = "red" )
				ax.set_xlabel( f"Ramachandran favored (native)", fontsize = XY_LABEL_SIZE )
				ax.set_ylabel( f"Ramachandran favored", fontsize = XY_LABEL_SIZE )
			ax.tick_params(
				axis = "both",
				labelsize = 20,
				length = 10,
				width = 4
			)
	plt.tight_layout()
	plt.savefig( plot_file, dpi = 300 )
	plt.close()

################################################################################
def plot_rmsf_scatter( config_name: str ):
	"""
	For all methods, for all complexes, plot the
		per-residue RMSF vs pLDDT as a line plot.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records = prep_rmsf_input( config_name = config_name )
	MAX_ROWS = 6
	MAX_COLS = 6

	for i, model in enumerate( MODELS ):
		fig, ax = plt.subplots( MAX_ROWS, MAX_COLS, figsize = ( 25, 20 ) )
		r, c = 0, 0
		for j, sys_name in enumerate( records[model]["complex"] ):
			ax[r, c].scatter(
				records[model]["plddt"][j],
				records[model]["rmsf"][j]
			)
			ax[r, c].axhline( 75, c = "red" )
			ax[r, c].set_xlim( 0 )
			ax[r, c].set_ylim( 0, 100 )
			ax[r, c].set_xlabel( "per-residue RMSF" )
			ax[r, c].set_ylabel( "pLDDT" )

			c += 1
			if c >= MAX_COLS:
				c = 0
				r += 1
			if r >= MAX_ROWS:
				break

		plt.tight_layout()
		file = os.path.join(
			RMSF_DIR,
		f"{model}_{config_name}.png"
		)
		plt.savefig( file, dpi = 300 )
		plt.close()

################################################################################
def plot_rmsf_contour( config_name: str ):
	"""
	Create a contour plot for analyzing the relationship between
		per-residue RMSF and pLDDT.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records = prep_rmsf_input( config_name = config_name )

	fig, ax = plt.subplots( 1, 3, figsize = ( 25, 10 ) )

	for i, model in enumerate( MODELS ):
		rmsf, plddt = [], []
		for j, sys_name in enumerate( records[model]["complex"] ):
			r = records[model]["rmsf"][j]
			p = records[model]["plddt"][j]

			r = ( r - r.mean( axis = 0 ) )/r.std( axis = 0 )
			p = p/100
			rmsf.append( r.reshape( -1 ) )
			plddt.append( p.reshape( -1 ) )

		rmsf = np.concatenate( rmsf, axis = 0 )
		plddt = np.concatenate( plddt, axis = 0 )

		# Density estimation
		xy = np.vstack( [rmsf, plddt] )
		kde = gaussian_kde( xy )

		xgrid = np.linspace( 0, 1, 200 )
		ygrid = np.linspace( 0, 5, 200 )
		X, Y = np.meshgrid( xgrid, ygrid )

		positions = np.vstack( [X.ravel(), Y.ravel()] )
		Z = np.reshape( kde( positions ).T, X.shape )

		contour = ax[i].contourf( X, Y, Z, levels = 20 )
		ax[i].contour( X, Y, Z, levels = 20, linewidths = 0.5 )

		cbar = plt.colorbar( contour )
		cbar.set_label( "Density" )
		ax[i].set_xlabel( "Per-residue pLDDT" )
		ax[i].set_ylabel( "Per-residue RMSF" )

		plt.tight_layout()
		file = os.path.join(
			RMSF_DIR,
		f"contour_{config_name}.png"
		)
		plt.savefig( file, dpi = 300 )
		plt.close()

################################################################################
def plot_xlsat_v_dockq(
	config_name: str
):
	"""
	Pot the mean/max XL satisfaction vs DockQ across all complexes for
		all specified models.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	if len( MODELS ) == 3:
		fig, axes = plt.subplots( 1, 3, figsize = ( 20, 8 ) )
	elif len( MODELS ) == 2:
		fig, axes = plt.subplots( 1, 2, figsize = ( 15, 8 ) )
	elif len( MODELS ) == 1:
		fig, axes = plt.subplots( 1, 1, figsize = ( 15, 8 ) )

	for i, model in enumerate( MODELS ):
		records_xl = return_metric(
			metric = "xl_sat",
			config_name = config_name
		)
		records_dockq = return_metric(
			metric = "dockq",
			config_name = config_name
		)

		if len( MODELS ) == 1:
			ax = axes
		else:
			ax = axes[i]

		ax.scatter(
			records_xl[model][f"per_model_xl_sat"],
			records_dockq[model][f"per_model_dockq"]
			)
		# ax.axhline( THRESHOLD["xl_sat"], color = "red" )
		# ax.axvline( THRESHOLD["dockq"], color = "red" )
		ax.set_title( METHOD_LABELS[model], fontsize = TITLE_SIZE )
		ax.set_xlabel( f"Per model XL satisfaction", fontsize = XY_LABEL_SIZE )
		ax.set_ylabel( f"Per model DockQ", fontsize = XY_LABEL_SIZE )

		ax.plot( [0, 1], [0, 1], c = "gray" )

		ax.set_xlim( -0.05, 1.05 )
		ax.set_ylim( -0.05, 1.05 )
		ax.tick_params(
			axis = "both",
			labelsize = 16,
			length = 10,
			width = 4
		)
	plt.tight_layout()
	file = os.path.join(
		MEAN_MAX_DIR,
	f"{config_name}_xlsat_v_dockq.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def plot_mean_v_max_metrics(
	config_name: str
):
	"""
	For the following metrics, compare the mean vs max value across all complexes,
		XL satisfaction
		DockQ

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	if len( MODELS ) == 3:
		fig, axes = plt.subplots( 3, 2, figsize = ( 10, 15 ) )
	elif len( MODELS ) == 2:
		fig, axes = plt.subplots( 2, 2, figsize = ( 10, 15 ) )
	elif len( MODELS ) == 1:
		fig, axes = plt.subplots( 1, 2, figsize = ( 10, 7 ) )

	for i, model in enumerate( MODELS ):
		titles = ["XL satisfaction", "DockQ"]
		for j, metric in enumerate( ["xl_sat", "dockq"] ):
			records = return_metric(
				metric = metric,
				config_name = config_name
			)
			if len( MODELS ) == 1:
				ax = axes[j]
			else:
				ax = axes[i, j]

			ax.scatter(
				records[model][f"mean_{metric}"],
				records[model][f"max_{metric}"],
				)
			# if THRESHOLD[metric] is not None:
			# 	ax.axhline( THRESHOLD[metric], color = "red" )
			# 	ax.axvline( THRESHOLD[metric], color = "red" )
			ax.set_title( METHOD_LABELS[model], fontsize = TITLE_SIZE )
			ax.set_xlabel( f"Mean {titles[j]}", fontsize = XY_LABEL_SIZE )
			ax.set_ylabel( f"Max {titles[j]}", fontsize = XY_LABEL_SIZE )

			ax.plot( [0, 1], [0, 1], c = "gray" )

			ax.set_xlim( -0.05, 1.05 )
			ax.set_ylim( -0.05, 1.05 )
			ax.tick_params(
				axis = "both",
				labelsize = 16,
				length = 10,
				width = 4
			)
	plt.tight_layout()
	file = os.path.join(
		MEAN_MAX_DIR,
	f"{config_name}.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def plot_tm_v_dockq(
	config_name: str
):
	"""
	For all methods, plot the mean DockQ wrt native vs mean TM-score wrt native
		for each complex.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	if len( MODELS ) == 3:
		fig, axes = plt.subplots( 3, 2, figsize = ( 10, 15 ) )
	elif len( MODELS ) == 2:
		fig, axes = plt.subplots( 2, 2, figsize = ( 10, 10 ) )
	elif len( MODELS ) == 1:
		fig, axes = plt.subplots( 1, 2, figsize = ( 10, 7 ) )

	for i, model in enumerate( MODELS ):
		for j, agg in enumerate( ["mean", "max"] ):
			records_tm = return_metric(
				metric = "tm",
				config_name = config_name
			)
			records_dockq = return_metric(
				metric = "dockq",
				config_name = config_name
			)

			if len( MODELS ) == 1:
				ax = axes[j]
			else:
				ax = axes[i, j]

			ax.scatter(
				records_tm[model][f"{agg}_tm"],
				records_dockq[model][f"{agg}_dockq"],
				)

			ax.axvline( THRESHOLD["tm"], color = "red" )
			ax.axhline( THRESHOLD["dockq"], color = "red" )
			ax.set_xlim( -0.05, 1.05 )
			ax.set_ylim( -0.05, 1.05 )
			ax.set_title( METHOD_LABELS[model], fontsize = TITLE_SIZE )
			ax.set_xlabel( f"{agg.capitalize()} TM-score", fontsize = XY_LABEL_SIZE )
			ax.set_ylabel( f"{agg.capitalize()} DockQ", fontsize = XY_LABEL_SIZE )

			# ax.plot( [0, 1], [0, 1], c = "gray" )

			ax.tick_params(
				axis = "both",
				labelsize = 16,
				length = 10,
				width = 4
			)
	plt.tight_layout()
	file = os.path.join(
		TM_DOCKQ_DIR,
	f"{config_name}.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
# def plot_restraint_v_sampling():
# 	"""
# 	For Boltz2 only.
# 	We wanna asess the effect of two hyperparameters on structural accuracy (DockQ):
# 		Restraint-guidance: 0 (unguided) and 1 (guided)
# 		Sampling diversity: 1.638 (default; less diverse) and 1.0 (more diverse)
# 	We plot the restraint-guidance vs sampling diversity with the DockQ denoted
# 		by the colour.
# 	"""
# 	model = "boltz2"

# 	for r, config_name1 in zip( [0, 1], ["alpha", "beta1"] ):
# 		records_r = return_metric( config_name = config_name1, metric = "dockq" )
# 		dockq_r = records_r[model]["per_model_dockq"]
# 		for s, config_name2 in zip( [0, 1], ["alpha2", "epsilon1"] ):
# 			records_s = return_metric( config_name = config_name2, metric = "dockq" )
# 			dockq_s = records_s[model]["per_model_dockq"]

# 			plt.scatter( dockq_r, dockq_s, label = f"{config_name1}-{config_name2}" )
# 	plt.legend()
# 	plt.tight_layout()
# 	plt.show()
# 	plt.close()

################################################################################
def plot_cross_configs(
	config1: str,
	config2: str
):
	"""
	Comprae the following metrics for unguided vs guided predictions:
		Mean XL satisfaction
		Mean TM
		Mean DockQ
		Mean MolProbity score
	Plot the mean DockQ ( wrt the native structure) for the
		unguided and guided predictions for all models across
		all complexes as a scatter plot.
	using configs: "alpha", "beta"

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	np.random.seed( 1 )

	MAX_ROWS, MAX_COLS = 2, 3
	agg = "max"

	for model in MODELS:
		fig, ax = plt.subplots( 2, 3, figsize = ( 15, 10 ) )
		titles = [
			f"{agg.capitalize()} XL satisfaction",
			f"{agg.capitalize()} TM-score",
			f"{agg.capitalize()} DockQ",
			"Mean MolProbity score",
			"Mean Clash score",
			"Unique interface"
			]
		r, c = 0, 0
		for i, metric in enumerate(
			[
				"xl_sat", "tm", "dockq", "molprob-score",
				"molprob-clash", "unique_interface"
			]
		):
			if "molprob" in metric:
				metric, sub_metric = metric.split( "-" )
			records1 = return_metric(
				metric = metric,
				config_name = config1
			)
			records2 = return_metric(
				metric = metric,
				config_name = config2
			)
			if "molprob" in metric:
				if sub_metric == "score":
					key = f"mean_molprob"
				elif sub_metric == "clash":
					key = f"mean_clash"
			elif metric in ["unique_struct", "unique_interface"]:
				# Adding an eps noise to differential points with the same value.
				key = metric
				eps = np.random.normal( 0, 0.2, len( records2[model][key] ) )
			else:
				key = f"{agg}_{metric}"
				eps = np.array( [0]*len( records2[model][key] ) )
			ax[r,c].scatter(
				records1[model][key] + eps,
				records2[model][key] + eps,
				)

			ax[r,c].set_title( titles[i], fontsize = TITLE_SIZE )
			ax[r,c].set_xlabel( f"{config1}", fontsize = XY_LABEL_SIZE )
			ax[r,c].set_ylabel( f"{config2}", fontsize = XY_LABEL_SIZE )
			if metric in ["unique_struct", "unique_interface"]:
				ax[r,c].plot( [0, 26], [0, 26], c = "gray" )
			elif "molprob" in metric:
				if sub_metric == "score":
					ax[r,c].plot( [0, 5], [0, 5], c = "gray" )
				elif sub_metric == "clash":
					max_y1 = max( records1[model][key] )
					max_y2 = max( records2[model][key] )
					ax[r,c].plot( [0, max_y1], [0, max_y2], c = "gray" )
			else:
				ax[r,c].plot( [0, 1], [0, 1], c = "gray" )

			ax[r, c].tick_params(
				axis = "both",
				labelsize = 16,
				length = 10,
				width = 4
			)
			c += 1
			if c >= MAX_COLS:
				c = 0
				r += 1
			if r >= MAX_ROWS:
				break


		plt.tight_layout()
		file = os.path.join(
			CROSS_CONFIG_DIR,
		f"{model}_{config1}_{config2}_{agg}.png"
		)
		plt.savefig( file, dpi = 300 )
		plt.close()

################################################################################
################################################################################
def plot( config_name: str ):
	plot_xl_satisfaction_across_models( config_name = config_name )

	plot_native_tm_across_models( config_name = config_name )

	plot_native_dockq_across_models( config_name = config_name )

	plot_tm_v_dockq( config_name = config_name )

	plot_unique_models_across_models( config_name = config_name )

	plot_molprobity_metrics_across_models( config_name = config_name )

	plot_xlsat_v_dockq( config_name = config_name )

	plot_mean_v_max_metrics( config_name = config_name )

	plot_rmsf_scatter( config_name = config_name )

	# plot_rmsf_contour( config_name = config_name )

if __name__ == "__main__":
	# MODELS = ["boltz2"]
	# plot_restraint_v_sampling()
	MODELS = ["alphalink2", "boltz2", "grasp"]
	plot_pred_time_taken_across_models( config_name = "beta1" )
	# plot_molprobity_metrics_across_models( config_name = "beta1" )
	print( f"\nUsing: {MODELS} " + "-"*20 )
	# Per-config plots.
	for config_name in ["beta1"]:
		print( f"Creating plots for config: {config_name}..." )
		plot( config_name = config_name )

	# Following configs are specific for Boltz2 and GRASP
	MODELS = ["boltz2", "grasp"]
	print( f"\nUsing: {MODELS} " + "-"*20 )
	for config_name in ["alpha", "beta2", "beta3", "gamma"]:
		print( f"Creating plots for config: {config_name}..." )
		plot( config_name = config_name )

	print( "\nCreating cross-config plots..." )
	for config1, config2 in [
		["alpha", "beta1"],
		["beta1", "beta2"],
		["beta1", "beta3"],
	]:
		# Cross-config plots
		print( f"{config1}-{config2}..." )
		plot_cross_configs(
			config1 = config1,
			config2 = config2
		)

	# Following configs are specific for Boltz2
	MODELS = ["boltz2"]
	print( f"\nUsing: {MODELS} " + "-"*20 )
	for config_name in ["beta4", "delta1", "delta2", "delta3", "epsilon1", "epsilon2", "epsilon3", "zeta1", "theta1", "iota1"]:
		print( f"Creating plots for config: {config_name}..." )
		plot( config_name = config_name )

	# Cross-config plots
	print( "\nCreating cross-config plots..." )
	for config1, config2 in [
		["alpha", "alpha2"],
		["alpha", "beta4"],
		["alpha", "delta1"],
		["alpha", "delta2"],
		["alpha", "delta3"],
		["alpha", "epsilon1"],
		["alpha", "zeta1"],
		["alpha", "theta1"],
		["alpha2", "beta4"],
		["alpha2", "epsilon1"],
		# --------------------
		["beta1", "beta4"],
		["beta1", "delta1"],
		["beta1", "delta2"],
		["beta1", "delta3"],
		["beta1", "epsilon1"],
		["beta2", "epsilon2"],
		["beta3", "epsilon3"],
		["beta1", "zeta1"],
		["beta1", "theta1"],
		# --------------------
		["epsilon1", "epsilon2"],
		["epsilon1", "epsilon3"],
		["epsilon1", "epsilon4"],
		# --------------------
		["epsilon1", "iota1"],
		["delta1", "iota1"],
	]:
		print( f"{config1}-{config2}..." )
		plot_cross_configs(
			config1 = config1,
			config2 = config2
		)

