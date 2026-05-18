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
XY_LABEL_SIZE = 25
TITLE_SIZE = 25

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
	# for model in ["alphalink2", "boltz2", "grasp"]:
	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", "mean_xl_sat", "max_xl_sat"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			xl_sat = data[sys_name]["xl_metrics"]["xl_satisfaction"]

			records[model]["complex"].append( sys_name )
			records[model]["mean_xl_sat"].append( xl_sat.mean() )
			records[model]["max_xl_sat"].append( xl_sat.max() )
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
			k:[] for k in ["complex", "mean_tm", "max_tm"]
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
	# for model in ["alphalink2", "boltz2", "grasp"]:
	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", "mean_dockq", "max_dockq"]
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
	# for model in ["alphalink2", "boltz2", "grasp"]:
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
def prep_molprobity_input(
	config_name: str
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
	# for model in ["alphalink2", "boltz2", "grasp"]:
	for model in MODELS:
		records[model] = {
			k:[] for k in ["complex", "mean_molprob", "resolution"]
			}
		data = load_analysis_dict(
			model = model,
			config_name = config_name
		)

		for sys_name in data:
			if sys_name in IGNORE_SYSTEMS:
				continue
			molprob = np.array(
				[v["MolProbity score"] for k,v in data[sys_name]["molprob"].items()]
				)
			records[model]["complex"].append( sys_name )
			records[model]["mean_molprob"].append( molprob.mean() )
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
			# print( rmsf.shape, "  ", plddt.shape )

			records[model]["complex"].append( sys_name )
			records[model]["rmsf"].append( rmsf )
			records[model]["plddt"].append( plddt )
	return records

################################################################################
def return_metric(
	metric: str,
	config_name: str
) -> Dict[str, List]:
	"""
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
	fig, ax = plt.subplots( 1, 1, figsize = ( 20, 10 ) )
	widths = [-0.25, 0, 0.25]

	for i, model in enumerate( MODELS ):
		X = np.arange( 0, len( records[model]["complex"] ), 1 )
		ax.bar(
			X+widths[i],
			records[model]["time"], 0.1, # widths[i],
			color = COLOR[model],
			label = f"{METHOD_LABELS[model]}"
		)

		ax.set_xticks( X )
		ax.tick_params( axis = "both", width = 2, length = 5 )
		ax.set_xticklabels(
			records[model]["complex"], rotation = 90, fontsize = 16
			)
		ax.set_ylabel( f"Time taken (hours)", fontsize = XY_LABEL_SIZE )
		ax.set_xlabel( "Complexes", fontsize = XY_LABEL_SIZE )
		ax.legend( title = "Method" )
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

	fig, ax = plt.subplots( 2, 1, figsize = ( 25, 20 ) )
	for i, agg in enumerate( ["mean", "max"] ):
		# Create a scatter plot for each method.
		for j, model in enumerate( MODELS ):
			X = np.arange( 0, len( records[model]["complex"] ), 1 )
			ax[i].scatter(
				X,
				np.array( records[model][f"{agg}_xl_sat"] ), # + 0.01*( i+1 )
				c = COLOR[model],
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
		ax[i].legend( title = "Method" )
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

	fig, ax = plt.subplots( 2, 1, figsize = ( 25, 20 ) )
	for i, agg in enumerate( ["mean", "max"] ):
		# Create a scatter plot for each method.
		for j, model in enumerate( MODELS ):
			X = np.arange( 0, len( records[model]["complex"] ), 1 )
			ax[i].scatter( X, records[model][f"{agg}_dockq"],
			c = COLOR[model], label = METHOD_LABELS[model]
			)

		ax[i].axhline( THRESHOLD["tm"], color = "red" )
		ax[i].set_xticks( X )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels(
			records[model]["complex"], rotation = 90, fontsize = 12
			)
		ax[i].set_ylabel( f"{agg.capitalize()} TM-score", fontsize = XY_LABEL_SIZE )
		ax[i].set_xlabel( "Complexes", fontsize = XY_LABEL_SIZE )
		ax[i].set_ylim( 0.0, 1.1 )
		ax[i].legend( title = "Method" )
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

	fig, ax = plt.subplots( 2, 1, figsize = ( 25, 20 ) )
	for i, agg in enumerate( ["mean", "max"] ):
		# Create a scatter plot for each method.
		for j, model in enumerate( MODELS ):
			X = np.arange( 0, len( records[model]["complex"] ), 1 )
			ax[i].scatter( X, records[model][f"{agg}_dockq"],
			c = COLOR[model], label = METHOD_LABELS[model]
			)

		ax[i].axhline( 0.23, color = "red" )
		ax[i].set_xticks( X )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels(
			records[model]["complex"], rotation = 90, fontsize = 12
			)
		ax[i].set_ylabel( f"{agg.capitalize()} DockQ", fontsize = XY_LABEL_SIZE )
		ax[i].set_xlabel( "Complexes", fontsize = XY_LABEL_SIZE )
		ax[i].set_ylim( 0.0, 1.1 )
		ax[i].legend( title = "Method" )
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

	fig, ax = plt.subplots( 2, 1, figsize = ( 25, 20 ) )
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
			ax[j].set_ylim( 0.0, 25+1 )
			ax[j].legend( title = "Method" )
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
def plot_molprobity_across_models(
	config_name: str
):
	"""
	Plot the mean Molprobity score for all models
		across all complexes as a scatter plot.

	Inputs:
	----------
	config_name: str identifier for the model configuration
		used for prediction.
		See model_configs.py.
	"""
	records = prep_molprobity_input( config_name = config_name )

	fig, ax = plt.subplots( 1, 3, figsize = ( 25, 10 ) )
	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.arange( 0, len( records[model]["complex"] ), 1 )
		ax[i].scatter(
			records[model]["resolution"],
			records[model]["mean_molprob"],
			c = COLOR[model], label = METHOD_LABELS[model]
			)

		ax[i].plot( [0, 4], [0, 4], c = "gray" )
		ax[i].set_title( METHOD_LABELS[model], fontsize = TITLE_SIZE )
		ax[i].set_ylabel( f"MolProbity score", fontsize = XY_LABEL_SIZE )
		ax[i].set_xlabel( "Experimental resolution", fontsize = XY_LABEL_SIZE )
		ax[i].tick_params(
			axis = "both",
			labelsize = 20,
			length = 10,
			width = 4
		)

	plt.tight_layout()
	file = os.path.join( MOLPROB_DIR, f"molprobity_{config_name}.png" )
	plt.savefig( file, dpi = 300 )
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
		fig, axes = plt.subplots( 3, 2, figsize = ( 20, 20 ) )
	elif len( MODELS ) == 2:
		fig, axes = plt.subplots( 2, 2, figsize = ( 15, 15 ) )
	elif len( MODELS ) == 1:
		fig, axes = plt.subplots( 1, 2, figsize = ( 15, 8 ) )

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
			if THRESHOLD[metric] is not None:
				ax.axhline( THRESHOLD[metric], color = "red" )
				ax.axvline( THRESHOLD[metric], color = "red" )
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
		fig, axes = plt.subplots( 3, 2, figsize = ( 20, 20 ) )
	elif len( MODELS ) == 2:
		fig, axes = plt.subplots( 2, 2, figsize = ( 15, 15 ) )
	elif len( MODELS ) == 1:
		fig, axes = plt.subplots( 1, 2, figsize = ( 15, 8 ) )

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
	agg = "mean"

	for model in MODELS:
		fig, ax = plt.subplots( 2, 3, figsize = ( 25, 20 ) )
		titles = ["XL satisfaction", "TM-score", "DockQ", "MolProbity score", "Number of unique structure", "Number of unique interface"]
		r, c = 0, 0
		for i, metric in enumerate( ["xl_sat", "tm", "dockq", "molprob", "unique_struct", "unique_interface"] ):
			records1 = return_metric(
				metric = metric,
				config_name = config1
			)
			records2 = return_metric(
				metric = metric,
				config_name = config2
			)
			if metric in ["unique_struct", "unique_interface"]:
				# Adding an epx noise to differential points with the same value.
				key = metric
				eps = np.random.normal( 0, 0.2, len( records2[model][key] ) )
			else:
				key = f"{agg}_{metric}"
				eps = np.array( [0]*len( records2[model][key] ) )
			ax[r,c].scatter(
				records1[model][key] + eps,
				records2[model][key] + eps,
				)
			if THRESHOLD[metric] is not None:
				ax[r,c].axhline( THRESHOLD[metric], color = "red" )
				ax[r, c].axvline( THRESHOLD[metric], color = "red" )
			ax[r,c].set_title( titles[i], fontsize = TITLE_SIZE )
			ax[r,c].set_xlabel( f"{config1}", fontsize = XY_LABEL_SIZE )
			ax[r,c].set_ylabel( f"{config2}", fontsize = XY_LABEL_SIZE )
			if metric in ["unique_struct", "unique_interface"]:
				ax[r,c].plot( [0, 26], [0, 26], c = "gray" )
			elif metric == "molprob":
				ax[r,c].plot( [0, 4], [0, 4], c = "gray" )
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

	plot_molprobity_across_models( config_name = config_name )

	plot_mean_v_max_metrics( config_name = config_name )

	plot_rmsf_scatter( config_name = config_name )

	# plot_rmsf_contour( config_name = config_name )

if __name__ == "__main__":

	MODELS = ["alphalink2", "boltz2", "grasp"]
	plot_pred_time_taken_across_models( config_name = "beta1" )
	print( f"\nUsing: {MODELS} " + "-"*20 )
	# Per-config plots.
	for config_name in ["beta1"]:
		print( f"Creating plots for config: {config_name}..." )
		plot( config_name = config_name )

	# Following configs are specific for Boltz2 and GRASP
	MODELS = ["boltz2", "grasp"]
	print( f"\nUsing: {MODELS} " + "-"*20 )
	print( f"Creating plots for config: {config_name}..." )
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
	for config_name in ["delta1", "delta2", "delta3", "epsilon1", "zeta1", "theta1"]:
		print( f"Creating plots for config: {config_name}..." )
		plot( config_name = config_name )

	# Cross-config plots
	print( "\nCreating cross-config plots..." )
	for config1, config2 in [
		["alpha", "delta1"],
		["alpha", "delta2"],
		["alpha", "delta3"],
		["alpha", "epsilon1"],
		["alpha", "zeta1"],
		["alpha", "theta1"],
		["beta1", "delta1"],
		["beta1", "delta2"],
		["beta1", "delta3"],
		["beta1", "epsilon1"],
		["beta1", "zeta1"],
		["beta1", "theta1"]
	]:
		print( f"{config1}-{config2}..." )
		plot_cross_configs(
			config1 = config1,
			config2 = config2
		)

