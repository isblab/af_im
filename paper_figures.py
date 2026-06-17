"""
Methods to create figures for the paper.
"""
from typing import List, Dict, Any
import os, warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_config_dict

from utils.analysis_utils import return_metric
from analysis.stats import return_stats_for_metric
from utils.paths import (
	get_benchmark_analysis_dir_path
)
warnings.filterwarnings( "ignore" )

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
	ANALYSIS_DIR, "paper_figures"
)
for d in [FIG_DIR]:
	os.makedirs( d, exist_ok = True )

################################################################################
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

################################################################################
################################################################################
def figure1a():
	"""
	Categorize max data satisfaction into : low, medium, and high.
	Create a bar plot depicting no. of complexes in each category for all methods.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( figsize = ( 10, 7 ) )
	plt.rcParams["font.family"] = "sans-serif"

	bar_width = 0.25
	offsets = [-bar_width, 0, bar_width]

	levels = ["low", "medium", "high"]
	X = np.arange( len( levels ) )

	for i, model in enumerate( MODELS ):
		stats = return_stats_for_metric(
			model = model,
			config_name = config_name,
			metric = "xl_sat"
		)
		counts = list( stats.values() )

		bars = ax.bar(
			X + offsets[i],
			counts,
			width = bar_width,
			color = COLOR[model],
			label = METHOD_LABELS[model]
		)
		ax.bar_label(
			bars,
			labels = [str(v) for v in counts],
			padding = 3,
			fontsize = 14,
			rotation = 0
		)

	ax.set_xticks( X )
	ax.set_xticklabels(
		[l.capitalize() for l in levels], rotation = 90, fontsize = 16
		)
	ax.set_xlabel("Data satisfaction", fontsize = 16 )
	ax.set_ylabel("No. of complexes", fontsize = 16)
	ax.set_ylim( 0, 40 )
	ax.tick_params(
		axis = "both",
		labelsize = 16,
		length = 10,
		width = 2,
		rotation = 0
	)
	ax.legend( fontsize = 14 )

	plt.tight_layout()
	file = os.path.join(FIG_DIR, "figure_1a.png")
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def figure1b():
	"""
	Categorize max DockQ wrt native into: low, medium, and high.
	Create a bar plot depicting no. of complexes in each category for all methods.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( figsize = ( 10, 7 ) )
	plt.rcParams["font.family"] = "sans-serif"

	bar_width = 0.25
	offsets = [-bar_width, 0, bar_width]

	levels = ["low", "medium", "high"]
	X = np.arange( len( levels ) )

	for i, model in enumerate( MODELS ):
		stats = return_stats_for_metric(
			model = model,
			config_name = config_name,
			metric = "dockq"
		)
		counts = list( stats.values() )

		bars = ax.bar(
			X + offsets[i],
			counts,
			width = bar_width,
			color = COLOR[model],
			label = METHOD_LABELS[model]
		)
		ax.bar_label(
			bars,
			labels = [str(v) for v in counts],
			padding = 3,
			fontsize = 14,
			rotation = 0
		)

	ax.set_xticks( X )
	ax.set_xticklabels(
		[l.capitalize() for l in levels], rotation = 90, fontsize = 16
		)
	ax.set_xlabel("Data satisfaction", fontsize = 16 )
	ax.set_ylabel("No. of complexes", fontsize = 16)
	ax.set_ylim( 0, 40 )
	ax.tick_params(
		axis = "both",
		labelsize = 16,
		length = 10,
		width = 2,
		rotation = 0
	)
	ax.legend( fontsize = 14 )

	plt.tight_layout()
	file = os.path.join( FIG_DIR, "figure_1b.png" )
	plt.savefig( file, dpi = 300 )
	plt.close()


if __name__ == "__main__":
	figure1a()
	figure1b()

