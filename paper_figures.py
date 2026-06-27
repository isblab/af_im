"""
Methods to create figures for the paper.
"""
from typing import List, Dict, Any
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from config import get_config_dict

from utils.analysis_utils import return_metric, prep_native_dockq_input
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
	"alphalink2": "#8A62FF",
	"boltz2": "#41D041FE",
	"grasp": "#FFB340",
}
MARKER = {
	"alphalink2": "s",
	"boltz2": "v",
	"grasp": "o"
}

################################################################################
################################################################################
def fig_2a():
	"""
	Plot the max XL satisfaction for all models
		across all complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 12, 6 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		metric = "xl_sat"
	)

	# Create a scatter plot for each method.
	for idx, model in enumerate( MODELS ):
		X = np.arange( 0, len( records[model]["complex"] ), 1 )
		Y = np.array( records[model]["max_xl_sat"] )

		# Grey bands
		for i in X[::2]:
			ax.axvspan(
				i - 0.5,
				i + 0.5,
				color = "#DDDDDD",
				alpha = 0.3,
				zorder = 0
			)

		ax.scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
			label = METHOD_LABELS[model]
		)

	ax.axhline( 0.75, color = "red" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"Max XL satisfaction", fontsize = 14 )
	ax.set_xlabel( "Complexes", fontsize = 14 )
	ax.set_ylim( -0.02, 1.02 )
	ax.legend( loc = "lower right" )
	ax.tick_params(
		axis = "both",
		labelsize = 12,
		length = 8,
		width = 2
	)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_2a.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_2b():
	"""
	For guided vs unguided predictions, plot max
		XL satisfaction for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	fig, ax = plt.subplots( 1, 2, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		metric = "xl_sat"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		metric = "xl_sat"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["max_xl_sat"] )
		Y = np.array( records2[model]["max_xl_sat"] )
		stat, p = wilcoxon( X, Y, alternative = "greater" )
		p = round( p, 4 )

		ax[i].scatter(
			X, Y,
			s = 25,
			c = COLOR[model],
			marker = MARKER[model],
		)
		ax[i].text(
			0.02, 0.98,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].plot( [-0.02, 1.02], [-0.02, 1.02] )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"Max XL satisfaction (guided)", fontsize = 12 )
		ax[i].set_ylabel( f"Max XL satisfaction (unguided)", fontsize = 12 )
		ax[i].axis( [-0.02, 1.02, -0.02, 1.02] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_2b.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_2c():
	"""
	For guided vs unguided predictions, plot the
		mean XL satisfaction for all models
		across all complexes as a scatter plot.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	fig, ax = plt.subplots( 1, 2, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		metric = "xl_sat"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		metric = "xl_sat"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["mean_xl_sat"] )
		Y = np.array( records2[model]["mean_xl_sat"] )
		stat, p = wilcoxon( X, Y, alternative = "greater" )
		p = round( p, 4 )

		ax[i].scatter(
			X, Y,
			s = 25,
			c = COLOR[model],
			marker = MARKER[model],
			# label = METHOD_LABELS[model]
		)
		ax[i].text(
			0.02, 0.98,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].plot( [-0.02, 1.02], [-0.02, 1.02] )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"Mean XL satisfaction (guided)", fontsize = 12 )
		ax[i].set_ylabel( f"Mean XL satisfaction (unguided)", fontsize = 12 )
		ax[i].axis( [-0.02, 1.02, -0.02, 1.02] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 10,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_2c.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
################################################################################
def fig_3a():
	"""
	Plot the max DockQ for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 12, 6 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		metric = "dockq"
	)

	# Create a scatter plot for each method.
	for idx, model in enumerate( MODELS ):
		X = np.arange( 0, len( records[model]["complex"] ), 1 )
		Y = np.array( records[model]["max_dockq"] )

		# Grey bands
		for i in X[::2]:
			ax.axvspan(
				i - 0.5,
				i + 0.5,
				color = "#DDDDDD",
				alpha = 0.3,
				zorder = 0
			)

		ax.scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
			label = METHOD_LABELS[model]
		)

	ax.axhline( 0.23, color = "red" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"Max DockQ", fontsize = 14 )
	ax.set_xlabel( "Complexes", fontsize = 14 )
	ax.set_ylim( -0.02, 1.02 )
	ax.legend( loc = "lower left" )
	ax.tick_params(
		axis = "both",
		labelsize = 12,
		length = 8,
		width = 2
	)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_3a.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_3b():
	"""
	For guided vs unguided predictions, plot the
		mean DockQ for all models
		across all complexes as a scatter plot.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	fig, ax = plt.subplots( 1, 2, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		metric = "dockq"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		metric = "dockq"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["mean_dockq"] )
		Y = np.array( records2[model]["mean_dockq"] )
		stat, p = wilcoxon( X, Y, alternative = "greater" )
		p = round( p, 4 )

		np.random.seed( 3407 )
		eps = np.random.normal( 0, 0.001, len( X ) )
		ax[i].scatter(
			X+eps, Y+eps,
			s = 25,
			c = COLOR[model],
			marker = MARKER[model],
		)
		ax[i].text(
			0.02, 0.98,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].plot( [-0.02, 1.02], [-0.02, 1.02] )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"DockQ wrt native (guided)", fontsize = 12 )
		ax[i].set_ylabel( f"DockQ wrt native (unguided)", fontsize = 12 )
		ax[i].axis( [-0.02, 1.02, -0.02, 1.02] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_3b.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_3c():
	"""
	Plot the Molprobity score for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 3, figsize = ( 14, 6 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		config_name = config_name,
		models = MODELS,
		metric = "molprob",
		molprob_native = False
		)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records[model]["resolution"] )
		Y = np.array( records[model]["mean_molprob"] )
		stat, p = wilcoxon( X.reshape( -1 ), Y.reshape( -1 ), alternative = "greater" )
		p = round( p, 4 )

		ax[i].scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
		)
		ax[i].text(
			0.02, 2.98,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].plot( [0, 3], [0, 3] )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_ylabel( f"Mean molProbity score", fontsize = 14 )
		ax[i].set_xlabel( "Experimental resolution", fontsize = 14 )
		ax[i].axis( [0.0, 3.0, 0.0, 3.0] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 12,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_3c.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_3d():
	"""
	Plot the clashscore for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 3, figsize = ( 14, 6 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records_pred = return_metric(
		config_name = config_name,
		models = MODELS,
		metric = "molprob",
		molprob_native = False
		)
	records_native = return_metric(
		config_name = config_name,
		models = MODELS,
		metric = "molprob",
		molprob_native = True
		)
	# q2_xy, q3_xy = {}, {}
	# for model in MODELS:
	# 	if model == "grasp":
	# 		q2_xy[model] = [2, 100]
	# 		q3_xy[model] = [15, 100]
	# 	else:
	# 		q2_xy[model] = [2, 25]
	# 		q3_xy[model] = [15, 25]
	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records_native[model]["mean_clash"] )
		Y = np.array( records_pred[model]["mean_clash"] )

		# j = np.where( X <=5 )
		# k = np.where( X >=5 )
		# Q1 = np.count_nonzero( np.where( Y[j] <=5, 1, 0 ) )
		# Q2 = np.count_nonzero( np.where( Y[j] >=5, 1, 0 ) )
		# Q3 = np.count_nonzero( np.where( Y[k] >=5, 1, 0 ) )

		stat, p = wilcoxon( X.reshape( -1 ), Y.reshape( -1 ), alternative = "greater" )
		p = round( p, 4 )

		ax[i].scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
			# label = METHOD_LABELS[model]
		)
		# # Add counts
		# ax[i].text(
		# 	q2_xy[model][0], q2_xy[model][1],
		# 	f"{Q2}",
		# 	ha = "right", va = "top",
		# 	fontsize = 16
		# )
		# ax[i].text(
		# 	q3_xy[model][0], q3_xy[model][1],
		# 	f"{Q3}",
		# 	ha = "right", va = "top",
		# 	fontsize = 16
		# )
		ax[i].axvline( 5.0, color = "red" )
		ax[i].axhline( 5.0, color = "red" )
		x_offset = X.max()
		y_offset = 30 if i <=1 else 125
		ax[i].text(
			x_offset,
			y_offset-1 if i <=1 else y_offset-4,
			f"p-value = {p}",
			ha = "right", va = "top",
			fontsize = 10
		)

		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_ylabel( f"Mean Clashscore (prediction)", fontsize = 14 )
		ax[i].set_xlabel( "Mean Clashscore (native)", fontsize = 14 )
		ax[i].axis( [-1, x_offset+1, 0, y_offset] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 12,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_3d.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
################################################################################
def fig_4a():
	"""
	Plot the no. of unique models based on interface similarity for
		all models across all complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 12, 6 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		metric = "unique_interface"
	)

	# Create a scatter plot for each method.
	for idx, model in enumerate( MODELS ):
		X = np.arange( 0, len( records[model]["complex"] ), 1 )
		Y = np.array( records[model]["unique_interface"] )

		# Grey bands
		for i in X[::2]:
			ax.axvspan(
				i - 0.5,
				i + 0.5,
				color = "#DDDDDD",
				alpha = 0.3,
				zorder = 0
			)

		ax.scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
			label = METHOD_LABELS[model]
		)

	ax.axhline( 1, color = "red" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"No. of unique structures", fontsize = 14 )
	ax.set_xlabel( "Complexes", fontsize = 14 )
	ax.set_ylim( -0.5, 25 )
	ax.legend( loc = "upper left" )
	ax.tick_params(
		axis = "both",
		labelsize = 12,
		length = 8,
		width = 2
	)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_4a.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_4b():
	"""
	For guided vs unguided predictions, plot the no. of
		unique models based on interface similarity for
		all models across all complexes.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	fig, ax = plt.subplots( 1, 2, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		metric = "unique_interface"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		metric = "unique_interface"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["unique_interface"] )
		Y = np.array( records2[model]["unique_interface"] )
		stat, p = wilcoxon( X, Y, alternative = "two-sided" )
		p = round( p, 4 )

		np.random.seed( 3407 )
		eps = np.random.normal( 0, 0.2, len( X ) )
		ax[i].scatter(
			X+eps, Y+eps,
			s = 25,
			c = COLOR[model],
			marker = MARKER[model],
		)
		ax[i].text(
			0.02, 24.5,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].plot( [-0.5, 25], [-0.5, 25] )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"No. of unique structures (guided)", fontsize = 12 )
		ax[i].set_ylabel( f"No. of unique structures (unguided)", fontsize = 12 )
		ax[i].axis( [-0.5, 25, -0.5, 25] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_4b.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_4c():
	"""
	Plot the no. of unique models based on interface similarity vs the
		max model confidence for all models across all complexes.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 3, figsize = ( 14, 6 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name,
		metric = "unique_interface"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name,
		metric = "confidence"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["unique_interface"] )
		# X /= 25
		Y = np.array( records2[model]["max_confidence"] )
		stat, p = wilcoxon( X, Y, alternative = "two-sided" )
		p = round( p, 4 )

		# np.random.seed( 3407 )
		# eps = np.random.normal( 0, 0.2, len( X ) )
		ax[i].scatter(
			X, Y,
			s = 25,
			c = COLOR[model],
			marker = MARKER[model],
			# label = METHOD_LABELS[model]
		)
		ax[i].text(
			19, 1.08,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"No. of unique structures", fontsize = 14 )
		ax[i].set_ylabel( f"Max model confidence", fontsize = 14 )
		ax[i].axis( [-0.5, 25, -0.5, 1.1] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_4c.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()
################################################################################
################################################################################
def fig_5a():
	"""
	Plot the max confidence score for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 12, 6 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		metric = "confidence"
	)

	# Create a scatter plot for each method.
	for idx, model in enumerate( MODELS ):
		X = np.arange( 0, len( records[model]["complex"] ), 1 )
		Y = np.array( records[model]["max_confidence"] )

		# Grey bands
		for i in X[::2]:
			ax.axvspan(
				i - 0.5,
				i + 0.5,
				color = "#DDDDDD",
				alpha = 0.3,
				zorder = 0
			)

		ax.scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
			label = METHOD_LABELS[model]
		)

	ax.axhline( 0.75, color = "red" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"Max model confidence", fontsize = 14 )
	ax.set_xlabel( "Complexes", fontsize = 14 )
	ax.set_ylim( -0.02, 1.02 )
	ax.legend( loc = "lower left" )
	ax.tick_params(
		axis = "both",
		labelsize = 12,
		length = 8,
		width = 2
	)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_5a.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_5b():
	"""
	For guided vs unguided predictions, plot the
		max confidence for all models
		across all complexes as a scatter plot.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	fig, ax = plt.subplots( 1, 2, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		metric = "confidence"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		metric = "confidence"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["max_confidence"] )
		Y = np.array( records2[model]["max_confidence"] )
		stat, p = wilcoxon( X, Y, alternative = "greater" )
		p = round( p, 4 )

		np.random.seed( 3407 )
		eps = np.random.normal( 0, 0.001, len( X ) )
		ax[i].scatter(
			X+eps, Y+eps,
			s = 25,
			c = COLOR[model],
			marker = MARKER[model],
		)
		ax[i].text(
			0.02, 0.98,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].plot( [-0.02, 1.02], [-0.02, 1.02] )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"Max model confidence (guided)", fontsize = 12 )
		ax[i].set_ylabel( f"Max model confidence (unguided)", fontsize = 12 )
		ax[i].axis( [-0.02, 1.02, -0.02, 1.02] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_5b.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_5c():
	"""
	For guided vs unguided predictions, plot the
		mean confidence for all models
		across all complexes as a scatter plot.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	fig, ax = plt.subplots( 1, 2, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		metric = "confidence"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		metric = "confidence"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["mean_confidence"] )
		Y = np.array( records2[model]["mean_confidence"] )
		stat, p = wilcoxon( X, Y, alternative = "greater" )
		p = round( p, 4 )

		np.random.seed( 3407 )
		eps = np.random.normal( 0, 0.001, len( X ) )
		ax[i].scatter(
			X+eps, Y+eps,
			s = 25,
			c = COLOR[model],
			marker = MARKER[model],
		)
		ax[i].text(
			0.02, 0.98,
			f"p-value = {p:.3g}",
			ha = "left", va = "top",
			fontsize = 10
		)

		ax[i].plot( [-0.02, 1.02], [-0.02, 1.02] )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"Mean model confidence (guided)", fontsize = 12 )
		ax[i].set_ylabel( f"Mean model confidence (unguided)", fontsize = 12 )
		ax[i].axis( [-0.02, 1.02, -0.02, 1.02] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_5c.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
################################################################################
# def fig_10():
# 	"""
# 	Plot data satisfaction for multistate benchmark.
# 	"""
# 	MODEL = "boltz2"

# 	fig, ax = plt.subplots( 1, 3, figsize = ( 7, 4 ) )
# 	plt.rcParams["font.family"] = "sans-serif"

# 	states = ["State1", "State2", "State1+2"]
# 	for i, config_name in enumerate( ["kappa1", "kappa2", "kappa3"] ):
# 		records = return_metric(
# 			models = [MODEL],
# 			config_name = config_name,
# 			metric = "xl_sat"
# 		)
# 		X = np.arange( 0, len( records[MODEL]["complex"] ), 1 )
# 		Y = np.array( records[MODEL]["max_xl_sat"] )

# 		# Grey bands
# 		for j in X[::2]:
# 			ax[i].axvspan(
# 				j - 0.5,
# 				j + 0.5,
# 				color = "#DDDDDD",
# 				alpha = 0.3,
# 				zorder = 0
# 			)

# 		ax[i].scatter(
# 			X, Y,
# 			s = 50,
# 			c = COLOR[MODEL],
# 			marker = MARKER[MODEL],
# 			label = METHOD_LABELS[MODEL]
# 		)

# 		ax[i].axhline( 0.75, color = "red" )
# 		ax[i].set_xticks( X )
# 		ax[i].tick_params(axis = "both", width = 2, length = 5 )
# 		ax[i].set_xticklabels(
# 			records[MODEL]["complex"], rotation = 90
# 		)
# 		ax[i].set_ylabel( f"Max XL satisfaction", fontsize = 14 )
# 		ax[i].set_xlabel( "Complexes", fontsize = 14 )
# 		ax[i].set_ylim( -0.1, 1.1 )

# 		ax[i].tick_params(
# 			axis = "both",
# 			labelsize = 12,
# 			length = 8,
# 			width = 2
# 		)
# 	plt.legend( loc = "lower right" )
# 	plt.tight_layout()
# 	file = os.path.join(
# 		FIG_DIR, f"figure_3a_kappa.png"
# 	)
# 	plt.savefig( file, dpi = 300 )
# 	plt.close()


# def fig_11():
# 	"""
# 	Plot dockq wrt state1 and state2 for multistate benchmark.
# 	Create a scatter plot for Dockq wrt state1 vs Dockq wrt state2.
# 	"""
# 	MODEL = "boltz2"

# 	fig, ax = plt.subplots( 1, 3, figsize = ( 7, 4 ) )
# 	plt.rcParams["font.family"] = "sans-serif"

# 	states = ["State1", "State2", "State1+2"]
# 	for i, config_name in enumerate( ["kappa1", "kappa2", "kappa3"] ):
# 		records_s1 = prep_native_dockq_input(
# 			config_name = config_name,
# 			native_model_id = 1000,
# 			models = [MODEL]
# 		)
# 		records_s2 = prep_native_dockq_input(
# 			config_name = config_name,
# 			native_model_id = 1000,
# 			models = [MODEL]
# 		)

# 		dockq_s1 = records_s1[MODEL]["per_model_dockq"]
# 		dockq_s2 = records_s2[MODEL]["per_model_dockq"]

# 		ax[i].scatter(
# 			dockq_s1, dockq_s2
# 		)
# 		# ax[i].set_xticks( X )
# 		ax[i].set_xlim( 0, 1 )
# 		ax[i].set_ylim( 0, 1.1 )
# 		ax[i].tick_params(axis = "both", width = 2, length = 5 )
# 		ax[i].set_xticklabels(
# 			records_s1[MODEL]["complex"], rotation = 90
# 		)
# 		ax[i].set_xlabel( f"DockQ wrt State1", fontsize = 14 )
# 		ax[i].set_ylabel( f"DockQ wrt State2", fontsize = 14 )

# 		ax[i].tick_params(
# 			axis = "both",
# 			labelsize = 12,
# 			length = 8,
# 			width = 2
# 		)
# 	plt.legend( loc = "lower right" )
# 	plt.tight_layout()
# 	file = os.path.join(
# 		FIG_DIR, f"figure_3b_kappa.png"
# 	)
# 	plt.savefig( file, dpi = 300 )
# 	plt.close


def fig_X():
	"""
	Plot the time taken for prediction for all models
		across all complexes as a bar plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 5, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		metric = "time" )

	for i, model in enumerate( MODELS ):
		time_taken = records[model]["time"]
		sem = np.std( time_taken, ddof = 1 )/np.sqrt( len( time_taken ) )
		mean_time = np.mean( time_taken )
		ax.bar(
			i,
			mean_time,
			yerr = sem,
			width = 0.5,
			color = COLOR[model]
		)
		ax.text(
			i,
			mean_time + sem + 0.01*ax.get_ylim()[1],
			f"{mean_time:.2f} ± {sem:.2f}",
			ha = "center",
			va = "bottom",
			fontsize = 8,
			rotation = 0,
		)

	ax.set_ylim( -0.05 )
	ax.set_xticks( [0, 1, 2] )
	ax.tick_params( axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		["AlphaLink2", "Boltz2", "GRASP"],
		rotation = 0
		)
	ax.set_ylabel( f"Mean time taken (hours)", fontsize = 14 )
	ax.set_xlabel( "Method", fontsize = 14 )
	ax.tick_params(
		axis = "both",
		labelsize = 10,
		length = 6,
		width = 2
	)

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"time_taken.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()


if __name__ == "__main__":
	fig_2a()
	fig_2b()
	fig_2c()
	fig_3a()
	fig_3b()
	fig_3c()
	fig_3d()
	fig_4a()
	fig_4b()
	fig_4c()
	fig_5a()
	fig_5b()
	fig_5c()
	fig_X()
