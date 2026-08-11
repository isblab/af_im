"""
Methods to create figures for the paper.
"""
from typing import List
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.stats import wilcoxon, mannwhitneyu

from config import get_config_dict

from utils.analysis_utils import (
	return_metric,
	prep_native_tm_input,
	compute_tp_fp_xl_sat
)

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
	BASE_DIR, "paper_figures"
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
# Main figures
################################################################################
################################################################################
def fig_2():
	"""
	Plot the max XL satisfaction for all models
		across all complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		benchmark_name = "crosslink",
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

	ax.axhline( 0.75, color = "red", linewidth = 0.5, linestyle = "--" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"Max XL satisfaction", fontsize = 14 )
	ax.set_xlabel( "Complexes", fontsize = 14 )
	ax.set_ylim( -0.05, 1.05 )
	ax.legend( loc = "lower right" )
	ax.tick_params(
		axis = "both",
		labelsize = 12,
		length = 8,
		width = 2
	)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_2.png"
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

	fig, ax = plt.subplots( 1, 1, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		benchmark_name = "crosslink",
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

	ax.axhline( 0.23, color = "red", linewidth = 0.5, linestyle = "--" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"Max DockQ", fontsize = 14 )
	ax.set_xlabel( "Complexes", fontsize = 14 )
	ax.set_ylim( -0.05, 1.05 )
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
	Plot the Molprobity score for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 3, figsize = ( 10, 3 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		config_name = config_name,
		benchmark_name = "crosslink",
		models = MODELS,
		metric = "molprob",
		molprob_native = False
		)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records[model]["resolution"] )
		Y = np.array( records[model]["mean_molprob"] )
		# stat, p = wilcoxon( X.reshape( -1 ), Y.reshape( -1 ), alternative = "greater" )
		# p = round( p, 4 )

		ax[i].scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
		)
		# ax[i].text(
		# 	0.02, 2.98,
		# 	f"p-value = {p:.3g}",
		# 	ha = "left", va = "top",
		# 	fontsize = 10
		# )

		ax[i].plot( [0, 3], [0, 3], c = "#DDDDDD" )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_ylabel( f"Mean MolProbity score", fontsize = 12 )
		ax[i].set_xlabel( "Experimental resolution", fontsize = 12 )
		ax[i].axis( [0.0, 3.05, 0.0, 3.05] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 10,
			length = 6,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_3b.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
################################################################################
def fig_4():
	"""
	Plot the no. of unique models based on interface similarity for
		all models across all complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		benchmark_name = "crosslink",
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

	ax.axhline( 1, color = "red", linewidth = 0.5, linestyle = "--" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"Number of unique structures", fontsize = 14 )
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
		FIG_DIR, f"figure_4.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def fig_5():
	"""
	Plot the TP and FP Xl satisfationfor dataset with
		higher FP XLs wrt the control (10% FP XLs).
	Create plots for all models across all complexes as
		a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta3"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		benchmark_name = "crosslink",
		metric = "xl_sat"
	)

	fig, ax = plt.subplots( 3, 1, figsize = ( 12, 10 ) )
	plt.rcParams["font.family"] = "sans-serif"
	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):

		complexes = records[model]["complex"]
		X = np.arange( 0, len( complexes ), 1 )
		total = X.shape[0]
		tp_xl_sat, fp_xl_sat, max_tp_sat, max_fp_sat = compute_tp_fp_xl_sat(
			xl_pair_sat = records[model]["xl_pair_sat"],
			labels = records[model]["label"]
		)
		Y1 = np.array( max_tp_sat ).reshape( total, -1 )
		Y2 = np.array( max_fp_sat ).reshape( total, -1 )

		ax[i].scatter(
			X, Y1,
			s = 80,
			c = COLOR[model],
			marker = "+",
			label = "TP XL"
		)
		ax[i].scatter(
			X, Y2,
			s = 80,
			c = COLOR[model],
			marker = "x",
			label = "FP XL"
		)

		ax[i].set_xticks( X )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels(
			records[model]["complex"], rotation = 90
			)
		ax[i].set_ylabel( f"XL satisfaction", fontsize = 14 )
		ax[i].set_xlabel( "Complexes", fontsize = 14 )
		ax[i].set_ylim( -0.1, 1.1 )
		ax[i].tick_params(
			axis = "both",
			labelsize = 12,
			length = 8,
			width = 2
		)

		ax[i].legend(
			loc = "upper right",
			bbox_to_anchor = ( 0.98, 0.665),
			fontsize = 12,
		)

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_5.png"
	)
	plt.savefig( file, dpi = 300, bbox_inches = "tight" )
	plt.close()

################################################################################
def fig_6():
	"""
	Plot the distribution of per-model TM-score for each method,
		each complex wrt state1 and state2.
	Create a plot for all multistate configs - kappa1, kappa2, kappa3.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_names = ["kappa1", "kappa2","kappa3"]

	fig, ax = plt.subplots( 3, 1, figsize = ( 6, 7 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = prep_native_tm_input(
		config_name = config_names[0],
		benchmark_name = "multistate",
		models = MODELS,
		native_model_id = 1000
	)
	complexes = records[MODELS[0]]["complex"]
	# Spacing between violins for different complexes
	complex_offset = [-0.25, 0, 0.25]
	# Width of violins
	model_width = 0.2

	# offsets within each complex
	model_spacing = 0.9
	state_spacing = 0.22

	titles = [
		"XLs from Apo state",
		"XLs from Holo state",
		"XLs from Apo+Holo states"
	]
	prot_name = {
		"ms4": "RNase H",
		"ms5": "Calmodulin1"
	}

	for i, config_name in enumerate( config_names ):
		X, Y = [], []
		colours = []
		complex_centers = []
		state_centers = []
		state_labels = []
		# X-tick labels for complexes
		xtick_labels = []
		# For alternating grey spans
		shade_regions = []
		base = 1

		for c_idx, complex_name in enumerate( complexes ):
			complex_centers.append( base + model_spacing )
			xtick_labels.append( prot_name[complex_name] )

			group_start = base
			for m_idx, model in enumerate( MODELS ):
				x_model = base + m_idx * model_spacing

				for s_idx, ( native_state_id, s_label ) in enumerate(
						zip( [1000, 1001], ["Apo", "Holo"] )
					):
					records = prep_native_tm_input(
						config_name = config_name,
						benchmark_name = "multistate",
						models = MODELS,
						native_model_id = native_state_id,
					)

					data = np.array(
						records[model]["per_model_tm"]
					).reshape( len( complexes ), -1 )

					x = x_model + ( -state_spacing/2 if s_idx == 0 else state_spacing/2 )

					state_centers.append( x )
					state_labels.append( s_label )
					X.append( x )
					Y.append( data[c_idx] )
					colours.append( COLOR[model] )

			group_end = x_model + state_spacing/2
			shade_regions.append( ( group_start - 0.3, group_end + 0.3 ) )
			base = group_end + 1.0

		vp = ax[i].violinplot(
			Y,
			positions = X,
			widths = model_width,
			showmeans = True,
			# showmedians = True,
			showextrema = False,
		)
		for j, ( xmin, xmax ) in enumerate( shade_regions ):
			if j % 2 == 1:
				ax[i].axvspan(
					xmin,
					xmax,
					color = "#DDDDDD",
					alpha = 0.3,
					zorder = 0
				)
		ax[i].axhline( 0.5, color = "red", linewidth = 0.5, linestyle = "--" )

		for k, body in enumerate( vp["bodies"] ):
			body.set_facecolor( colours[k] )
			body.set_edgecolor( "black" )
			body.set_alpha( 0.6 )

		# Add figure legends only on one of the subplots.
		legend_handles = [
			Patch(
				facecolor = COLOR[MODELS[0]],
				edgecolor = "black",
				alpha = 0.6, label = METHOD_LABELS[MODELS[0]]
				),
			Patch(
				facecolor = COLOR[MODELS[1]],
				edgecolor = "black",
				alpha = 0.6, label = METHOD_LABELS[MODELS[1]]
			),
			Patch(
				facecolor = COLOR[MODELS[2]],
				edgecolor = "black",
				alpha = 0.6, label = METHOD_LABELS[MODELS[2]]
			),
		]

		fig.legend(
			handles = legend_handles,
			loc = "lower right",
			bbox_to_anchor = ( 0.99, 0.075),
			frameon = False,
			fontsize = 7,
		)
		ax[i].set_title( titles[i], fontsize = 10 )
		# State name
		ax[i].set_ylim( -0.05, 1.1 )
		ax[i].set_xticks( state_centers )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels( state_labels, fontsize = 10, rotation = 90 )
		# Complex name
		ax_top = ax[i].secondary_xaxis( "top" )
		ax_top.set_xticks( complex_centers )
		ax_top.set_xticklabels( xtick_labels, fontsize = 10 )
		ax_top.tick_params(length = 0, pad = 6 )

		ax[i].set_ylabel( "TM-score", fontsize = 10 )

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"figure_6.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
################################################################################
# Supplementary figures
################################################################################
################################################################################
def supp_fig_1():
	"""
	For guided vs unguided predictions, plot max
		XL satisfaction for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	fig, ax = plt.subplots( 1, 2, figsize = ( 6, 3 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		benchmark_name = "crosslink",
		metric = "xl_sat"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		benchmark_name = "crosslink",
		metric = "xl_sat"
	)

	# text_offset_y = [0.98, 0.94]
	# opacity = [1.0, 0.5]
	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records1[model]["max_xl_sat"] )
		Y = np.array( records2[model]["max_xl_sat"] )
		# stat, p = wilcoxon( X, Y, alternative = "greater" )
		# p = round( p, 4 )

		ax[i].scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
			# alpha = opacity[i],
			label = METHOD_LABELS[model]
		)
		# ax.text(
		# 	0.02, text_offset_y[i],
		# 	f"p-value = {p:.3g}",
		# 	ha = "left", va = "top",
		# 	fontsize = 10,
		# 	color = COLOR[model]
		# )

		ax[i].plot( [-0.05, 1.05], [-0.05, 1.05], c = "#DDDDDD" )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xlabel( f"Max XL satisfaction (guided)", fontsize = 10 )
		ax[i].set_ylabel( f"Max XL satisfaction (unguided)", fontsize = 10 )
		ax[i].axis( [-0.05, 1.05, -0.05, 1.05] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 8,
			length = 6,
			width = 2
		)
		ax[i].legend( loc = "lower right" )
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_1.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_2():
	"""
	For guided vs unguided predictions, plot the
		XL satisfaction for all models across
		all complexes as violin plots.
	"""
	MODELS = ["boltz2", "grasp"]
	config_name1 = "beta1" # guided
	config_name2 = "alpha" # unguided

	records1 = return_metric(
		models = MODELS,
		config_name = config_name1,
		benchmark_name = "crosslink",
		metric = "xl_sat"
	)
	records2 = return_metric(
		models = MODELS,
		config_name = config_name2,
		benchmark_name = "crosslink",
		metric = "xl_sat"
	)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		fig, ax = plt.subplots( 2, 1, figsize = ( 10, 8 ) )
		plt.rcParams["font.family"] = "sans-serif"

		complexes = records1[model]["complex"]
		X = np.arange( 0, len( complexes ), 1 )
		total = X.shape[0]
		# List elements are in order: al mdoels per complex, for all complexes (37*25).
		# [37*25] -> [37, 25]
		Y1 = np.array( records1[model]["per_model_xl_sat"] ).reshape( total, -1 )
		# [37*25] -> [37, 25]
		Y2 = np.array( records2[model]["per_model_xl_sat"] ).reshape( total, -1 )

		for subplot_idx, (  start, end ) in enumerate( [( 0, 19 ), ( 19, total )] ):

			y1 = Y1[start:end]
			y2 = Y2[start:end]
			names = complexes[start:end]

			n = len( names )

			data = []
			positions = []

			for j in range(n):
				data.extend( [y1[j], y2[j]] )
				positions.extend( [j - 0.2, j + 0.2] )

			# Grey vertical bands
			for j in range( n ):
				if j % 2 == 0:
					ax[subplot_idx].axvspan(
						j - 0.5,
						j + 0.5,
						color = "#DDDDDD",
					)

			vp = ax[subplot_idx].violinplot(
				data,
				positions = positions,
				widths = 0.35,
				showmeans = True,
				showmedians = True,
				showextrema = False,
			)

			for k, body in enumerate( vp["bodies"] ):
				body.set_facecolor( "#4C72B0" if k % 2 == 0 else "#DD8452" )
				body.set_edgecolor( "black" )
				body.set_alpha( 0.6 )

			ax[subplot_idx].set_xticks( np.arange( n ) )
			ax[subplot_idx].set_xticklabels( names, rotation = 90, fontsize = 12 )
			ax[subplot_idx].set_xlim( -0.6, n - 0.4 )
			ax[subplot_idx].set_ylim( -0.5, 1.1 )
			ax[subplot_idx].tick_params( axis = "both", width = 2, length = 5 )

			ax[subplot_idx].set_xlabel( "Complexes", fontsize = 14 )
			ax[subplot_idx].set_ylabel( "Per-model XL satisfaction", fontsize = 14 )

		legend_handles = [
			Patch(
				facecolor = "#4C72B0",
				edgecolor = "black",
				alpha = 0.6, label = "Guided"
				),
			Patch(
				facecolor = "#DD8452",
				edgecolor = "black",
				alpha = 0.6, label = "Unguided"
			),
		]

		fig.legend(
			handles = legend_handles,
			loc = "upper right",
			bbox_to_anchor = ( 0.98, 0.7),
			frameon = False,
			fontsize = 12,
		)

		plt.tight_layout()
		file = os.path.join(
			FIG_DIR, f"supp_figure_2_{model}.png"
		)
		plt.savefig( file, dpi = 300, bbox_inches = "tight" )
		plt.close()

################################################################################
def supp_fig_3():
	"""
	Plot the clashscore for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 3, figsize = ( 10, 3 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records_pred = return_metric(
		config_name = config_name,
		benchmark_name = "crosslink",
		models = MODELS,
		metric = "molprob",
		molprob_native = False
		)
	records_native = return_metric(
		config_name = config_name,
		benchmark_name = "crosslink",
		models = MODELS,
		metric = "molprob",
		molprob_native = True
		)

	# Create a scatter plot for each method.
	for i, model in enumerate( MODELS ):
		X = np.array( records_native[model]["mean_clash"] )
		Y = np.array( records_pred[model]["mean_clash"] )

		ax[i].scatter(
			X, Y,
			s = 50,
			c = COLOR[model],
			marker = MARKER[model],
		)

		ax[i].axvline( 5.0, color = "red", linewidth = 0.5, linestyle = "--" )
		ax[i].axhline( 5.0, color = "red", linewidth = 0.5, linestyle = "--" )
		x_offset = X.max()
		y_offset = 31 if i <=1 else 125

		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_ylabel( f"Mean Clashscore (prediction)", fontsize = 10 )
		ax[i].set_xlabel( "Clashscore (native)", fontsize = 10 )
		ax[i].axis( [-1, x_offset+1, 0, y_offset] )
		ax[i].tick_params(
			axis = "both",
			labelsize = 8,
			length = 8,
			width = 2
		)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_3.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_4():
	"""
	Plot the mean confidence score for all models across all
		complexes as a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_name = "beta1"

	fig, ax = plt.subplots( 1, 1, figsize = ( 10, 5 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		models = MODELS,
		config_name = config_name,
		benchmark_name = "crosslink",
		metric = "confidence"
	)

	# Create a scatter plot for each method.
	for idx, model in enumerate( MODELS ):
		X = np.arange( 0, len( records[model]["complex"] ), 1 )
		Y = np.array( records[model]["mean_confidence"] )

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

	ax.axhline( 0.8, color = "red", linewidth = 0.5, linestyle = "--" )
	ax.set_xticks( X )
	ax.tick_params(axis = "both", width = 2, length = 5 )
	ax.set_xticklabels(
		records[model]["complex"], rotation = 90
		)
	ax.set_ylabel( f"Mean model confidence", fontsize = 14 )
	ax.set_xlabel( "Complexes", fontsize = 14 )
	ax.set_ylim( -0.02, 1.02 )
	ax.legend( loc = "center left" )
	ax.tick_params(
		axis = "both",
		labelsize = 12,
		length = 8,
		width = 2
	)
	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_4.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_5a():
	"""
	Plot the no. of structures with unique interface for
		dataset with higher FP XLs wrt the control (10% FP XLs).
	Create plots for all models across all complexes as
		a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_names = ["beta3"]

	fig, ax = plt.subplots( 1, 3, figsize = ( 10, 3.4 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = "beta1",
		benchmark_name = "crosslink",
		metric = "unique_interface"
	)

	fp_label = ["33% FP XLs"]
	for i, config_name in enumerate( config_names ):
		records2 = return_metric(
			models = MODELS,
			config_name = config_name,
			benchmark_name = "crosslink",
			metric = "unique_interface"
		)

		for j, model in enumerate( MODELS ):
			X = records1[model]["unique_interface"]
			Y = records2[model]["unique_interface"]

			ax[j].scatter(
				X, Y,
				s = 50,
				c = COLOR[model],
				marker = MARKER[model],
				label = METHOD_LABELS[model]
			)
			ax[j].plot( [0, 25], [0, 25], c = "#DDDDDD" )
			ax[j].set_xlabel( "Number of unique structures (10% FP XLs)", fontsize = 10 )
			ax[j].set_ylabel( f"Number of unique structures ({fp_label[i]})", fontsize = 10 )
			ax[j].axis( [-0.05, 26, -0.05, 26] )
			ax[j].tick_params(
				axis = "both",
				labelsize = 8,
				length = 8,
				width = 2
			)
			ax[j].legend( loc = "lower right" )

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_5a.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_5b():
	"""
	Plot the mean MolProbity score vs experimental resolution for
		dataset with higher FP XLs wrt the control (10% FP XLs).
	Create plots for all models across all complexes as
		a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_names = ["beta3"]

	fig, ax = plt.subplots( 1, 3, figsize = ( 10, 3.4 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = "beta1",
		benchmark_name = "crosslink",
		metric = "molprob"
	)

	fp_label = ["33% FP XLs"]
	for i, config_name in enumerate( config_names ):
		records2 = return_metric(
			models = MODELS,
			config_name = config_name,
			benchmark_name = "crosslink",
			metric = "molprob"
		)

		for j, model in enumerate( MODELS ):
			X = records1[model]["mean_molprob"]
			Y = records2[model]["mean_molprob"]

			ax[j].scatter(
				X, Y,
				s = 50,
				c = COLOR[model],
				marker = MARKER[model],
				label = METHOD_LABELS[model]
			)

			ax[j].plot( [0, 25], [0, 25], c = "#DDDDDD" )
			ax[j].set_xlabel( "Experimental resolution", fontsize = 10 )
			ax[j].set_ylabel( f"Mean Molprobity Score ({fp_label[i]})", fontsize = 10 )
			ax[j].axis( [-0.05, 3, -0.05, 3] )
			ax[j].tick_params(
				axis = "both",
				labelsize = 8,
				length = 8,
				width = 2
			)
			ax[j].legend( loc = "lower right" )

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_5b.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_5c():
	"""
	Plot the mean Clashscore for dataset with higher
		FP XLs wrt the control (10% FP XLs).
	Create plots for all models across all complexes as
		a scatter plot.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_names = ["beta3"]

	fig, ax = plt.subplots( 1, 3, figsize = ( 10, 3.4 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records1 = return_metric(
		models = MODELS,
		config_name = "beta1",
		benchmark_name = "crosslink",
		metric = "molprob"
	)

	fp_label = ["33% FP XLs"]
	for i, config_name in enumerate( config_names ):
		records2 = return_metric(
			models = MODELS,
			config_name = config_name,
			benchmark_name = "crosslink",
			metric = "molprob"
		)

		for j, model in enumerate( MODELS ):
			X = records1[model]["mean_clash"]
			Y = records2[model]["mean_clash"]

			ax[j].scatter(
				X, Y,
				s = 50,
				c = COLOR[model],
				marker = MARKER[model],
				label = METHOD_LABELS[model]
			)

			ax[j].set_xlabel( f"Mean Clashscore (10% FP XLs)", fontsize = 10 )
			ax[j].set_ylabel( f"Mean Clashscore ({fp_label[i]})", fontsize = 10 )
			if model == "grasp":
				ax[j].plot( [0, 125], [0, 125], c = "#DDDDDD" )
				ax[j].axis( [-0.05, 125, -0.05, 125] )
			else:
				ax[j].plot( [0, 35], [0, 35], c = "#DDDDDD" )
				ax[j].axis( [-0.05, 35, -0.05, 35] )
			ax[j].tick_params(
				axis = "both",
				labelsize = 8,
				length = 8,
				width = 2
			)
			ax[j].legend( loc = "lower right" )

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_5c.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_5d():
	"""
	For GRASP plot the mean clashscore and no. of unique interfaces for complexes
		with FP XLs satisfied.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	# model = MODELS[0]
	config_name = "beta3"

	records1, records2 = {}, {}
	for metric in ["xl_sat", "molprob", "unique_interface"]:
		records1[metric] = return_metric(
			models = MODELS,
			config_name = "beta1",
			benchmark_name = "crosslink",
			metric = metric
		)

		records2[metric] = return_metric(
			models = MODELS,
			config_name = config_name,
			benchmark_name = "crosslink",
			metric = metric
		)

	fig, ax = plt.subplots( 1, 2, figsize = ( 8, 4 ) )
	plt.rcParams["font.family"] = "sans-serif"

	labels = {
		"molprob": "Mean Clashscore",
		"unique_interface": "Number of unique interfaces"
	}

	for i, model in enumerate( MODELS ):
		tp_xl_sat, fp_xl_sat, max_tp_sat, max_fp_sat = compute_tp_fp_xl_sat(
			xl_pair_sat = records2["xl_sat"][model]["xl_pair_sat"],
			labels = records2["xl_sat"][model]["label"]
		)
		for j, metric in enumerate( ["molprob", "unique_interface"] ):
			X, Y = [], []
			for k, fp_sat in enumerate( max_fp_sat ):
				if metric == "molprob":
					m = "mean_clash"
				else:
					m = "unique_interface"
				if fp_sat > 0:
					X.append(
						records1[metric][model][m][k]
					)
					Y.append(
						records2[metric][model][m][k]
					)
			ax[j].scatter(
				X, Y,
				s = 50,
				c = COLOR[model],
				marker = MARKER[model],
				label = METHOD_LABELS[model]
			)

			ax[j].set_xlabel( f"{labels[metric]} (10% FP XLs)", fontsize = 10 )
			ax[j].set_ylabel( f"{labels[metric]} (33% FP XLs)", fontsize = 10 )
			if metric == "molprob":
				ax[j].plot( [0, 125], [0, 125], c = "#DDDDDD" )
				ax[j].axis( [-0.05, 125, -0.05, 125] )
			else:
				ax[j].plot( [0, 26], [0, 26], c = "#DDDDDD" )
				ax[j].axis( [-0.5, 26, -0.5, 26] )
			ax[j].tick_params(
				axis = "both",
				labelsize = 8,
				length = 8,
				width = 2
			)
			ax[j].legend( loc = "lower right" )

	plt.tight_layout()
	plt.subplots_adjust( wspace = 0.4 )
	file = os.path.join(
		FIG_DIR, f"supp_figure_5d.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_6():
	"""
	Plot the distribution of per-model XL satisfaction for each method,
		each complex wrt state1 and state2.
	Create a plot for all multistate configs - kappa1, kappa2, kappa3.
	"""
	MODELS = ["alphalink2", "boltz2", "grasp"]
	config_names = ["kappa1", "kappa2","kappa3"]

	fig, ax = plt.subplots( 1, 1, figsize = ( 9, 4 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = return_metric(
		config_name = config_names[0],
		benchmark_name = "multistate",
		models = MODELS,
		metric = "xl_sat"
	)

	complexes = records[MODELS[0]]["complex"]

	# Spacing between violins for different model per complex
	complex_spacing = 1.0
	# Spacing between the states
	state_spacing = 1.5
	model_offset = [-0.25, 0.0, 0.25]
	# Width of violins
	model_width = 0.2

	X = []
	Y = []
	colours = []
	xtick_pos = []
	xtick_label = []
	state_xtick = []
	# X-tick labels for state name
	state_xticklabel = [
		"XLs from Apo state",
		"XLs from Holo state",
		"XLs from Apo+Holo states"
		]
	prot_name = {
		"ms4": "RNase H",
		"ms5": "Calmodulin1"
	}
	base = 1
	for s_idx, config_name in enumerate( config_names ):

		records = return_metric(
			config_name = config_name,
			benchmark_name = "multistate",
			models = MODELS,
			metric = "xl_sat"
		)

		state_base = s_idx * ( len( complexes )*complex_spacing + state_spacing )
		if s_idx == 0:
			state_base += 0.7
		if s_idx == 2:
			state_base -= 0.7
		state_xtick.append(
			state_base + ( len( complexes )-1 )*complex_spacing/2
		)
		for c_idx, complex_name in enumerate( complexes ):
			complex_center = state_base + c_idx * complex_spacing
			xtick_pos.append( complex_center )
			xtick_label.append( prot_name[complex_name] )
			for m_idx, model in enumerate( MODELS ):
				data = np.array(
					records[model]["per_model_xl_sat"]
				).reshape( len( complexes ), -1 )

				X.append( complex_center + model_offset[m_idx] )
				Y.append( data[c_idx] )
				colours.append( COLOR[model] )

	vp = ax.violinplot(
		Y,
		positions = X,
		widths = model_width,
		showmeans = True,
		# showmedians = True,
		showextrema = False,
	)
	ax.axvspan(
		state_xtick[1] - 1.5,
		state_xtick[1] + 1.5,
		color = "#DDDDDD",
		alpha = 0.3,
		zorder = 0
	)
	ax.axhline( 0.75, color = "red", linewidth = 0.5, linestyle = "--" )

	for k, body in enumerate( vp["bodies"] ):
		body.set_facecolor( colours[k] )
		body.set_edgecolor( "black" )
		body.set_alpha( 0.6 )

	# Add figure legens only on one of the subplots.
	legend_handles = [
		Patch(
			facecolor = COLOR[MODELS[0]],
			edgecolor = "black",
			alpha = 0.6, label = METHOD_LABELS[MODELS[0]]
			),
		Patch(
			facecolor = COLOR[MODELS[1]],
			edgecolor = "black",
			alpha = 0.6, label = METHOD_LABELS[MODELS[1]]
		),
		Patch(
			facecolor = COLOR[MODELS[2]],
			edgecolor = "black",
			alpha = 0.6, label = METHOD_LABELS[MODELS[2]]
		),
	]

	fig.legend(
		handles = legend_handles,
		loc = "lower right",
		bbox_to_anchor = ( 0.97, 0.2),
		frameon = False,
		fontsize = 8,
	)
	ax.set_ylim( -0.05, 1.1 )
	ax.set_xticks( xtick_pos )
	ax.tick_params( axis = "both", width = 2, length = 5 )
	ax.set_xticklabels( xtick_label, rotation = 0, fontsize = 12 )
	# State name
	ax_top = ax.secondary_xaxis( "top" )
	ax_top.set_xticks( state_xtick )
	ax_top.set_xticklabels( state_xticklabel, fontsize = 12 )
	ax_top.tick_params(length = 0, pad = 6 )

	ax.set_xlabel( "Complexes", fontsize = 12 )
	ax.set_ylabel( "XL satisfaction", fontsize = 12 )

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_6.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_7():
	"""
	Plot the distribution of per-model TM-score for each method,
		each complex wrt state1 and state2.
	Compare the across configs for AlphaLink2 and Boltz2 - kappa3, mu3, nu3.
	"""
	MODELS = ["alphalink2", "boltz2"]
	config_names = ["kappa3", "nu3", "mu3"]

	fig, ax = plt.subplots( 3, 1, figsize = ( 7, 8 ) )
	plt.rcParams["font.family"] = "sans-serif"

	records = prep_native_tm_input(
		config_name = config_names[0],
		benchmark_name = "multistate",
		models = MODELS,
		native_model_id = 1000
	)
	complexes = records[MODELS[0]]["complex"]
	# Spacing between violins for different complexes
	complex_offset = [-0.25, 0, 0.25]
	# Width of violins
	model_width = 0.2

	# offsets within each complex
	model_spacing = 0.9
	state_spacing = 0.22
	config_spacing = 0.2

	titles = [
		"Control",
		"Increased sampled models",
		"MSA subsampling",
	]

	for i, config_name in enumerate( config_names ):
		if config_name in ["kappa3", "nu3"]:
			MODELS = ["alphalink2", "boltz2", "grasp"]
		else:
			MODELS = ["alphalink2", "boltz2"]
		X, Y = [], []
		colours = []
		complex_centers = []
		state_centers = []
		state_labels = []
		# X-ticks for configs
		config_centers = []
		config_labels = []
		# X-tick labels for complexes
		xtick_labels = []
		# For alternating grey spans
		shade_regions = []
		base = 1

		for c_idx, complex_name in enumerate( complexes ):
			complex_centers.append( base + 0.8 )
			xtick_labels.append( complex_name )

			group_start = base
			for m_idx, model in enumerate( MODELS ):
				x_model = base + m_idx * model_spacing

				for s_idx, ( native_state_id, s_label ) in enumerate(
						zip( [1000, 1001], ["Apo", "Holo"] )
					):
					records = prep_native_tm_input(
						config_name = config_name,
						benchmark_name = "multistate",
						models = MODELS,
						native_model_id = native_state_id,
					)

					data = np.array(
						records[model]["per_model_tm"]
					).reshape( len( complexes ), -1 )

					x = x_model + ( -state_spacing/2 if s_idx == 0 else state_spacing/2 )

					state_centers.append( x )
					state_labels.append( s_label )
					X.append( x )
					Y.append( data[c_idx] )
					colours.append( COLOR[model] )

			group_end = x_model + state_spacing/2
			shade_regions.append( ( group_start - 0.3, group_end + 0.3 ) )
			base = group_end + 1.0

		vp = ax[i].violinplot(
			Y,
			positions = X,
			widths = model_width,
			showmeans = True,
			# showmedians = True,
			showextrema = False,
		)
		for j, ( xmin, xmax ) in enumerate( shade_regions ):
			if j % 2 == 1:
				ax[i].axvspan(
					xmin,
					xmax,
					color = "#DDDDDD",
					alpha = 0.3,
					zorder = 0
				)
		ax[i].axhline( 0.5, color = "red", linewidth = 0.5, linestyle = "--" )

		for k, body in enumerate( vp["bodies"] ):
			body.set_facecolor( colours[k] )
			body.set_edgecolor( "black" )
			body.set_alpha( 0.6 )

		# Add figure legends only on one of the subplots.
		legend_handles = [
			Patch(
				facecolor = COLOR[MODELS[0]],
				edgecolor = "black",
				alpha = 0.6, label = METHOD_LABELS[MODELS[0]]
				),
			Patch(
				facecolor = COLOR[MODELS[1]],
				edgecolor = "black",
				alpha = 0.6, label = METHOD_LABELS[MODELS[1]]
			),
		]
		if config_name != "mu3":
			legend_handles.append( Patch(
				facecolor = COLOR[MODELS[2]],
				edgecolor = "black",
				alpha = 0.6, label = METHOD_LABELS[MODELS[2]]
			)
			)

		ax[i].legend(
			handles = legend_handles,
			loc = "lower right",
			bbox_to_anchor = ( 0.995, 0.01),
			frameon = False,
			fontsize = 7,
		)
		ax[i].set_title( titles[i], fontsize = 10 )
		# State name
		ax[i].set_ylim( -0.05, 1.1 )
		ax[i].set_xticks( state_centers )
		ax[i].tick_params(axis = "both", width = 2, length = 5 )
		ax[i].set_xticklabels( state_labels, fontsize = 10, rotation = 90 )
		# Complex name
		ax_top = ax[i].secondary_xaxis( "top" )
		ax_top.set_xticks( complex_centers )
		ax_top.set_xticklabels( xtick_labels, fontsize = 10 )
		ax_top.tick_params(length = 0, pad = 6 )

		ax[i].set_ylabel( "TM-score", fontsize = 10 )

	plt.tight_layout()
	file = os.path.join(
		FIG_DIR, f"supp_figure_7.png"
	)
	plt.savefig( file, dpi = 300 )
	plt.close()

################################################################################
def supp_fig_8():
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
		benchmark_name = "crosslink",
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
			fontsize = 10,
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
	fig_2()
	fig_3a()
	fig_3b()
	fig_4()
	fig_5()
	fig_6()
	supp_fig_1()
	supp_fig_2()
	supp_fig_3()
	supp_fig_4()
	supp_fig_5a()
	supp_fig_5b()
	supp_fig_5c()
	supp_fig_5d()
	supp_fig_6()
	supp_fig_7()
	supp_fig_8()
