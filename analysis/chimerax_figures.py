"""
Plot the predicted structures by each method
	(AlphaLink2, Boltz2, ans GRASP) for a complex.
"""

import chimerax
from chimerax.core.commands import run

from typing import List
import os, json
import numpy as np

def plot_ensemble(
	model_dict: dict[str, dict],
	native_dict: dict[str, dict]
):
	# Set ChimeraX preset 1
	run( session, "preset 1" )

	for protein in native_dict:
		for state_id in native_dict[protein]:
			native_file = native_dict[protein][state_id]["native_file"]
			# Open the native model -> #1
			# Just to set the view
			run( session, f"open {native_file}")
			run( session, "view all" )
			run( session, "close" )
			# Open again
			run( session, f"open {native_file}")
			# Hide all atoms, reduce silhouette width
			run( session, "graphics silhouette width 1" )
			run( session, "hide #1" )
			# Reduce transparency of native model
			run( session, "transparency #1 70 target c" )

			for config_name in model_dict[protein][state_id]:
				for method in model_dict[protein][state_id][config_name]:
					model_file = model_dict[protein][state_id][config_name][method]["model_file"]
					png_file = model_dict[protein][state_id][config_name][method]["png_file"]
					color = model_dict[protein][state_id][config_name][method]["color"]

					# Open the predicted model -> #2
					run( session, f"open {model_file}")
					# Set the view for predicted model
					run( session, "view all" )
					# Hide all atoms
					run( session, "hide #2" )

					# Color the models
					run( session, f"color #1 cornflower blue" )
					run( session, f"color #2 {color}" )

					# Superpose the predicted model on the native model
					run( session, "matchmaker #2 to #1" )

					# Save image
					run( session, f"save {png_file} format png width 3600 height 2400" )
					# Close the predicted model
					run( session, "close #2" )
		# Close #1
		run( session, "close #1" )
		# Close chimerax
		run( session, "close" )

################################################################################
COLOR = {
	"alphalink2": "#8A62FF",
	"boltz2": "#41D041FE",
	"grasp": "#FFB340",
}
FIGURE_DIR = "/data2/kartik/IMP_Rewired/im_bench/imp_dl/benchmark/paper_figures/chimerax/"
os.makedirs( FIGURE_DIR, exist_ok = True )

native_dict = {}
model_dict = {}

for protein in ["ms4", "ms5"]:
	native_dict[protein] = {}
	model_dict[protein] = {}
	for state_id in [0, 1]:
		native_dict[protein][state_id] = {}
		model_dict[protein][state_id] = {}

		for config_name in ["kappa3"]:
			model_dict[protein][state_id][config_name] = {}
			for method in ["alphalink2", "boltz2", "grasp"]:
				logs_file = f"/data2/kartik/IMP_Rewired/im_bench/imp_dl/benchmark/multistate_analysis/Logs_multistate_{method}.npy"
				logs = np.load( logs_file, allow_pickle = True ).item()

				max_tm = 0
				max_tm_model_id = None
				for model_id in logs[f"{method}_{config_name}"][protein]["tm"]:
					tm = logs[f"{method}_{config_name}"][protein]["tm"][model_id][1000+state_id]["tm"]
					if tm > max_tm:
						max_tm = tm
						max_tm_model_id = model_id

				model_file = logs[f"{method}_{config_name}"][protein]["model_files"][max_tm_model_id]
				native_file = logs[f"{method}_{config_name}"][protein]["native_files"][state_id]

				if "native_file" not in native_dict[protein][state_id]:
					native_dict[protein][state_id] = {
						"native_file": native_file
					}
				png_file = os.path.join(
					FIGURE_DIR,
					f"{config_name}_{method}_{protein}_S{state_id+1}.png"
				)
				model_dict[protein][state_id][config_name][method] = {
					"model_file": model_file,
					"color": COLOR[method],
					"png_file": png_file
				}

plot_ensemble(
	model_dict = model_dict,
	native_dict = native_dict
)

