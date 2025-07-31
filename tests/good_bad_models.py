"""
Exploring and assessing different ways of selecting good-scoring models.
Good-scoring models should have:
	lower violations.
	higher data satisfaction.
1. Quantile-based filtering
2. Non-dominant sorting
3. GMM-based clustering
"""
from typing import List, Dict
import os
import numpy as np
import pandas as pd
from scipy import  stats
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.mixture import GaussianMixture

curr_dir = os.getcwd()
base_dir = os.path.abspath( "../benchmark/" )
meta_dir = os.path.join( base_dir, "xlsim_metadata/" )
modeling_dir_name = "xlsim_modeling"
modeling_version = 0
benchmark_csv_path = os.path.join(
		meta_dir,
		"selected_xlsim_benchmark.csv"
	)

benchmark = pd.read_csv( benchmark_csv_path )

########################################################################
########################################################################
def get_sys_path( sys_name: str ):
	sys_path = os.path.join( 
				os.path.abspath( f"{base_dir}/{modeling_dir_name}/{sys_name}" )
		)
	return sys_path


def get_stat_file_path( sys_path: str ) -> pd.DataFrame:
	"""
	Return the path to the stats file for the given system.
	"""
	stat_file_path = os.path.join( sys_path, f"version_{modeling_version}/Stats.npy" )
	return stat_file_path


def load_stat_file( sys_path: str ) -> Dict[str, Dict]:
	"""
	Load the stat file on memory.
	"""
	stat_file_path = get_stat_file_path( sys_path )
	stats_dict = np.load( stat_file_path, allow_pickle = True ).item()
	return stats_dict

########################################################################
"""
Craete some summary plots:
	Per-epoch distribution of violation loss and xlr metric.
"""
########################################################################
def plot_per_epoch_distribution():
	"""
	Plot the distribution of per epoch values for
		 loss, and xl satisfaction for each systems.
	 Create separate plots for each term.
	"""
	print( "Creating per epoch plots..." )
	plt.rcParams["font.family"] = "sans"
	_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )
	# for qty in ["violation", "xlr"]:
	sys_data_xl = []
	sys_data_viol = []
	categories = []
	for sys_name in benchmark["PDB ID"]:
		categories.append( sys_name )

		sys_path = get_sys_path( sys_name )
		stats_dict = load_stat_file( sys_path )

		sys_data_xl.append( stats_dict["metrics"]["xlr"] )
		sys_data_viol.append( stats_dict["loss"]["violation"] )

	ax[0].violinplot( sys_data_xl )
	# ax[0].boxplot( sys_data, orientation = "vertical" )
	ax[0].set_title( "XL satisfaction", fontsize = 25 )
	ax[0].tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )
	ax[0].set_xticks( np.arange(1, len( categories ) + 1 ), categories )

	ax[1].violinplot( sys_data_viol )
	# ax[1].boxplot( sys_data, orientation = "vertical" )
	ax[1].set_title( "Violations", fontsize = 25 )
	ax[1].tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )
	ax[1].set_xticks( np.arange(1, len( categories ) + 1 ), categories )

	path = os.path.join( curr_dir, f"sampled_models_per_epoch.png" )
	plt.savefig( path, dpi = 300 )
	plt.close()


########################################################################
########################################################################
def plot_structural_similarity():
	"""
	Compute structural similarity between the initial model and all.
	Plot the distribution of rmsd for all complexes.
	"""
	print( "Creating structural similarity plots..." )
	plt.rcParams["font.family"] = "sans"
	_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )
	categories = []
	structural_similarity = []
	for sys_name in benchmark["PDB ID"]:
		categories.append( sys_name )

		sys_path = get_sys_path( sys_name )
		stats_dict = load_stat_file( sys_path )

		prot = stats_dict["protein"]
		rmsd_list = []
		for i in range( len( prot ) ):
			a = prot[0].atom_positions
			b = prot[i].atom_positions
			a_ca = a[:,1,:]
			b_ca = b[:,1,:]
			rot, rssd = R.align_vectors( a_ca, b_ca )
			rmsd = rssd/np.sqrt( a_ca.shape[0] )
			rmsd_list.append( rmsd )
		structural_similarity.append( rmsd_list )
	ax.violinplot( structural_similarity )
	ax.set_title( "Structural similarity", fontsize = 25 )
	ax.set_ylabel( "RMSD wrt epoch0", fontsize = 25 )
	ax.tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )
	ax.set_xticks( np.arange(1, len( categories ) + 1 ), categories )

	path = os.path.join( curr_dir, f"structural_similarity.png" )
	plt.savefig( path, dpi = 300 )
	plt.close()


########################################################################
########################################################################
def quantile_filtering():
	"""
	Quantile-based approach for filtering bad-scoring models.
	"""
	print( "-"*70 + "\nQuantile-based filtering\n" + "-"*70 + "\n"+ "-"*70 )
	results_dict = {}
	for q_xl, q_viol in zip( [0.25, 0.5, 0.75], [0.25, 0.25, 0.25] ):
		print( f"Data quantile = {q_xl} \t Violation quantile = {q_viol}\n" + "-"*70 )
		results_dict[f"{q_xl}-{q_viol}"] = {k:[] for k in ["pdb_id", "num_models",
														"avg_viols", "avg_xl_sat",
														"global_xl_sat", 
														"all_xl_sat", "all_viol"]}
		selected_xl, epoch0_xl = [], []
		global_xl, avg_xl = [], []
		selected_viol, epoch0_viol = [], []
		structural_similarity = []
		categories = []
		for sys_name in benchmark["PDB ID"]:
			categories.append( sys_name )
			sys_path = get_sys_path( sys_name )
			stats_dict = load_stat_file( sys_path )

			# Get models above Q3 for xl_satisfaction.
			# q3 = np.percentile( stats_dict["metrics"]["xlr"], q = 75 )
			data_sat = np.array( stats_dict["metrics"]["xlr"] )
			viols = np.array( stats_dict["loss"]["violation"] )
			q = np.quantile( data_sat, q_xl )
			mask_xl = data_sat >= q
			qxl_idx = np.where( data_sat >= q )

			# Compute quantiles for violations based on the subset that satisfies data.
			q = np.quantile( viols[qxl_idx], q_viol )
			mask_viol = viols <= q
			qviol_idx = np.where( viols[qxl_idx] <= q )

			good_models = mask_xl & mask_viol

			global_sat_arr = stats_dict["metadata"]["xlr"]["xl_satisfaction_array"]
			global_sat_arr = global_sat_arr[good_models]
			total = global_sat_arr.shape[1]
			total_sat = np.count_nonzero( np.sum( global_sat_arr, axis = 0 ) )
			global_xl.append( np.round( total_sat/total, 3 ) )
			avg_xl.append( np.round( np.mean( data_sat[good_models] ), 3 ) )

			structural_similarity.append( 
				all_v_all_rmsd( stats_dict["protein"],
				np.where( good_models == 1 )[0] )
			)

			epoch0_xl.append( benchmark[benchmark["PDB ID"]==sys_name]["XLR metric 0"].tolist()[0] )
			selected_xl.append( data_sat[good_models] )
			epoch0_viol.append( benchmark[benchmark["PDB ID"]==sys_name]["Viol0"].tolist()[0] )
			selected_viol.append( viols[good_models] )

			print( sys_name )
			print( data_sat[good_models] )
			print( viols[good_models] )
			print( data_sat[good_models].shape[0] )
			print( "Global XL satisfaction = ", global_xl[-1] )
			print( "\n" )

			results_dict[f"{q_xl}-{q_viol}"]["pdb_id"].append( sys_name )
			results_dict[f"{q_xl}-{q_viol}"]["num_models"].append( viols[good_models].shape[0] )
			results_dict[f"{q_xl}-{q_viol}"]["avg_viols"].append( np.round( np.mean( viols[good_models] ), 3 ) )
			results_dict[f"{q_xl}-{q_viol}"]["avg_xl_sat"].append( avg_xl[-1] )
			results_dict[f"{q_xl}-{q_viol}"]["global_xl_sat"].append( global_xl[-1] )
			results_dict[f"{q_xl}-{q_viol}"]["all_xl_sat"].append( ", ".join( map( str, data_sat[good_models] ) ) )
			results_dict[f"{q_xl}-{q_viol}"]["all_viol"].append( ", ".join( map( str, viols[good_models] ) ) )

		create_rmsd_plots(
			categories = categories,
			struct_sim = structural_similarity,
			file = f"good_models_{q_xl}_{q_viol}_rmsd.png"
		)

		create_metric_plots(
			categories = categories,
			selected_xl = selected_xl,
			selected_viol = selected_viol,
			epoch0_xl = epoch0_xl,
			epoch0_viol = epoch0_viol,
			global_xl = global_xl,
			avg_xl = avg_xl,
			file = f"good_models_{q_xl}_{q_viol}.png" )

	return results_dict

########################################################################
########################################################################
def nondominant_sorting():
	"""
	Select the non-dominant set of models (Pareto set).
	"""
	print( "-"*70 + "\nNon-dominant sorting\n" + "-"*70 + "\n"+ "-"*70 )
	results_dict = {k:[] for k in ["pdb_id", "num_models",
									"avg_viols", "avg_xl_sat",
									"global_xl_sat",
									"all_xl_sat", "all_viol"]}
	selected_xl, epoch0_xl = [], []
	global_xl, avg_xl = [], []
	selected_viol, epoch0_viol = [], []
	structural_similarity = []
	categories = []

	for sys_name in benchmark["PDB ID"]:
		categories.append( sys_name )
		sys_path = get_sys_path( sys_name )
		stats_dict = load_stat_file( sys_path )

		data_sat = np.array( stats_dict["metrics"]["xlr"] )
		viols = np.array( stats_dict["loss"]["violation"] )

		objectives = np.column_stack(
			[viols,
			-1*data_sat]
		)

		nds = NonDominatedSorting()
		pareto_models = nds.do( objectives, only_non_dominated_front = True )

		epoch0_xl.append( benchmark[benchmark["PDB ID"]==sys_name]["XLR metric 0"].tolist()[0] )
		selected_xl.append( data_sat[pareto_models] )
		epoch0_viol.append( benchmark[benchmark["PDB ID"]==sys_name]["Viol0"].tolist()[0] )
		selected_viol.append( viols[pareto_models] )

		global_sat_arr = stats_dict["metadata"]["xlr"]["xl_satisfaction_array"]
		global_sat_arr = global_sat_arr[pareto_models]
		total = global_sat_arr.shape[1]
		total_sat = np.count_nonzero( np.sum( global_sat_arr, axis = 0 ) )
		global_xl.append( np.round( total_sat/total, 3 ) )
		avg_xl.append( np.round( np.mean( data_sat[pareto_models] ), 3 ) )

		structural_similarity.append( 
			all_v_all_rmsd( stats_dict["protein"],
			pareto_models )
		)

		print( sys_name )
		print( data_sat[pareto_models] )
		print( viols[pareto_models] )
		print( data_sat[pareto_models].shape[0] )
		print( "Global XL satisfaction = ", np.round( total_sat/total, 3 ) )
		print( "-"*70)
		print( "\n" )

		results_dict["pdb_id"].append( sys_name )
		results_dict["num_models"].append( len( pareto_models ) )
		results_dict["avg_viols"].append( np.round( np.mean( viols[pareto_models] ), 3 ) )
		results_dict["avg_xl_sat"].append( avg_xl[-1] )
		results_dict["global_xl_sat"].append( global_xl[-1] )
		results_dict["all_xl_sat"].append( ", ".join( map( str, data_sat[pareto_models] ) ) )
		results_dict["all_viol"].append( ", ".join( map( str, viols[pareto_models] ) ) )


	create_rmsd_plots(
		categories = categories,
		struct_sim = structural_similarity,
		file = "good_models_nds_rmsd.png"
	)

	create_metric_plots(
		categories = categories,
		selected_xl = selected_xl,
		selected_viol = selected_viol,
		epoch0_xl = epoch0_xl,
		epoch0_viol = epoch0_viol,
		global_xl = global_xl,
		avg_xl = avg_xl,
		file = "good_models_nds.png" )

	return results_dict

########################################################################
########################################################################
def gmm_clustering():
	"""
	Using GMMs for clustering models to obtain the good-scoring models.
	"""
	print( "-"*70 + "\nGMM clustering\n" + "-"*70 + "\n"+ "-"*70 )
	results_dict = {k:[] for k in ["pdb_id", "num_models",
									"avg_viols", "avg_xl_sat",
									"global_xl_sat",
									"all_xl_sat", "all_viol"]}
	selected_xl, epoch0_xl = [], []
	global_xl, avg_xl = [], []
	selected_viol, epoch0_viol = [], []
	structural_similarity = []
	categories = []
	plt.rcParams["font.family"] = "sans"
	_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )
	for sys_name in benchmark["PDB ID"]:
		categories.append( sys_name )
		sys_path = get_sys_path( sys_name )
		stats_dict = load_stat_file( sys_path )

		data_sat = np.array( stats_dict["metrics"]["xlr"] )
		viols = np.array( stats_dict["loss"]["violation"] )

		data = np.column_stack(
			[viols,
			-1*data_sat]
		)

		gm = GaussianMixture( n_components = 2,
								random_state = 1 )
		gm.fit( data )
		labels = gm.predict( data )
		clust1 = np.where( labels == 0 )
		clust2 = np.where( labels == 1 )

		if np.mean( viols[clust1] ) < np.mean( viols[clust2] ):
			good_models = clust1
		else:
			good_models = clust2

		epoch0_xl.append( benchmark[benchmark["PDB ID"]==sys_name]["XLR metric 0"].tolist()[0] )
		selected_xl.append( data_sat[good_models] )
		epoch0_viol.append( benchmark[benchmark["PDB ID"]==sys_name]["Viol0"].tolist()[0] )
		selected_viol.append( viols[good_models] )

		global_sat_arr = stats_dict["metadata"]["xlr"]["xl_satisfaction_array"]
		global_sat_arr = global_sat_arr[good_models]
		total = global_sat_arr.shape[1]
		total_sat = np.count_nonzero( np.sum( global_sat_arr, axis = 0 ) )
		global_xl.append( total_sat/total )
		avg_xl.append( np.round( np.mean( data_sat[good_models] ), 3 ) )

		structural_similarity.append( 
			all_v_all_rmsd( stats_dict["protein"],
			good_models )
		)

		print( sys_name )
		print( data_sat[good_models] )
		print( viols[good_models] )
		print( data_sat[good_models].shape[0] )
		print( "Global XL satisfaction = ", np.round( total_sat/total, 3 ) )
		print( "-"*70)
		print( "\n" )

		results_dict["pdb_id"].append( sys_name )
		results_dict["num_models"].append( viols[good_models].shape[0] )
		results_dict["avg_viols"].append( np.round( np.mean( viols[good_models] ), 3 ) )
		results_dict["avg_xl_sat"].append( np.round( np.mean( data_sat[good_models] ), 3 ) )
		results_dict["global_xl_sat"].append( np.round( total_sat/total, 3 ) )
		results_dict["all_xl_sat"].append( ", ".join( map( str, data_sat[good_models] ) ) )
		results_dict["all_viol"].append( ", ".join( map( str, viols[good_models] ) ) )

	create_rmsd_plots(
		categories = categories,
		struct_sim = structural_similarity,
		file = "good_models_gmm_rmsd.png"
	)
	create_metric_plots(
		categories = categories,
		selected_xl = selected_xl,
		selected_viol = selected_viol,
		epoch0_xl = epoch0_xl,
		epoch0_viol = epoch0_viol,
		global_xl = global_xl,
		avg_xl = avg_xl,
		file = "good_models_gmm.png" )
	return results_dict

########################################################################
########################################################################
def all_v_all_rmsd( prot_dict: Dict, good_models: np.array ):
	"""
	Given the indices for good-scoring models, compute the all-vs-all RMSD.
	"""
	# Both model_id and array index are 0-indexed.
	rmsd_list = []
	if len( good_models ) == 1:
		j_start = 0
	else:
		j_start = 1
	for i in range( len( good_models ) ):
		for j in range( j_start, len( good_models ) ):
			a = prot_dict[i].atom_positions
			b = prot_dict[j].atom_positions
			a_ca = a[:,1,:]
			b_ca = b[:,1,:]
			rot, rssd = R.align_vectors( a_ca, b_ca )
			rmsd = rssd/np.sqrt( b_ca.shape[0] )
			rmsd_list.append( rmsd )
	return rmsd_list


########################################################################
########################################################################
def create_rmsd_plots( categories: List, struct_sim: List, file: str ):
	plt.rcParams["font.family"] = "sans"
	_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )
	ax.violinplot( struct_sim )
	ax.set_title( "Structural similarity", fontsize = 25 )
	ax.set_ylabel( "RMSD wrt epoch0", fontsize = 25 )
	ax.tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )
	ax.set_xticks( np.arange(1, len( categories ) + 1 ), categories )

	path = os.path.join( curr_dir, file )
	plt.savefig( path, dpi = 300 )
	plt.close()


def create_metric_plots(
		categories: List,
		selected_xl: List,
		selected_viol: List,
		epoch0_xl: List,
		epoch0_viol: List,
		global_xl: List,
		avg_xl: List,
		file: str ):
	plt.rcParams["font.family"] = "sans"
	_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )
	ax[0].violinplot( selected_xl )
	for i in range( len( epoch0_xl ) ):
		x_jitter = np.random.normal( i+1, 0.04, size = 1 )
		ax[0].scatter( x_jitter, epoch0_xl[i], color = "red", s = 70, alpha = 1, label = "epoch0" if i == 1 else "" )
		ax[0].scatter( x_jitter, global_xl[i], color = "green", s = 70, alpha = 1, label = "global" if i == 1 else "" )
		ax[0].scatter( x_jitter, avg_xl[i], color = "orange", s = 70, alpha = 1, label = "avg" if i == 1 else "" )
	ax[0].set_title( "XL satisfaction", fontsize = 25 )
	ax[0].tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )
	ax[0].set_xticks( np.arange(1, len( categories ) + 1 ), categories )
	
	ax[1].violinplot( selected_viol )
	for i in range( len( epoch0_viol ) ):
		x_jitter = np.random.normal( i+1, 0.04, size = 1 )
		ax[1].scatter( x_jitter, epoch0_viol[i], color = "red", s = 70, alpha = 1, label = "epoch0" if i == 1 else "" )
	ax[1].set_title( "Violations", fontsize = 25 )
	ax[1].tick_params( axis = "both" , labelsize = 18, length = 10, width = 4 )
	ax[1].set_xticks( np.arange(1, len( categories ) + 1 ), categories )

	ax[0].legend( fontsize = 16 )
	ax[1].legend( fontsize = 16 )
	path = os.path.join( curr_dir, file )
	plt.savefig( path, dpi = 300 )
	plt.close()

########################################################################
########################################################################
plot_per_epoch_distribution()
print( "\n" )
plot_structural_similarity()
print( "\n" )
quant_results = quantile_filtering()
print( "\n" )
nds_results = nondominant_sorting()
print( "\n" )
gmm_results = gmm_clustering()

for k in quant_results:
	df = pd.DataFrame( quant_results[k] )
	df.to_csv( f"good_models_{k}.csv" )

df = pd.DataFrame( nds_results )
df.to_csv( f"good_models_nds.csv" )

df = pd.DataFrame( gmm_results )
df.to_csv( f"good_models_gmm.csv" )

print( "may the Force be with you..." )
