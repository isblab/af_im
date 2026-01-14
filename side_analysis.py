"""
Contains some wrapper for perforimng analysis for answering specific questions:
	1. Distribution of max data satisfaction per frame from pose sampler.
	2. Why use num_steps=20?
	3. Distribution of predicted rotations.
"""
from typing import List, Tuple, Dict, Any, Iterator
import os, pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import open_file_handler
from utils.paths import (
	get_meta_dir_path,
	get_sys_data_dir_path,
	get_benchmark_csv_file,
	get_sys_modeling_path,
	get_stat_file_path,
	get_native_struct_file,
	get_init_pred_file
	)
from utils.pdb_utils import get_chain_id, remap_chains_cif


base_dir = "/data2/kartik/IMP_Rewired/imp_dl/benchmark"
benchmark_name = "xlmerged"
meta_dir = get_meta_dir_path(
	base_dir = base_dir,
	benchmark_name = benchmark_name )
modeling_dir_name = f"{benchmark_name}_modeling"
# Load the banchmark csv.
csv_file = get_benchmark_csv_file(
	base_dir = base_dir,
	benchmark_name = benchmark_name )
df = pd.read_csv( csv_file )

# Dir to save analysis results.
side_analysis_dir = os.path.join( base_dir, f"side_analysis_{benchmark_name}/" )
os.makedirs( side_analysis_dir, exist_ok = True )


################################################################################
################################################################################
def get_pose_dict( sys_name: str, version: Any ) -> Dict[str, Any]:
	"""
	Return the stats dict for pose sampling.
	"""
	modeling_dir = get_sys_modeling_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name )
	file = os.path.join( modeling_dir, f"version_{version}/", "Stats_pose.npy" )

	stats_pose_dict = np.load( file, allow_pickle = True ).item()
	return stats_pose_dict


def yield_sys_name() -> Iterator[str]:
	"""
	Generator that yields the sys_name from the benchmark.
	"""
	for sys_name in df["PDB ID"]:
		yield sys_name

################################################################################
################################################################################
def get_data_satisfaction( version: Any, pose_sampling: True ):
	"""
	For all complexes in the benchmark get the data satisfcation:
		Per frame
		Per cycle (pose smapling only)
	"""
	complexes = []
	data_satisfaction = []
	
	for sys_name in yield_sys_name():
		complexes.append( sys_name )
		if pose_sampling:
			try:
				stats = get_pose_dict( sys_name = sys_name, version = version )		
			except:
				continue
		else:
			stats_file = get_stat_file_path(
				base_dir = base_dir,
				sys_name = sys_name,
				modeling_dir_name = modeling_dir_name,
				modeling_version = version )
			if not os.path.exists( stats_file ):
				continue
			stats = np.load( stats_file, allow_pickle = True ).item()
		data_satisfaction.append( np.array( stats["metrics"]["xlr"] ) )
	return complexes, data_satisfaction


def get_max_data_satisfaction( version: Any ):
	"""
	For all complexes in the benchmark get the max
		data satisfcation for each round of pose sampling.
	version_5 -> only pose sampling.
	"""
	max_data_sat = []
	complexes, data_satisfaction = get_data_satisfaction( version = version )
	for data in data_satisfaction:
		data_sat = np.array( data )
		# Split into num_frames.
		split_by_frame = np.array_split( data_sat, 50-1 )
		data_sat_frame = np.stack( split_by_frame )

		# get max data satisfaction in each frame.
		max_sat = np.max( data_sat_frame, axis = 1 )
		max_data_sat.append( list( max_sat ) )
	return complexes, max_data_sat


def plot_dist_max_data_sat():
	"""
	For xlmerged benchmark, plot the distribution of max data
		satisfaction across at every frame frame from the pose sampler.
	"""
	print( "Creating max data satisfaction plot..." )
	complexes, max_data_sat = get_max_data_satisfaction( version = 5 )
	complex_idx = np.arange( 1, len( complexes ) +1 )

	_, ax = plt.subplots( 1, 1, figsize = ( 15, 8 ) )
	vp = ax.violinplot( max_data_sat, showmeans = True )
	# Add the mean line.
	vp["cmeans"].set_color( "red" )
	vp["cmeans"].set_linewidth( 2 )
	ax.set_xticks( complex_idx, complexes )
	ax.tick_params( axis = "both" , labelsize = 8, length = 5, width = 2 )
	plt.tight_layout()
	plt.savefig( f"{side_analysis_dir}/max_data_sat.png", dpi = 300 )

################################################################################
################################################################################
def pose_sampling_convergence():
	"""
	Compare the distribution of data satisfaction when pose sampling for 20 and 500 epochs.
		version_11 -> pose sampling
		version_11.1 -> random pose sampling
	"""
	print( "Creating convergence plot..." )
	_, ax = plt.subplots( 2, 1, figsize = ( 20, 10 ) )
	for i, version in enumerate( [11, 11.1] ):
		title = "Pose sampling" if i == 0 else "Random Pose sampling"
		complexes, data_satisfaction = get_data_satisfaction( version = version )
		steps20, steps500 = [], []
		for data in data_satisfaction:
			data = np.array( data )
			steps20.append( data[:20] )
			steps500.append( data )
		
		v1 = ax[i].violinplot( steps20, showmeans = True )
		v1["bodies"][0].set_label( "20" )
		v2 = ax[i].violinplot( steps500, showmeans = True )
		v2["bodies"][0].set_label( "500" )
		complex_idx = np.arange( 1, len( complexes )+1 )
		ax[i].set_title( title, fontsize = 14 )
		ax[i].set_xticks( complex_idx, complexes )
		ax[i].tick_params( axis = "both" , labelsize = 10, length = 5, width = 2 )
		ax[i].legend()
	plt.tight_layout()
	# plt.legend()
	plt.savefig( f"{side_analysis_dir}/convergence.png", dpi = 300 )

################################################################################
################################################################################
def quat_to_angles( quat: np.array ) -> np.ndarray:
	"""
	Convert the quaternion to an angle.
	radians = 2*arccos( q_w )
	"""
	w = quat[..., 0]
	radians = 2*np.arccos( w )
	degrees = radians*( 180/np.pi )
	return degrees


def get_pred_angles( version: Any ):
	"""
	Get predicted angles in degrees.
	rotation -> {N,M,R,4}
		N -> num_frames; M -> num_steps; R -> no. of rigid bodies.
	"""
	complexes = []
	angles = []
	for sys_name in yield_sys_name():
		complexes.append( sys_name )
		stats_file = get_stat_file_path(
			base_dir = base_dir,
			sys_name = sys_name,
			modeling_dir_name = modeling_dir_name,
			modeling_version = version )
		stats = np.load( stats_file, allow_pickle = True ).item()
		rotation = np.array( stats["transformations"]["rotation"] )[0]
		angles.append( list( quat_to_angles( rotation ) ) )
	return complexes, angles


def sampled_rotations():
	"""
	I want visualize the distribution of rotations.
	For this I will convert the quaternion to an angle in degrees.
	"""
	print( "Creating angule distribution plots" )
	_, ax = plt.subplots( 2, 1, figsize = ( 20, 10 ) )
	for i, version in enumerate( [11, 11.1] ):
		title = "Pose sampling" if i == 0 else "Random Pose sampling"
		complexes, angles = get_pred_angles( version = version )
		angles20, angles500 = [], []
		for angle in angles:
			angle = np.array( angle )
			angles20.append( angle[:20,:].reshape( -1 ) )
			angles500.append( angle.reshape( -1 ) )

		v1 = ax[i].violinplot( angles20, showmeans = True )
		v1["bodies"][0].set_label( "20" )
		v2 = ax[i].violinplot( angles500, showmeans = True )
		v2["bodies"][0].set_label( "500" )
		complex_idx = np.arange( 1, len( complexes )+1 )
		ax[i].set_title( title, fontsize = 14 )
		ax[i].set_ylim( 0, 360 )
		ax[i].set_xticks( complex_idx, complexes )
		ax[i].tick_params( axis = "both" , labelsize = 10, length = 5, width = 2 )
		ax[i].legend()
	plt.tight_layout()
	plt.savefig( f"{side_analysis_dir}/Angles.png", dpi = 300 )

################################################################################
################################################################################
def get_modeling_runs_to_compare():
	"""
	Return a list of tuple for the modeling versions to be compared.
	"""
	return [
		( 5, 6 ),
		( 5, 7 ),
		( 5, 8 ),
		# ( 5, 13 ),
		# ( 12, 5 ),
		# ( 12, 6 ),
		# ( 12, 7 ),
		# ( 12, 8 ),
		# ( 12, 13 ),
	]


def compare_max_data_satisfaction():
	"""
	Compare different modeling runs for the mean data satisfaction
		achieved on the benchmark.
	"""
	for v1, v2 in get_modeling_runs_to_compare():
		print( f"Obtaining max data satisfaction per frame for version_{v1}" )
		_, data_sat1 = get_data_satisfaction( version = v1, pose_sampling = False )
		print( f"Obtaining max data satisfaction per frame for version_{v2}" )
		_, data_sat2 = get_data_satisfaction( version = v2, pose_sampling = False )

		data_sat1 = np.mean( np.array( data_sat1 ), axis = 1 )
		data_sat2 = np.mean( np.array( data_sat2 ), axis = 1 )

		count1 = np.sum( data_sat1 > data_sat2 )
		print( f"Mean data satisfaction for version_{v1} > version_{v2} = {count1}" )
		count2 = np.sum( data_sat1 < data_sat2 )
		print( f"Mean data satisfaction for version_{v2} > version_{v1} = {count2}\n" )
		print( "------\n" )

################################################################################
################################################################################
def get_chain_mapping( native_chain_ids: str ) -> Dict[str, str]:
	"""
	Map the native chain IDs to the system chain IDs.
	"""
	map_dict = {}
	idx = 0
	for chains in native_chain_ids.split( "," ):
		for chain_id in chains.split( ":" ):
			sys_chain_id = get_chain_id( idx = idx )
			map_dict[chain_id] = sys_chain_id
			idx += 1
	return map_dict


def remap_chains_in_native():
	"""
	For running DockQ, the chain IDs in the native
		and predicted structures must be the same.
	We remap chains in the native structure as its cheaper
		than doing so for all predicted structures.
	"""
	for i, sys_name in enumerate( df["PDB ID"] ):
		native_chain_ids = df.loc[i, "Auth Asym ID"]
		print( f"{i}. {sys_name}:: {native_chain_ids}" )
		native_struct_file = get_native_struct_file(
			base_dir = base_dir,
			benchmark_name = benchmark_name,
			sys_name = sys_name )

		map_dict = get_chain_mapping( native_chain_ids )
		remap_chains_cif( struct_file = native_struct_file, map_dict = map_dict )

################################################################################
################################################################################
def pad_array( arr: np.ndarray, max_len: int, dims: str ) -> np.ndarray:
	"""
	Pad the input array to max length.
	"""
	if dims == "1d":
		pad = np.zeros( [max_len] )
		m = arr.shape[0]
		pad[:m] = arr
	elif dims == "2d":
		pad = np.zeros( [max_len, max_len] )
		m, n = arr.shape
		pad[:m,:n] = arr
	else:
		raise ValueError( f"Padding arrays of size {dims} not supported..." )

	return pad


def pad_n_stack( metric_dict: Dict[str, Any]
	) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Pad pLDDT and PAE arrays to max length.
	Stack them together.
	"""
	max_len = max( metric_dict["length"] )
	plddt_arr, pae_arr = [], []
	for plddt in metric_dict["plddt"]:
		plddt_arr.append(
			pad_array( plddt, max_len = max_len, dims = "1d" ) )
	plddt_arr = np.stack( plddt_arr )

	for pae in metric_dict["pae"]:
		pae_arr.append(
			pad_array( pae, max_len = max_len, dims = "2d" )  )
	pae_arr = np.stack( pae_arr )

	return plddt_arr, pae_arr


def get_conf_metrics_per_sys( sys_name: str
	) -> Tuple[np.ndarray, np.ndarray, int]:
	"""
	Return the pLDDT and PAE as np.ndarray for the given system.
	Also return the length of the system.
	"""
	out_dict_file = get_init_pred_file(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	f = open_file_handler( out_dict_file, "rb" )
	out = pkl.load( f )
	f.close()

	plddt = np.array( out["plddt"] )
	pae = np.array( out["predicted_aligned_error"] )
	pae = ( pae + pae.T )/2
	length = plddt.shape[0]
	print( plddt.shape, "  ", pae.shape, "  ", length )
	return plddt, pae, length


def get_conf_metrics_for_benchmark() -> Tuple[np.ndarray, np.ndarray]:
	"""
	Return the pLDDT and PAE as np.ndarray for the entire benchmark.
	Pad them upto max length.
	"""
	file = os.path.join( side_analysis_dir, "conf_metrics.npy" )

	if os.path.exists( file ):
		dict_ = np.load( file, allow_pickle = True ).item()
		complexes = dict_["complex"]
		plddt, pae = dict_["plddt"], dict_["pae"]
	else:
		metric_dict = {k:[] for k in ["complex", "plddt", "pae", "length"]}
		i = 0
		for sys_name in df["PDB ID"]:
			plddt, pae, length = get_conf_metrics_per_sys( sys_name = sys_name )
			metric_dict["complex"].append( sys_name )
			metric_dict["plddt"].append( plddt )
			metric_dict["pae"].append( pae )
			metric_dict["length"].append( length )
			# i += 1
			# if i == 5:
			# 	break

		plddt, pae = pad_n_stack( metric_dict = metric_dict )
		complexes = np.stack( metric_dict["complex"] )

		np.save( file,
			{"complex": complexes, "plddt": plddt, "pae": pae},
			allow_pickle = True )

	print( plddt.shape, "  ", pae.shape )
	return plddt, pae, complexes


def plot_confidence_metrics_for_benchmark():
	"""
	Plot the pLDDT and PAE metrics for the entire benchmark.
	"""
	plddt, pae, complexes = get_conf_metrics_for_benchmark()

	plddt_plot_file = os.path.join( side_analysis_dir, f"plddt_{benchmark_name}.png" )
	plt.figure( figsize = ( 10, 15 ) )
	ax = sns.heatmap( plddt, cmap = "Greens" )
	ax.set_yticklabels( complexes, rotation = 0 )
	plt.tight_layout()
	plt.savefig( plddt_plot_file, dpi = 399 )
	plt.close()

	pae_plot_file = os.path.join( side_analysis_dir, f"pae_{benchmark_name}.png" )
	plt.rcParams["font.family"] = "sans"
	fig, ax = plt.subplots( 8, 4, figsize = ( 20, 10 ) )
	fig.subplots_adjust(
		left = 0.01,
		right = 0.98,
		top = 0.95,
		bottom = 0.05,
		wspace = 0.01,
		hspace = 0.01
	)
	i = 0
	for r in range( 8 ):
		for c in range( 4 ):
			ax[r, c].imshow( pae[i], cmap = "Greens_r" )
			ax[r, c].set_title( complexes[i], fontsize = 12 )
			i += 1
	plt.tight_layout()
	plt.savefig( pae_plot_file, dpi = 399 )
	plt.close()


if __name__ == "__main__":
	# plot_dist_max_data_sat()
	# pose_sampling_convergence()
	# sampled_rotations()
	# compare_max_data_satisfaction()
	# remap_chains_in_native()
	plot_confidence_metrics_for_benchmark()
	print( "May the Force be with you..." )
