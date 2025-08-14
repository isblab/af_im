"""
Test whether AF2 learns a rigid prior or not.
Wrapper over the BenchmarkModeling module.
	Here, I do not run the analyis module and also
		switch off creating plots in BenchmarkModeling module.
"""
from typing import List, Tuple, Dict
import os, glob, pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from benchmark_runs import BenchmarkModeling
from utils.paths import (
	get_stat_file_path,
	get_sys_data_dir_path,
	get_benchmark_results_dir_path,
	get_unrelaxed_model_file,
	get_unrelaxed_ensemble_file )
from utils.tools import (
	get_residue_ids,
	compute_rmsf_wrt_avg_model,
	compute_rmsf_wrt_first_model,
	compute_rmsd_post_align )

"""
Using a fixed set of xl weights. Each corresponds to a modeling version.
For subsequent experiments with loss and XL type, using sub_version
	to distinguish the runs and keep my sanity intact.
"""

class RigigPrior():
	"""
	Run the simulations to test the rigid prior theory.
	Use the BenchmarkModeling module for running simulations.
	"""
	def __init__( self ):
		self.benchmark_name = "rigid"
		self.xl_restraint_weights = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
		self.sub_version = 0.1
		# Provide shorthand for the loss to be included.
		self.include_loss = ""
		# File format to save the structure.
		self.struct_format = "pdb"
		# Suffix for the sys_config and xls file.
		self.sys_conf_suff = "_1fp"
		self.remove_sys_modeling_dir = False


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.load_benchmark()
		# self.get_modeling_versions()

		for sv, loss in zip( [0, 0.1, 0.2, 0.3, 0.4, 0.5], ["", "v", "i", "s", "vi", "vs"] ):
			self.sub_version = sv
			self.include_loss = loss
			self.run_modeling()

		# self.convert_ensemble_to_gif()

	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		self.base_dir = os.path.join( os.path.abspath( "./benchmark" ) )
		self.meta_dir = os.path.join(
			self.base_dir,
			f"{self.benchmark_name}_metadata" )
		# Name for the dir to store modeling output for all systems.
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"

		self.benchmark_csv_file = os.path.join(
			self.meta_dir,
			f"selected_{self.benchmark_name}_benchmark.csv" )

		# this dir stores all the plots created by this script.
		# 	Separates from the benchmark_results directory.
		self.rigid_prior_results_dir = os.path.join(
			self.base_dir,
			"rigid_prior_plots" )
		os.makedirs( self.rigid_prior_results_dir, exist_ok = True )


	def create_plots_dir( self ):
		"""
		Create the required directories.
		"""
		self.plots_dir = os.path.join(
			self.rigid_prior_results_dir,
			f"plots{self.sys_conf_suff}" )
		os.makedirs( self.plots_dir, exist_ok = True )


	def load_benchmark( self ):
		"""
		Load the benchmak .csv file.
		"""
		self.benchmark = pd.read_csv( self.benchmark_csv_file )


	def get_modeling_versions( self ):
		"""
		Create a dict containing the version for the modeling run.
		"""
		if self.sys_conf_suff == "_1fp":
			start, end = 1, 5
		elif self.sys_conf_suff == "_1tp":
			start, end = 6, 10

		self.modeling_versions = np.arange(
			start+self.sub_version, end+self.sub_version + 1, 1
			)

	################################################################################
	################################################################################
	def run_modeling( self ):
		"""
		Run modeling for all required parameters.
		"""
		for suff in ["_1fp", "_1tp"]:
			self.sys_conf_suff = suff
			print( f"Config file suffix: {self.sys_conf_suff}..." )
			self.create_plots_dir()
			self.get_modeling_versions()
			print( self.modeling_versions )
			for i, xl_weight in enumerate( self.xl_restraint_weights ):
				if self.sub_version == 0:
					modeling_version = int( self.modeling_versions[i] )
				else:
					modeling_version = float( self.modeling_versions[i] )

				print( f"\n\033[1mModeling version = {modeling_version}" +
						f"\t xl_weight = {xl_weight}\t" +
						f"losses = {self.include_loss} \033[0m\n" )
				self.run_modeling_benchmark(
					modeling_version = modeling_version,
					xl_weight = xl_weight
					)

			self.plot_modeling_results()


	def run_modeling_benchmark( self,
		modeling_version: int,
		xl_weight: float ):
		"""
		Instantiate the BenchmarkModeling module and run the simulations.
		"""
		objective = f"Rigid prior experiments. Using xl_weight = {xl_weight} and losses = {self.include_loss}"
		sim = BenchmarkModeling()
		sim.benchmark_name = self.benchmark_name
		sim.modeling_version = modeling_version
		sim.modeling_objective = objective
		sim.sys_conf_suff = self.sys_conf_suff
		sim.device = "cuda:0"
		sim.enable_relax_validate = False
		sim.skip_rerun = True
		sim.remove_sys_modeling_dir = self.remove_sys_modeling_dir
		sim.xlr["add_penalty"] = True
		sim.xlr["weight"] = xl_weight
		sim.create_summary_plots_and_files = False

		self.enable_weights( obj = sim )

		sim.forward()


	def enable_weights( self, obj: BenchmarkModeling ):
		"""
		Enable the required weights as specified in self.include_loss.
		Using shorthands for the different loss combinations.
			v -> violations loss.
			c -> ccom loss.
			f -> full FAPE (backbone+interface+sidechain) loss
			i -> only backbone+sidechain FAPE loss (intrafape)
			s -> supervised_chi loss
		e.g. vis -> violation + intrafape + supervised_chi loss
		"""
		if "v" in self.include_loss:
			obj.violation["add_penalty"] = True
		else:
			obj.violation["add_penalty"] = False

		if "c" in self.include_loss:
			obj.ccom["add_penalty"] = True
		else:
			obj.ccom["add_penalty"] = False

		if "f" in self.include_loss:
			obj.fape["add_penalty"] = True
		else:
			obj.fape["add_penalty"] = False

		if "i" in self.include_loss:
			obj.fape["add_penalty"] = True
			obj.fape["interfape"] = True
		else:
			obj.fape["interfape"] = False

		if "s" in self.include_loss:
			obj.supervised_chi["add_penalty"] = True
		else:
			obj.supervised_chi["add_penalty"] = False

	################################################################################
	################################################################################
	def plot_modeling_results( self ):
		"""
		Create the following plots:
			Per-epoch loss and xl restraint.
			RMSD wrt 1st model.
			RMSF wrt the average model.
		"""
		print( "\n" + "-"*70 + "\n" + "-"*70 + "\n\033[1mRigid prior plots\033[0m\n" )
		for sys_name in self.benchmark["PDB ID"]:
			sys_plot_dir = os.path.join( self.plots_dir, sys_name )
			os.makedirs( sys_plot_dir, exist_ok = True )
			print( f"Creating per-epoch loss plots for {sys_name}..." )
			self.per_epoch_loss_plots(
				sys_name = sys_name,
				sys_plot_dir = sys_plot_dir
				)

			print( f"Creating RMSD plots for {sys_name}..." )
			self.plot_rmsd_across_runs(
				sys_name = sys_name,
				sys_plot_dir = sys_plot_dir
				)

			print( f"Creating RMSF plots for {sys_name}..." )
			self.plot_rmsf(
				sys_name = sys_name,
				sys_plot_dir = sys_plot_dir
				)


	def get_input_for_per_epoch_plots( self,
		sys_name: str,
		modeling_version: int
		) -> Tuple[List[float], List[float], List[int]]:
		"""
		Obtain violation loss and xl restraint per epoch for the given modeling version.
		"""
		stat_file_path = get_stat_file_path(
			base_dir = "./benchmark",
			modeling_dir_name = self.modeling_dir_name,
			sys_name = sys_name,
			modeling_version = modeling_version )
		stats_dict = np.load( stat_file_path, allow_pickle = True ).item()

		violation = stats_dict["loss"]["violation"]
		xlr = stats_dict["loss"]["xlr"]
		# model_id is the same as epoch.
		epochs = stats_dict["model_id"]
		return violation, xlr, epochs


	def per_epoch_loss_plots( self, sys_name: str, sys_plot_dir: str ):
		"""
		Plot the per-epoch xl restraint and violation loss across all runs.
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )

		for i, xl_weight in enumerate( self.xl_restraint_weights ):
			modeling_version = self.modeling_versions[i]
			violation, xlr, epochs = self.get_input_for_per_epoch_plots(
				sys_name = sys_name,
				modeling_version = modeling_version )

			ax[0].plot( epochs, violation, label = xl_weight )
			ax[0].set_xlabel( "No. of epochs", fontsize = 20 )
			ax[0].set_ylabel( "Violation loss", fontsize = 20 )
			ax[0].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[0].legend( fontsize = 14 )

			ax[1].plot( epochs, xlr, label = xl_weight )
			ax[1].set_xlabel( "No. of epochs", fontsize = 20 )
			ax[1].set_ylabel( "XL restraint", fontsize = 20 )
			ax[1].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[1].legend( fontsize = 14 )

		ax[0].set_xlim( -0.1 )
		ax[0].set_ylim( -0.1 )
		ax[1].set_xlim( -0.1 )
		ax[1].set_ylim( -0.1 )
		loss_plot_file = os.path.join(
			sys_plot_dir, f"loss_per_epoch_{sys_name}_{self.include_loss}.png" )
		plt.savefig( loss_plot_file, dpi = 300, bbox_inches = "tight" , pad_inches = 0.1 )
		plt.close()

	################################################################################
	def plot_rmsd_across_runs( self,  sys_name: str, sys_plot_dir: str ):
		"""
		Across all runs, plot the RMSD for all models wrt the 1st model.
		"""
		plt.rcParams["font.family"] = "sans"
		# cmap = plt.get_cmap( "RdYlGn_r" )
		# norm = plt.Normalize( min( self.xl_restraint_weights ), max( self.xl_restraint_weights ) )
		_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )

		for i, xl_weight in enumerate( self.xl_restraint_weights ):
			modeling_version = self.modeling_versions[i]
			ensemble_file = get_unrelaxed_ensemble_file(
				base_dir = self.base_dir,
				modeling_dir_name = self.benchmark_name,
				sys_name = sys_name,
				modeling_version = modeling_version,
				struct_format = self.struct_format
				)

			stat_file_path = get_stat_file_path(
				base_dir = "./benchmark",
				modeling_dir_name = self.modeling_dir_name,
				sys_name = sys_name,
				modeling_version = modeling_version )
			stats_dict = np.load( stat_file_path, allow_pickle = True ).item()
			epochs = stats_dict["model_id"]

			rmsd = compute_rmsd_post_align( ensemble_file = ensemble_file )
			rmsd = rmsd[:,2]

			# color = cmap( norm( xl_weight ) )
			ax.plot( epochs, rmsd, label = xl_weight )
			ax.set_xlabel( "No. of epochs", fontsize = 25 )
			ax.set_ylabel( "RMSD wrt first model", fontsize = 25 )
			ax.tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax.legend( fontsize = 14 )

		ax.set_xlim( -0.1 )
		ax.set_ylim( -0.1 )
		rmsd_plot_file = os.path.join(
			sys_plot_dir, f"rmsd_plots_{sys_name}_{self.include_loss}.png" )
		plt.savefig( rmsd_plot_file, dpi = 300, bbox_inches = "tight" , pad_inches = 0.1 )
		plt.close()

	################################################################################
	def get_input_for_rmsf_plots( self, sys_name: str, ensemble_file: str ):
		"""
		Given an ensemble file, compute,
			RMSF wrt average model
			Get pLDDT of initial predicted structure
		"""
		rmsf_avg = compute_rmsf_wrt_avg_model( ensemble_file = ensemble_file )
		# rmsf_first = compute_rmsf_wrt_first_model( ensemble_file = ensemble_file )
		resids = get_residue_ids( ensemble_file )

		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		out_dict_path = f"{data_dir}{sys_name}_output/predictions/*_output_dict.pkl"
		init_model_out_file = glob.glob( out_dict_path )
		if len( init_model_out_file ) == 0:
			raise FileNotFoundError( f"output_dict file not found for {sys_name} --> {out_dict_path}..." )
		init_model_out_file = init_model_out_file[0]
		with open( init_model_out_file, "rb" ) as f:
			out_dict = pkl.load( f )
		plddt = out_dict["plddt"]

		return rmsf_avg, plddt, resids


	def plot_rmsf( self, sys_name: str, sys_plot_dir: str ):
		"""
		Across all runs,
			Plot RMSF wrt average model (this is essentially per-residue model precision)
			RMSF vs pLDDT of initial predicted structure
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( len( self.xl_restraint_weights ), 2, figsize = ( 20, 40 ) )
		for i, xl_weight in enumerate( self.xl_restraint_weights ):
			modeling_version = self.modeling_versions[i]

			ensemble_file = get_unrelaxed_ensemble_file(
				base_dir = self.base_dir,
				modeling_dir_name = self.benchmark_name,
				sys_name = sys_name,
				modeling_version = modeling_version,
				struct_format = self.struct_format
				)

			rmsf_avg, plddt, resids = self.get_input_for_rmsf_plots(
				sys_name = sys_name,
				ensemble_file = ensemble_file )
			# resids start from 1 for all chains. So, using system indices.
			sys_idx = np.arange( 0, resids.shape[0], 1 )

			corrs, _ = spearmanr( rmsf_avg, plddt )
			corrs = np.round( corrs, 3 )
			# RMSF wrt avg model per run.
			ax[i, 0].plot( sys_idx, rmsf_avg, label = "RMSF" )
			ax[i, 0].plot( sys_idx, plddt, label = "pLDDT" )
			ax[i, 0].set_title( f"XL weight = {xl_weight}; Spearman = {corrs}", fontsize = 20 )
			ax[i, 0].set_xlabel( "Residue numbers", fontsize = 20 )
			ax[i, 0].set_ylabel( "RMSF wrt average model", fontsize = 20 )
			ax[i, 0].set_xlim( -0.5 )
			ax[i, 0].set_ylim( -0.5 )
			ax[i, 0].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[i, 0].legend( fontsize = 14 )

			# RMSF wrt avg model per run.
			ax[i, 1].scatter( plddt, rmsf_avg )
			ax[i, 1].set_title( f"XL weight = {xl_weight}; Spearman = {corrs}", fontsize = 20 )
			ax[i, 1].set_xlabel( "pLDDT", fontsize = 20 )
			ax[i, 1].set_ylabel( "RMSF wrt average model", fontsize = 20 )
			ax[i, 1].set_xlim( -0.5 )
			ax[i, 1].set_ylim( -0.5 )
			ax[i, 1].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )

		plt.subplots_adjust( hspace = 0.5, wspace = 0.4 )

		rmsf_plot_file = os.path.join(
			sys_plot_dir, f"rmsf_plots_{sys_name}_{self.include_loss}.png" )
		plt.savefig( rmsf_plot_file, dpi = 300, bbox_inches = "tight" , pad_inches = 0.1 )
		plt.close()

	################################################################################
	################################################################################
	# def convert_ensemble_to_gif( self,  modeling_versions: List ):
	# 	"""
	# 	Craete .gifs for the ensemble in VMD.
	# 	"""
	# 	print( "\n" + "-"*70 + "\n" + "-"*70 + "\n\033[1mEnsemble to GIFs\033[0m\n" )
	# 	for viol_label in modeling_versions:
	# 		for sys_name in self.benchmark["PDB ID"]:
	# 			print( f"Creating GIFs for {sys_name}..." )

	# 			for i, xl_weight in enumerate( self.xl_restraint_weights ):
	# 				modeling_version = modeling_versions[viol_label][i]

	# 				self.run_vmd(
	# 					sys_name = sys_name,
	# 					modeling_version = modeling_version
	# 					)


	# def run_vmd( self, sys_name: str, modeling_version: int ):
	# 	"""
	# 	Run VMD to 
	# 	"""		
	# 	ensemble_file = get_unrelaxed_ensemble_file(
	# 		base_dir = self.base_dir,
	# 		modeling_dir_name = self.benchmark_name,
	# 		sys_name = sys_name,
	# 		modeling_version = modeling_version,
	# 		struct_format = self.struct_format
	# 		)
	# 	sys_modeling_version_dir = get_sys_modeling_version_path(
	# 		base_dir: str,
	# 		modeling_dir_name: str,
	# 		sys_name: str, modeling_version )
	# 	output_file = os.path.join(
	# 		sys_modeling_version_dir,
	# 		f"{sys_name}_v{modeling_version}.gif" )
	# 	tmp_file = os.path.join( self.base_dir, f"tmp_{sys_name}" )
	# 	speed = 0.9
	# 	movie_duration = 10
	# 	cmd = ["vmd", "-dispdev", "text",
	# 	"-e", f"{self.vmd_script}",
	# 	"-args", f"{ensemble_file}",
	# 	f"{output_file}",
	# 	f"{tmp_file}",
	# 	f"{speed}",
	# 	f"{movie_duration}"
	# 	]
	# 	run_subprocess( cmd )


if __name__ == "__main__":
	RigigPrior().forward()

