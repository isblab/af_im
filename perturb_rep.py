"""
Modifying the single or pair representation to incorporate data.
Wrapper over the BenchmarkModeling module.
	Here, I do not run the analyis module and also
		switch off creating plots in BenchmarkModeling module.
"""
from typing import List, Tuple, Dict
import os, glob, pickle as pkl
import numpy as np
import pandas as pd
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from topology import topology_dict
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


class PerturbRepresentation():
	"""
	Testing different way of modifying sigle or pair representation to incorporate data.
	Use the BenchmarkModeling module for running simulations.
	"""
	def __init__( self ):
		self.benchmark_name = "rigidchain"
		# File format to save the structure.
		self.struct_format = "pdb"
		# Suffix for the sys_config and xls file.
		self.sys_conf_suff = "_tpfp"
		self.device = "cuda:0"
		self.enable_relax_validate = False
		self.remove_sys_modeling_dir = False
		self.create_summary_plots_and_files = False


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.load_benchmark()
		# self.get_modeling_versions()

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

		# if self.benchmark_name == "pairrep":
		# 	self.output_dir = os.path.join( self.base_dir, "pair_perturbation" )
		# elif self.benchmark_name == "singlerep":
		# 	self.output_dir = os.path.join( self.base_dir, "single_perturbation" )

		# self.output_dir = os.path.join( self.base_dir, "alternate_loss" )
		self.output_dir = os.path.join( self.base_dir, "rigidchain_loss" )
		os.makedirs( self.output_dir, exist_ok = True )



	def load_benchmark( self ):
		"""
		Load the benchmak .csv file.
		"""
		self.benchmark = pd.read_csv( self.benchmark_csv_file )

	################################################################################
	################################################################################
	def run_modeling( self ):
		"""
		Run modeling for all required parameters.
		"""
		# Run the experiment and create the plots.
		# self.experiment1()
		# self.experiment2()
		# self.experiment3()
		self.experiment4()


	def init_modeling_obj( self, topo_dict: ConfigDict ):
		"""
		Initialize the modeling object.
		"""
		sim = BenchmarkModeling( topo_dict )
		sim.benchmark_name = self.benchmark_name
		sim.sys_conf_suff = self.sys_conf_suff
		sim.device = self.device
		sim.enable_relax_validate = self.enable_relax_validate
		sim.skip_rerun = True
		sim.remove_sys_modeling_dir = self.remove_sys_modeling_dir
		sim.create_summary_plots_and_files = self.create_summary_plots_and_files
		return sim


	################################################################################
	def experiment1( self ):
		"""
		Try different ways for modifying the single or pair representation.
		"""
		# Name of the experiment.
		self.expt_name = "experiment1"
		plot_suffix = ""
		# Parameters to be tested.
		self.expt_params = ["linear_perturb", "sigmoid_gating", "tanh_gating", "lora", "film"]
		total_expt = len( self.expt_params )
		# versions.
		start, end = 0, len( self.expt_params )
		step = 1
		self.modeling_versions = list( map( int, np.arange( start, end, step ) ) )

		self.expt_dir = os.path.join( self.output_dir, self.expt_name )
		os.makedirs( self.expt_dir, exist_ok = True )

		for expt_param, ver in zip( self.expt_params, self.modeling_versions ):
			print( f"\n\033[1mModeling version = {ver}" +
					f"Experiment: {expt_param}\033[0m" )
			objective = f"Experiment: {self.expt_name} - testing = {expt_param}"

			topo_dict = topology_dict()
			topo_dict.loss.fape.add_penalty = False
			topo_dict.loss.supervised_chi.add_penalty = False
			topo_dict.loss.chain_center_of_mass.add_penalty = False
			topo_dict.loss.violation.add_penalty = True
			topo_dict.loss.violation.weight = 0.03
			topo_dict.loss.xlr.add_penalty = True
			topo_dict.loss.xlr.weight = 0.05

			if self.benchmark_name == "pairrep":
				topo_dict.model.name = "pair_perturbation"
			elif self.benchmark_name == "singlerep":
				topo_dict.model.name = "single_perturbation"
			topo_dict.model.adapter.name = expt_param
			topo_dict.model.adapter.alpha = 0.9
			topo_dict.train.version = ver
			topo_dict.train.max_epochs = 200

			sim = self.init_modeling_obj( topo_dict = topo_dict )
			sim.modeling_objective = objective
			sim.modeling_version = ver
			sim.forward()
		self.plot_modeling_results( plot_suffix = plot_suffix )

	################################################################################
	def experiment2( self ):
		"""
		For each adapter, assessing the effect of the scaling parameter alpha.
		"""
		# Name of the experiment.
		self.expt_name = "experiment2"

		# FiLM does not have an alpha parameter.
		adapters = ["linear_perturb", "sigmoid_gating", "tanh_gating", "lora"]
		for i, adapter in enumerate( adapters ):
			# Suffix for the plots file.
			plot_suffix = f"{adapter}"

			# Parameters to be tested.
			self.expt_params = [0.7, 0.5, 0.3, 0.1]
			total_expt = len( self.expt_params )
			start = i
			step = 0.1

			self.modeling_versions = [
			round(start+step*j, 1 ) for j in range( 1, total_expt + 1 )
			]
			print( self.modeling_versions )

			if len( self.expt_params ) != len( self.modeling_versions ):
				raise ValueError( "No. of experimental params and modeling versions do not match. " +
					f"{self.expt_params} \t {self.modeling_versions}" )

			self.expt_dir = os.path.join( self.output_dir, self.expt_name )
			os.makedirs( self.expt_dir, exist_ok = True )

			for expt_param, ver in zip( self.expt_params, self.modeling_versions ):
				print( f"\n\033[1mModeling version = {ver} " +
						f"Experiment: {self.expt_name} {adapter}.alpha {expt_param}\033[0m" )
				objective = f"Experiment: {self.expt_name} - testing {adapter}.alpha = {expt_param}"

				topo_dict = topology_dict()
				topo_dict.loss.fape.add_penalty = False
				topo_dict.loss.supervised_chi.add_penalty = False
				topo_dict.loss.chain_center_of_mass.add_penalty = False
				topo_dict.loss.violation.add_penalty = True
				topo_dict.loss.violation.weight = 0.03
				topo_dict.loss.xlr.add_penalty = True
				topo_dict.loss.xlr.weight = 0.05

				if self.benchmark_name == "pairrep":
					topo_dict.model.name = "pair_perturbation"
				elif self.benchmark_name == "singlerep":
					topo_dict.model.name = "single_perturbation"
				topo_dict.model.adapter.name = adapter
				topo_dict.model.adapter.alpha = expt_param
				topo_dict.train.version = ver
				topo_dict.train.max_epochs = 200

				sim = self.init_modeling_obj( topo_dict = topo_dict )
				sim.modeling_objective = objective
				sim.modeling_version = ver
				sim.forward()

			self.plot_modeling_results( plot_suffix = plot_suffix )

	################################################################################
	def experiment3( self ):
		"""
		Using pseudo huber loss for fine-tuning StructureModule.
		Try different violation loss weights.
		"""
		# Name of the experiment.
		self.expt_name = "experiment3"
		self.benchmark_name = "altloss"

		# Violation loss weights.
		self.expt_params = [0.03, 0.1, 0.5]
		total_expt = len( self.expt_params )
		start = 0
		step = 1
		self.modeling_versions = [
		start + i for i in range( total_expt )
		]
		print( self.modeling_versions )

		if len( self.expt_params ) != len( self.modeling_versions ):
			raise ValueError( "No. of experimental params and modeling versions do not match. " +
				f"{self.expt_params} \t {self.modeling_versions}" )

		for expt_param, ver in zip( self.expt_params, self.modeling_versions ):
			# Suffix for the plots file.
			plot_suffix = f"{expt_param}"

			self.expt_dir = os.path.join( self.output_dir, self.expt_name )
			os.makedirs( self.expt_dir, exist_ok = True )

			print( f"\n\033[1mModeling version = {ver} " +
					f"Experiment: {self.expt_name} violation weight {expt_param}\033[0m" )
			objective = f"Experiment: {self.expt_name} - testing pseudo huber loss"

			topo_dict = topology_dict()
			topo_dict.loss.fape.add_penalty = False
			topo_dict.loss.supervised_chi.add_penalty = False
			topo_dict.loss.chain_center_of_mass.add_penalty = False
			topo_dict.loss.violation.add_penalty = True
			topo_dict.loss.violation.weight = expt_param
			topo_dict.loss.xlr.add_penalty = True
			topo_dict.loss.xlr.type = "pseudo_huber"
			topo_dict.loss.xlr.weight = 0.05

			topo_dict.model.name = "structure_module_finetuning"
			topo_dict.train.version = ver
			topo_dict.train.max_epochs = 100

			sim = self.init_modeling_obj( topo_dict = topo_dict )
			sim.modeling_objective = objective
			sim.modeling_version = ver
			sim.forward()

		self.plot_modeling_results( plot_suffix = plot_suffix )

	################################################################################
	def experiment4( self ):
		"""
		Using a rigid chain loss to preserve intrachain distances.
		Try different weights for the rigid loss.
		Also using differenet violation loss weights.
		"""
		# Name of the experiment.
		self.expt_name = "experiment4"
		self.benchmark_name = "rigidchain"

		for i, viol_weight in enumerate( [0.03, 0.1, 0.3] ):
			# RigidChain loss weights.
			self.expt_params = [0.0, 0.03, 0.05, 0.1, 0.3, 1.0]
			total_expt = len( self.expt_params )
			total_expt = len( self.expt_params )
			start = i
			step = 0.1

			self.modeling_versions = [
			round(start+step*j, 1 ) for j in range( 0, total_expt )
			]
			self.modeling_versions[0] = i
			print( self.modeling_versions )

			if len( self.expt_params ) != len( self.modeling_versions ):
				raise ValueError( "No. of experimental params and modeling versions do not match. " +
					f"{self.expt_params} \t {self.modeling_versions}" )

			for expt_param, ver in zip( self.expt_params, self.modeling_versions ):
				# Suffix for the plots file.
				plot_suffix = f"viol_{viol_weight}"

				self.expt_dir = os.path.join( self.output_dir, self.expt_name )
				os.makedirs( self.expt_dir, exist_ok = True )

				print( f"\n\033[1mModeling version = {ver} " +
						f"Experiment: {self.expt_name} RigidChain loss weight {expt_param}\033[0m" )
				objective = f"Experiment: {self.expt_name} - testing RigidChain loss"

				topo_dict = topology_dict()
				topo_dict.loss.fape.add_penalty = False
				topo_dict.loss.supervised_chi.add_penalty = False
				topo_dict.loss.chain_center_of_mass.add_penalty = False
				topo_dict.loss.violation.add_penalty = True
				topo_dict.loss.violation.weight = viol_weight
				topo_dict.loss.rigid_chain.add_penalty = True
				topo_dict.loss.rigid_chain.weight = expt_param
				topo_dict.loss.xlr.add_penalty = True
				topo_dict.loss.xlr.type = "ub_harmonic"
				topo_dict.loss.xlr.weight = 0.05

				topo_dict.model.name = "structure_module_finetuning"
				topo_dict.train.version = ver
				topo_dict.train.max_epochs = 100

				sim = self.init_modeling_obj( topo_dict = topo_dict )
				sim.modeling_objective = objective
				sim.modeling_version = ver
				sim.forward()

			self.plot_modeling_results( plot_suffix = plot_suffix )

	################################################################################
	################################################################################
	def plot_modeling_results( self, plot_suffix: str ):
		"""
		Create the following plots for the given system:
			Per-epoch loss and xl restraint.
			RMSD wrt 1st model.
			RMSF wrt the average model.
		"""
		print( "\n" + "-"*70 + "\n" + "-"*70 + "\n\033[1mRigid prior plots\033[0m\n" )
		for sys_name in self.benchmark["PDB ID"]:
			sys_plot_dir = os.path.join( self.expt_dir, f"{sys_name}_plots/" )
			os.makedirs( sys_plot_dir, exist_ok = True )
			print( f"Creating per-epoch loss plots for {sys_name}..." )
			self.per_epoch_loss_plots(
				sys_name = sys_name,
				sys_plot_dir = sys_plot_dir,
				plot_suffix = plot_suffix
				)

			print( f"Creating RMSD plots for {sys_name}..." )
			self.plot_rmsd_across_runs(
				sys_name = sys_name,
				sys_plot_dir = sys_plot_dir,
				plot_suffix = plot_suffix
				)

			print( f"Creating RMSF plots for {sys_name}..." )
			self.plot_rmsf(
				sys_name = sys_name,
				sys_plot_dir = sys_plot_dir,
				plot_suffix = plot_suffix
				)


	def get_input_for_per_epoch_plots( self,
		sys_name: str,
		modeling_version: int
		) -> Tuple[List[float], List[float], List[int]]:
		"""
		Obtain violation loss and xl restraint per epoch for the given modeling version.
		"""
		stat_file_path = get_stat_file_path(
			base_dir = self.base_dir,
			modeling_dir_name = self.modeling_dir_name,
			sys_name = sys_name,
			modeling_version = modeling_version )
		stats_dict = np.load( stat_file_path, allow_pickle = True ).item()

		violation = stats_dict["loss"]["violation"]
		xlr = stats_dict["loss"]["xlr"]
		# model_id is the same as epoch.
		epochs = stats_dict["model_id"]
		return violation, xlr, epochs


	def per_epoch_loss_plots( self, sys_name: str, sys_plot_dir: str, plot_suffix: str ):
		"""
		Plot the per-epoch xl restraint and violation loss across all runs.
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )

		print( self.expt_params )
		for i, expt in enumerate( self.expt_params ):
			modeling_version = self.modeling_versions[i]
			violation, xlr, epochs = self.get_input_for_per_epoch_plots(
				sys_name = sys_name,
				modeling_version = modeling_version )

			ax[0].plot( epochs, violation, label = expt )
			ax[0].set_xlabel( "No. of epochs", fontsize = 20 )
			ax[0].set_ylabel( "Violation loss", fontsize = 20 )
			ax[0].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[0].legend( fontsize = 14 )

			ax[1].plot( epochs, xlr, label = expt )
			ax[1].set_xlabel( "No. of epochs", fontsize = 20 )
			ax[1].set_ylabel( "XL restraint", fontsize = 20 )
			ax[1].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[1].legend( fontsize = 14 )

		ax[0].set_xlim( -0.1 )
		ax[0].set_ylim( -0.1 )
		ax[1].set_xlim( -0.1 )
		ax[1].set_ylim( -0.1 )
		loss_plot_file = os.path.join(
			sys_plot_dir, f"loss_per_epoch_{plot_suffix}_{expt}.png" )
		plt.savefig( loss_plot_file, dpi = 300, bbox_inches = "tight" , pad_inches = 0.1 )
		plt.close()

	################################################################################
	def plot_rmsd_across_runs( self,  sys_name: str, sys_plot_dir: str, plot_suffix: str ):
		"""
		Across all runs, plot the RMSD for all models wrt the 1st model.
		"""
		plt.rcParams["font.family"] = "sans"
		# cmap = plt.get_cmap( "RdYlGn_r" )
		# norm = plt.Normalize( min( self.expt_params ), max( self.expt_params ) )
		_, ax = plt.subplots( 1, 1, figsize = ( 30, 20 ) )

		for i, expt in enumerate( self.expt_params ):
			modeling_version = self.modeling_versions[i]
			ensemble_file = get_unrelaxed_ensemble_file(
				base_dir = self.base_dir,
				modeling_dir_name = self.modeling_dir_name,
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

			# color = cmap( norm( expt ) )
			ax.plot( epochs, rmsd, label = expt )
			ax.set_xlabel( "No. of epochs", fontsize = 25 )
			ax.set_ylabel( "RMSD wrt first model", fontsize = 25 )
			ax.tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax.legend( fontsize = 14 )

		ax.set_xlim( -0.1 )
		ax.set_ylim( -0.1 )
		rmsd_plot_file = os.path.join(
			sys_plot_dir, f"rmsd_plots_{plot_suffix}_{expt}.png" )
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


	def plot_rmsf( self, sys_name: str, sys_plot_dir: str, plot_suffix: str ):
		"""
		Across all runs,
			Plot RMSF wrt average model (this is essentially per-residue model precision)
			RMSF vs pLDDT of initial predicted structure
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( len( self.expt_params ), 2, figsize = ( 20, 40 ) )
		for i, expt in enumerate( self.expt_params ):
			modeling_version = self.modeling_versions[i]

			ensemble_file = get_unrelaxed_ensemble_file(
				base_dir = self.base_dir,
				modeling_dir_name = self.modeling_dir_name,
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
			ax[i, 0].set_title( f"Expt param = {expt}; Spearman = {corrs}", fontsize = 20 )
			ax[i, 0].set_xlabel( "Residue numbers", fontsize = 20 )
			ax[i, 0].set_ylabel( "RMSF wrt average model", fontsize = 20 )
			ax[i, 0].set_xlim( -0.5 )
			ax[i, 0].set_ylim( -0.5 )
			ax[i, 0].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[i, 0].legend( fontsize = 14 )

			# RMSF wrt avg model per run.
			ax[i, 1].scatter( plddt, rmsf_avg )
			ax[i, 1].set_title( f"Expt param = {expt}; Spearman = {corrs}", fontsize = 20 )
			ax[i, 1].set_xlabel( "pLDDT", fontsize = 20 )
			ax[i, 1].set_ylabel( "RMSF wrt average model", fontsize = 20 )
			ax[i, 1].set_xlim( -0.5 )
			ax[i, 1].set_ylim( -0.5 )
			ax[i, 1].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )

		plt.subplots_adjust( hspace = 0.5, wspace = 0.4 )

		rmsf_plot_file = os.path.join(
			sys_plot_dir, f"rmsf_plots_{plot_suffix}_{expt}.png" )
		plt.savefig( rmsf_plot_file, dpi = 300, bbox_inches = "tight" , pad_inches = 0.1 )
		plt.close()

if __name__ == "__main__":
	PerturbRepresentation().forward()
