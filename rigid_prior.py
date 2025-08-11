"""
Test whether AF2 learns a rigid prior or not.
Wrapper over the BenchmarkModeling module.
"""
from typing import List, Tuple, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmark_runs import BenchmarkModeling
from utils.paths import (
	get_stat_file_path,
	get_benchmark_results_dir_path,
	get_unrelaxed_model_file,
	get_unrelaxed_ensemble_file )
from utils.tools import (
	get_residue_ids,
	compute_rmsf_wrt_avg_model,
	compute_rmsd_post_align )


class RigigPrior():
	"""
	Run the simulations to test the rigid prior theory.
	Use the BenchmarkModeling module for running simulations.
	"""
	def __init__( self ):
		self.benchmark_name = "rigid"
		self.xl_restraint_weights = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
		self.struct_format = "pdb"
		self.remove_sys_modeling_dir = False


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.load_benchmark()

		num_xl_weights = len( self.xl_restraint_weights )
		modeling_versions = {
			"no_viol": list( np.arange( 1, num_xl_weights + 1, 1 ) ),
			"viol": list( np.arange( num_xl_weights+1, 2*num_xl_weights + 1, 1 ) )
		}

		self.run_modeling( modeling_versions = modeling_versions )
		self.plot_modeling_results( modeling_versions = modeling_versions )


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


	def load_benchmark( self ):
		"""
		Load the benchmak .csv file.
		"""
		self.benchmark = pd.read_csv( self.benchmark_csv_file )

	################################################################################
	################################################################################
	def run_modeling( self, modeling_versions: Dict[str, List] ):
		"""
		Run modeling for all required parameters.
		"""
		modeling_version = 1
		for viol_label in modeling_versions:
			if viol_label == "no_viol":
				enable_violation_loss = False
			elif viol_label == "viol":
				enable_violation_loss = True

			for i, xl_weight in enumerate( self.xl_restraint_weights ):
				modeling_version = int( modeling_versions[viol_label][i] )
				print( f"\n\033[1mModeling version = {modeling_version}" +
						f"\t xl_weight = {xl_weight}\t" +
						f"violation loss enabled = {enable_violation_loss} \033[0m\n" )
				self.run_modeling_benchmark(
					modeling_version = modeling_version,
					xl_weight = xl_weight,
					enable_violation_loss = enable_violation_loss
					)


	def run_modeling_benchmark( self,
		modeling_version: int,
		xl_weight: float,
		enable_violation_loss: bool ):
		"""
		Instantiate the BenchmarkModeling module and run the simulations.
		"""
		objective = "Rigid prior experiments. No FAPE, supervised_chi and CCOM loss. " + \
					f"Using xl_weight = {xl_weight} and violation loss = {enable_violation_loss}"
		sim = BenchmarkModeling()
		sim.benchmark_name = self.benchmark_name
		sim.modeling_version = modeling_version
		sim.objective = objective
		sim.sys_conf_suff = "_tpfp"
		sim.device = "cuda:0"
		sim.enable_relax_validate = False
		sim.skip_rerun = True
		sim.remove_sys_modeling_dir = self.remove_sys_modeling_dir
		sim.violation["add_penalty"] = enable_violation_loss
		sim.xlr["weight"] = xl_weight

		sim.forward()

	################################################################################
	################################################################################
	def plot_modeling_results( self,  modeling_versions: List ):
		"""
		Create the following plots:
			Per-epoch loss and xl restraint.
			RMSD wrt 1st model.
			RMSF wrt the average model.
		"""
		print( "-"*70 + "\n" + "-"*70 + "\n\033[1mRigid prior plots\033[0m\n" )
		for viol_label in modeling_versions:
			for sys_name in self.benchmark["PDB ID"]:
				print( f"Creating per-epoch loss plots for {sys_name}..." )
				self.per_epoch_loss_plots(
					sys_name = sys_name,
					viol_label = viol_label,
					modeling_versions = modeling_versions
					)

				print( f"Creating RMS plots for {sys_name}..." )
				self.plot_rms(
					sys_name = sys_name,
					viol_label = viol_label,
					modeling_versions = modeling_versions
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


	def per_epoch_loss_plots( self, sys_name: str, viol_label: str, modeling_versions: np.array ):
		"""
		Plot the per-epoch xl restraint and violation loss across all runs.
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 2, 1, figsize = ( 20, 20 ) )

		for i, xl_weight in enumerate( self.xl_restraint_weights ):
			modeling_version = modeling_versions[viol_label][i]
			violation, xlr, epochs = self.get_input_for_per_epoch_plots(
				sys_name = sys_name,
				modeling_version = modeling_version )

			ax[0].plot( epochs, violation, label = xl_weight )
			ax[0].set_xlabel( "No. of epochs", fontsize = 25 )
			ax[0].set_ylabel( "Violation loss", fontsize = 25 )
			ax[0].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[0].legend()
			ax[1].plot( epochs, xlr, label = xl_weight )
			ax[1].set_xlabel( "No. of epochs", fontsize = 20 )
			ax[1].set_ylabel( "XL restraint", fontsize = 20 )
			ax[1].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[1].legend()

		benchmark_results_dir = get_benchmark_results_dir_path(
			base_dir = "./benchmark",
			benchmark_name = self.benchmark_name,
			modeling_version = modeling_version )
		loss_plot_file = os.path.join(
			benchmark_results_dir, f"loss_per_epoch_{sys_name}_{viol_label}.png" )
		plt.savefig( loss_plot_file, dpi = 300 )
		plt.close()

	################################################################################
	def get_input_for_rms_plots( self, ensemble_file: str ):
		"""
		Given an ensemble file, compute,
			RMSD for all models wrt the 1st model
			RMSF wrt average model
		"""
		rmsf = compute_rmsf_wrt_avg_model( ensemble_file = ensemble_file )
		rmsd = compute_rmsd_post_align( ensemble_file = ensemble_file )
		rmsd = rmsd[:,2]
		resids = get_residue_ids( ensemble_file )

		return rmsf, rmsd, resids


	def plot_rms( self, sys_name: str, viol_label: str, modeling_versions: np.array ):
		"""
		Across all runs,
			Plot the RMSD for all models wrt the 1st model
			RMSF wrt average model (this is essentially per-residue model precision)
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 2, 1, figsize = ( 30, 20 ) )
		for i, xl_weight in enumerate( self.xl_restraint_weights ):
			modeling_version = modeling_versions[viol_label][i]

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

			rmsf, rmsd, resids = self.get_input_for_rms_plots(
				ensemble_file = ensemble_file )
			# resids start from 1 for all chains. So, using system indices.
			sys_idx = np.arange( 0, resids.shape[0], 1 )

			ax[0].plot( epochs, rmsd, label = xl_weight )
			ax[0].set_xlabel( "No. of epochs", fontsize = 25 )
			ax[0].set_ylabel( "RMSD wrt first model", fontsize = 25 )
			ax[0].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[0].legend()
			ax[1].plot( sys_idx, rmsf, label = xl_weight )
			ax[1].set_xlabel( "Residue numbers", fontsize = 20 )
			ax[1].set_ylabel( "RMSF wrt average model", fontsize = 20 )
			ax[1].set_xticks( sys_idx, resids )
			ax[1].tick_params( axis = "both" , labelsize = 20, length = 10, width = 4 )
			ax[1].legend()

		benchmark_results_dir = get_benchmark_results_dir_path(
			base_dir = "./benchmark",
			benchmark_name = self.benchmark_name,
			modeling_version = modeling_version )
		loss_plot_file = os.path.join(
			benchmark_results_dir, f"rms_plots_{sys_name}_{viol_label}.png" )
		plt.savefig( loss_plot_file, dpi = 300 )
		plt.close()


if __name__ == "__main__":
	RigigPrior().forward()
