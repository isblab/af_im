"""
Assessing the effect of the no. of Evoformer blocks (num_blocks) on the prediction.
"""
from typing import List, Tuple, Dict, Any
import os, warnings, pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

import torch

from openfold.config import model_config

from openfold_wrapper import IntegrativeLearning
from system_representation2 import SystemRepresentation
from topology import topology_dict

from utils.utils import open_file_handler
from utils.paths import get_sys_data_dir_path
from utils.pdb_utils import SaveModels
from utils.tools import (
	load_ensemble,
	compute_rmsd )

class AblateEvoformer():
	"""
	Assessing the effect of the no. of Evoformer blocks (num_blocks) on the prediction.
	Ablate num_blocks from 1-48.
	Compare RMSD wrt the ground truth structure.
	Plot RMSF vs pLDDT.
	Compare pLDDT, PAE across all ablations.
		Plot pLDDT across each run as a heatmap.
		Compute mean and variance from all PAE matrices and plot.
	"""
	def __init__( self ):
		#self.num_blocks = [1, 5, 10, 15, 20, 25, 30, 35, 40, 48]
		self.num_blocks = np.arange( 1, 48+1, 1 )
		self.device = "cuda:0"


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.run_ablation()
		self.create_plots()


	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		self.base_dir = os.path.join( os.path.abspath( "./benchmark" ) )
		self.benchmark_name = "pairrep"
		self.meta_dir = os.path.join( self.base_dir,
									f"{self.benchmark_name}_metadata" )
		# Name for the dir to store modeling output for all systems.
		self.modeling_dir_name = "evo_ablation"
		# Output dir for storing each benchmark run results.
		self.benchmark_output_dir = os.path.join(
												self.base_dir,
												f"{self.benchmark_name}_benchmark_results" )
		self.benchmark_csv_file = os.path.join( self.meta_dir,
												f"selected_{self.benchmark_name}_benchmark.csv" )

		self.output_dir = os.path.join(
			self.base_dir, self.modeling_dir_name
		)
		os.makedirs( self.output_dir, exist_ok = True )

	def get_sys_dir( self, sys_name: str):
		sys_dir = os.path.join( self.output_dir, f"{sys_name}" )
		return sys_dir

	def get_ensemble_file( self, sys_name: str ):
		sys_dir = self.get_sys_dir( sys_name = sys_name )
		ensemble_file = os.path.join(
			sys_dir, f"{sys_name}_ensemble.pdb" )
		return ensemble_file

	################################################################################
	################################################################################
	def run_ablation( self ):
		"""
		For all PDB IDs:
			Initialize the SystemRepresentation object.
			Predict structures, ablating no. of Evoformer blocks..
		"""
		for sys_name in ["4rhz", "8wtd"]:
			print( f"\n\033[1m{sys_name}\033[0m\n" + "-"*80 )
			ensemble_file = self.get_ensemble_file( sys_name = sys_name )

			if os.path.exists( ensemble_file ):
				print( f"Ensemble file already exists for {sys_name}..." )
			else:
				sys_dir = self.get_sys_dir( sys_name = sys_name )
				os.makedirs( sys_dir, exist_ok = True )
				sys_rep_obj = self.init_system_representation( sys_name = sys_name )
				self.predict(
					sys_name = sys_name,
					sys_rep_obj = sys_rep_obj,
					sys_dir = sys_dir )
				torch.cuda.empty_cache()

	################################################################################
	################################################################################
	def init_system_representation( self, sys_name: str
			) -> SystemRepresentation:
		"""
		"""
		sys_conf_suff = ""
		# Initialize the topology dict.
		topo_dict = topology_dict()

		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		il_obj = IntegrativeLearning( 
				sys_name = sys_name,
				base_dir = self.base_dir,
				data_dir = data_dir,
				sys_config_file =  f"sys_config_{sys_name}{sys_conf_suff}.json",
				modeling_dir_name = self.modeling_dir_name,
				topology_dict = topo_dict,
				)
		il_obj.create_required_paths_dirs()

		sys_rep_obj = SystemRepresentation(
			sys_name = sys_name,
			is_multimer = True,
			sys_rep_config = topo_dict.system_representation,
			fasta_dir = il_obj.fasta_dir,
			alignment_dir = il_obj.alignment_dir,
			ofold_output_dir = il_obj.ofold_output_dir,
			seed_worker = il_obj.seed_worker,
			device = il_obj.device )
		return sys_rep_obj


	################################################################################
	def predict( self,
			 sys_name: str,
			 sys_rep_obj: SystemRepresentation,
			  sys_dir: str ):
		"""
		Ablate the no. of Evoformer blocks.
		"""
		sys_rep_obj.init_model_config()
		sys_rep_obj.init_feature_processor()
		feature_dict, processed_feature_dict, tag = sys_rep_obj.prepare_input()

		save_model_obj = SaveModels(
			title = sys_name,
			output_format = "pdb",
			ensemble_dir = "",
			save_single_model = False )
		save_model_obj.initialize_system()

		for num_blocks in self.num_blocks:
			print( f"\033[1mEvoformer block = {num_blocks}\033[0m" )
			out_file = os.path.join(
				sys_dir, f"out_{num_blocks}.pkl" )
			
			if os.path.exists( out_file ):
				f = open_file_handler( out_file, "rb" )
				out = pkl.load( f )
				f.close()
			else:
				sys_rep_obj.model_config.model.evoformer_stack.no_blocks = int( num_blocks )

				out, _ = sys_rep_obj.get_init_pred(
					processed_feature_dict = processed_feature_dict,
					tag = tag
				)

			unrelaxed_protein = sys_rep_obj.get_prot_from_pred(
				out = out,
				feature_dict = feature_dict,
				processed_feature_dict = processed_feature_dict )

			save_model_obj.add_model(
				prot = unrelaxed_protein, model_id = num_blocks )

			# Save the out dict on disk.
			w = open_file_handler( out_file, "wb" )
			pkl.dump( out, w, protocol = pkl.HIGHEST_PROTOCOL )
			w.close()
			torch.cuda.empty_cache()

		# Save ensemble on disk.
		ensemble_file = self.get_ensemble_file( sys_name = sys_name )
		save_model_obj.save(
			save_model_obj.system, os.path.splitext( ensemble_file )[0] )

	################################################################################
	################################################################################
	def create_plots( self ) -> Dict[str, Any]:
		"""
		Create the following plots:
			1. num_blocks vs RMSD wrt PDB structure.
			2. num_blocks vs ipTM.
			3. pLDDT across num_blocks.
			4. PAE mean and variance across num_blocks.
		"""
		print( "\n\033[1mCreating plots\033[0m" )
		for sys_name in ["4rhz", "8wtd"]:
			#self.plot_rmsd( sys_name = sys_name )
			print( f"\033[1m-> \033[0m{sys_name}" )
			self.plot_confidence_metrics( sys_name = sys_name )


	def load_out_dict( self, sys_name: str, num_blocks: int ):
		"""
		Load the out dict for the corresponding num_block pred.
		"""
		sys_dir = self.get_sys_dir( sys_name = sys_name )
		out_file = os.path.join(
			sys_dir, f"out_{num_blocks}.pkl" )
		f = open_file_handler( out_file, "rb" )
		out = pkl.load( f )
		return out

	################################################################################
	def compute_emsd_with_gt_struct( self, sys_name: str ) -> np.ndarray:
		"""
		Compute RMSD for all predicted structures wrt the ground truth (PDB) structure.
		"""
		warnings.filterwarnings( "ignore" )
		ensemble_file = self.get_ensemble_file( sys_name = sys_name )

		gt_struct_file = os.path.join(
			self.meta_dir, f"struct/{sys_name}.pdb" )

		u_gt = load_ensemble( ensemble_file = gt_struct_file )
		u_pred = load_ensemble( ensemble_file = ensemble_file )

		rmsd = compute_rmsd( to_align = u_pred, ref = u_gt )[:,-1]

		return rmsd


	def plot_rmsd( self, sys_name: str ):
		"""
		Create a plot for the RMSD (wrt PDB structure) vs the no. of blocks.
		"""
		rmsd = self.compute_emsd_with_gt_struct( sys_name = sys_name )

		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 0, 0, figsize = ( 10, 10 ) )

		ax.plot( self.num_blocks, rmsd )
		ax.set_title( "#Evoformer blocks vs RMSD", fontsize = 16 )
		ax.set_xlabel( "No. of Evoformer", fontsize = 16 )
		ax.set_ylabel( "RMSD", fontsize = 16 )

		plot_file = os.path.join( self.output_dir, f"{sys_name}_rmsd.png" )
		plt.savefig( plot_file, dpi = 300 )
		plt.close()

	################################################################################
	def get_confidence_metrics_per_run( self,
			sys_name: str, num_blocks: int
			) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Obtain the pLDDT, PAE, and ipTM for a given prediction.
		"""
		out = self.load_out_dict(
			sys_name = sys_name,
			num_blocks = num_blocks )
		#print( out.keys() )
		plddt = out["plddt"]
		pae = out["predicted_aligned_error"]
		iptm = out["iptm_score"]

		return plddt, pae, iptm


	def get_moments_from_pae( self, pae: np.array
			) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Given a list of pae matrices, compute the mean
			and standard deviation across all.
		"""
		# [N, N] -> [B, N, N]; B: no. of preds; N: no. of residues.
		pae = np.stack( pae )
		pae_mean = np.mean( pae, axis = 0 )
		pae_std = np.std( pae, axis = 0 )
		return pae_mean, pae_std


	def plot_confidence_metrics( self, sys_name: str ):
		"""
		Obtain confidence metrics for all num_blocks.
		Create the required plos for each.
		"""
		metrics = {k: [] for k in ["plddt", "pae", "iptm"]}
		for num_blocks in self.num_blocks:
			plddt, pae, iptm = self.get_confidence_metrics_per_run(
				sys_name = sys_name, num_blocks = num_blocks )
			metrics["plddt"].append( plddt )
			metrics["pae"].append( pae )
			metrics["iptm"].append( iptm )

		plddt = np.vstack( metrics["plddt"] )
		pae_mean, pae_std = self.get_moments_from_pae( pae = metrics["pae"] )

		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 2, 2, figsize = ( 30, 25 ) )

		m = plddt.shape[1]
		mean_plddt = np.mean( plddt, axis = 0 )
		std_plddt = np.std( plddt, axis = 0 )

		ax[0, 0].plot(
			np.arange( 1, m+1, 1 ), mean_plddt, c = "blue" )
		ax[0, 0].fill_between(
			np.arange( 1, m+1, 1 ),
			mean_plddt - std_plddt,
			mean_plddt + std_plddt,
			color = "gray"
		)
		ax[0, 0].set_ylim( 0, 100 )
		ax[0, 0].set_title( "#Evoformer blocks vs pLDDT", fontsize = 25 )
		ax[0, 0].set_xlabel( "Residues", fontsize = 20 )
		ax[0, 0].set_ylabel( "pLDDT", fontsize = 20 )
		ax[0, 0].tick_params( axis = "both" , labelsize = 16, length = 10, width = 3 )

		ax[0, 1].plot( self.num_blocks, metrics["iptm"] )
		ax[0, 1].set_title( "#Evoformer blocks vs ipTM", fontsize = 25 )
		ax[0, 1].set_xlabel( "No. of Evoformer blocks", fontsize = 20 )
		ax[0, 1].set_ylabel( "ipTM", fontsize = 20 )
		ax[0, 1].tick_params( axis = "both" , labelsize = 16, length = 10, width = 3 )

		im1 = ax[1, 0].imshow( pae_mean, cmap = "Greens_r" )
		ax[1, 0].set_title( "Mean PAE across #Evoformer blocks", fontsize = 25 )
		ax[1, 0].tick_params( axis = "both" , labelsize = 16, length = 10, width = 3 )
		cbar1 = plt.colorbar( im1, ax = ax[1, 0] )
		cbar1.ax.tick_params( labelsize = 16 )

		im2 = ax[1, 1].imshow( pae_std, cmap = "Greens_r" )
		ax[1, 1].set_title( "Standard deviation PAE across #Evoformer blocks", fontsize = 25 )
		ax[1, 1].tick_params( axis = "both" , labelsize = 16, length = 10, width = 3 )
		cbar2 = plt.colorbar( im2, ax = ax[1, 1] )
		cbar2.ax.tick_params( labelsize = 16 )

		plt.tight_layout()
		plot_file = os.path.join( self.output_dir, f"{sys_name}_metrics.png" )
		plt.savefig( plot_file, dpi = 300 )
		plt.close()


if __name__ == "__main__":
	AblateEvoformer().forward()

