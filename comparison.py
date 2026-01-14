"""
Compare the performance of our method with GRASP and AlphaLink2
	for the following metrics:
	1. TM score wrt native structure
	2. DockQ
	3. Data satisfaction
	4. Fraction of models satisfying an XL.
"""
from typing import List, Tuple, Dict, Any, Iterator
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ml_collections as mlc

import torch

from openfold_wrapper import IntegrativeLearning
from topology import topology_dict
from metrics import Metrics
from utils.utils import (
	read_json, write_json,
	run_subprocess
	)
from utils.paths import (
	BASE_DIR,
	get_sys_data_dir_path,
	get_benchmark_csv_file,
	get_native_struct_file,
	get_init_struct_file,
	get_relaxed_model_file,
	get_analysis_dict_path
)
from utils.pdb_utils import (
	Parser,
	get_chain_id,
	remap_chains_cif,
	remap_chains_pdb )
from utils.tools import usalign, get_alignment_score, dockq


def init_integrative_learning_module(
		sys_name: str,
		base_dir: str,
		benchmark_name: str,
		modeling_dir_name: str,
		sys_conf_suff: str
		) -> Tuple[IntegrativeLearning, mlc.ConfigDict]:
	"""
	Initialize the IntegrativeLearning module.

	Input:
	----------
	sys_name -> name of the complex modeled. For the benchmark, it's the PDB ID.

	Returns:
	----------
	il_obj -> an instance of IntegrativeLearning().
	topo_dict -> dict specifying the configs for modeling.
	"""
	# Initialize the topology dict.
	topo_dict = topology_dict()
	# Not using XL tolerance while computing metric.
	topo_dict.metrics.xlr.allow_xl_tolerance = False

	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )

	il_obj = IntegrativeLearning( 
			sys_name = sys_name,
			base_dir = base_dir,
			data_dir = data_dir,
			sys_config_file =  f"sys_config_{sys_name}{sys_conf_suff}.json",
			modeling_dir_name = modeling_dir_name,
			topology_dict = topo_dict,
			)
	il_obj.create_required_paths_dirs()
	return il_obj, topo_dict


class Comparison():
	"""
	Compare the performance of our method with GRASP and AlphaLink2.
	"""
	def __init__( self ):
		# IMP_DL version to be considered.
		self.modeling_version = 16
		self.model_type = {
			"imp_dl": True,
			"grasp": True,
			"alphalink2": False,
			"boltz2": True
		}
		# USalign script.
		self.usalign_script = "USalign"
		self.logs = {}


	def forward( self ):
		"""
		"""
		self.create_required_files()
		self.init_logs()
		self.create_required_dirs()

		# Perform structural filtering
		self.filtering()

		# RMSD and TM-score computation.
		self.compute_struct_similarity_for_benchmark()

		# Rema chains before DockQ computation.
		self.remap_chains_in_native()
		self.remap_chains_in_preds()

		self.compute_dockq_for_benchmark()

		self.compute_data_sat_for_benchmark()

		# Create plots.
		self.create_plots()


	def init_logs( self ):
		"""
		Initialize or load the logs dict.
		"""
		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )
			for k1 in ["data_satisfaction", "rmsd", "tm", "dockq"]:
				if self.imp_dl not in self.logs[k1]:
					self.logs[k1][self.imp_dl] = {}
		else:
			for k1 in ["data_satisfaction", "rmsd", "tm", "dockq"]:
				self.logs[k1] = {}
				for k2 in ["init", self.imp_dl, "grasp", "alphalink2", "boltz2"]:
					self.logs[k1][k2] = {}


	def create_required_files( self ):
		"""
		Create the required file and dir paths.
		"""
		self.base_dir = BASE_DIR
		# self.base_dir = "/data2/kartik/IMP_Rewired/imp_dl/benchmark/"
		self.benchmark_name = "xlmerged"
		self.modeling_dir_name = f"{self.benchmark_name}_modeling"

		# Our method name.
		self.imp_dl = f"imp_dl_{self.modeling_version}"

		# Dir containing GRASP/AlphaLink2 preds for the benchmark.
		self.grasp_output_dir = os.path.join( self.base_dir, "Grasp" )
		self.alink2_output_dir = os.path.join( self.base_dir, "Alphalink2" )
		self.boltz2_output_dir = os.path.join( self.base_dir, "Boltz2" )

		# Dir to store the results of the analysis in this script.
		self.output_dir = os.path.join( self.base_dir, "comparison" )
		# Temp dir to store intermediate outputs.
		self.tm_tmp_dir = os.path.join( self.output_dir, "tm_tmp" )
		self.dockq_tmp_dir = os.path.join( self.output_dir, "dockq_tmp" )

		# Load the benchmark.
		benchmark_file = get_benchmark_csv_file(
			self.base_dir,
			self.benchmark_name )
		self.benchmark = pd.read_csv( benchmark_file )

		self.selected_models_file = os.path.join( self.output_dir, f"Selected_models.json" )

		self.logs_file = os.path.join( self.output_dir, f"Logs.json" )


	def create_required_dirs( self ):
		"""
		Create the required directories.
		Temporary dir for storing intermediate
			files for rmsd computation.
		"""
		for dir_ in [self.output_dir,
			self.tm_tmp_dir, self.dockq_tmp_dir]:
			os.makedirs( dir_, exist_ok = True )


	def remove_tmp_dir( self ):
		cmd = ["rm", "-r", f"{self.tmp_dir}"]
		run_subprocess( cmd )


	def get_grasp_model_file( self,
		sys_name: str,
		remapped: str = None ) -> Iterator[Tuple[int, str]]:
		"""
		A generator that yields that model_file of all GRASP
			predicted structures for a given systen.
		GRASP predicts 25 models by default with 0-indexed model IDs.
		If remapped is True, returns the path for the remaped prediction file.
		"""
		for model_id in range( 0, 25 ):
			if remapped:
				model_file = os.path.join(
					self.dockq_tmp_dir,
					f"{sys_name}_grasp_{model_id}.pdb" )
			else:
				model_file = os.path.join(
					self.grasp_output_dir,
					f"{sys_name}/ranked_{model_id}.pdb" )
			yield model_id, model_file


	def get_alphalink2_model_file( self,
		sys_name: str,
		remapped: str = None ) -> Iterator[Tuple[int, str]]:
		"""
		A generator that yields that model_file of all AlphaLink2
			predicted structures for a given systen.
		AlphaLink2 predicts 25 models by default with 0-indexed model IDs.
		"""
		sys_dir = os.path.join( self.alink2_output_dir, sys_name )
		for model_id, model_file in enumerate(
			glob.glob( f"{sys_dir}/**.pdb" )
		):
			# Ignore the best model .pdb file.
			if "_best.pdb" in model_file:
				continue
			if remapped:
				model_file = os.path.join(
					self.dockq_tmp_dir,
					f"{sys_name}_alphalink2_{model_id}.pdb" )
				yield model_id, model_file
			else:
				yield model_id, model_file


	def get_boltz2_model_file( self,
		sys_name: str,
		remapped: str = None ) -> Iterator[Tuple[int, str]]:
		"""
		A generator that yields model_file for all Boltz2
			predicted structures for a given systen.
		We used Boltz-2 to predict 50 structures.
		"""
		sys_dir = os.path.join( self.boltz2_output_dir, sys_name )
		pred_dir = os.path.join( sys_dir,
			f"boltz_results_{sys_name}_restraint/predictions/{sys_name}_restraint/" )
		for model_id, model_file in enumerate(
			glob.glob( f"{pred_dir}/*.cif" )
		):
			if remapped:
				model_file = os.path.join(
					self.dockq_tmp_dir,
					f"{sys_name}_boltz2_{model_id}.pdb" )
				yield model_id, model_file
			else:
				yield model_id, model_file

	################################################################################
	################################################################################
	def filtering( self ):
		"""
		Remove structurally similar models for GRASP, AlphaLink2, and Boltz2.
		"""
		print( "\n" + "-"*70 +
			"\n\t\033[1m--> Filtering structurally similar models <--\033[0m\n" +
			"-"*70 )

		if os.path.exists( self.selected_models_file ):
			self.selected_models = read_json( self.selected_models_file )
		else:
			self.selected_models = {k:{} for k in ["grasp", "alphalink2", "boltz2"]}

		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( f"{i}. {sys_name}" )
			# Filter GRASP predicted models.
			if self.model_type["grasp"]:
				if sys_name not in self.selected_models["grasp"]:
					model_ids, model_files = [], []
					for model_id, model_file in self.get_grasp_model_file(
							sys_name = sys_name, remapped = False ):
						model_ids.append( model_id )
						model_files.append( model_file )
					selected_models = self.similarity_filtering(
							model_ids = model_ids,
							model_files = model_files )
					self.selected_models["grasp"][sys_name] = selected_models
				print( "GRASP: ", self.selected_models["grasp"][sys_name] )
				write_json( self.selected_models, self.selected_models_file )

			# Filter AlphaLink2 predicted models.
			if self.model_type["alphalink2"]:
				if sys_name not in self.selected_models["alphalink2"]:
					model_ids, model_files = [], []
					for model_id, model_file in self.get_alphalink2_model_file(
							sys_name = sys_name, remapped = False ):
						model_ids.append( model_id )
						model_files.append( model_file )
					selected_models = self.similarity_filtering(
							model_ids = model_ids,
							model_files = model_files )
					self.selected_models["alphalink2"][sys_name] = selected_models
				print( "Alphalink2: ", self.selected_models["alphalink2"][sys_name] )
				write_json( self.selected_models, self.selected_models_file )

			# Filter AlphaLink2 predicted models.
			if self.model_type["boltz2"]:
				if sys_name not in self.selected_models["boltz2"]:
					model_ids, model_files = [], []
					for model_id, model_file in self.get_boltz2_model_file(
							sys_name = sys_name, remapped = False ):
						model_ids.append( model_id )
						model_files.append( model_file )
					selected_models = self.similarity_filtering(
							model_ids = model_ids,
							model_files = model_files )
					self.selected_models["boltz2"][sys_name] = selected_models
				print( "Boltz2: ", self.selected_models["boltz2"][sys_name] )
				write_json( self.selected_models, self.selected_models_file )


	def similarity_filtering( self,
		model_ids: List[int],
		model_files: List[str] ):
		"""
		Filter models based on structural similarity.
		Two models are similar if they have a TM-score >=0.7.
		"""
		selected_models = []
		ignore_models = []

		total_models = len( model_ids )
		for i in range( total_models ):
			model_id1 = model_ids[i]
			if model_id1 in ignore_models:
				continue
			model1_file = model_files[i]
			for j in range( i, total_models ):
				model_id2 = model_ids[j]
				if model_id1 == model_id2 or model_id2 in ignore_models:
					continue

				_, tm = self.run_usalign(
					native_file = model1_file,
					model_id2 = model_id2,
					model2_file = model_files[j]
				)

				if tm >= 0.7:
					ignore_models.append( model_id2 )

			if model_id1 not in selected_models:
				selected_models.append( model_id1 )

		return selected_models

	################################################################################
	################################################################################
	def compute_struct_similarity_for_benchmark( self ):
		"""
		Compute the structural similarity wrt the native structure across the
			entire benchmark for using the TM-score and RMSD.
			Initial OpenFold structure
			IMP DL
			GRASP
			AlphaLink2
			Boltz2
		"""
		print( "\n" + "-"*70 +
			"\n\t\033[1m--> Computing TM-score wrt the ground truth structure <--\033[0m\n" +
			"-"*70 )

		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( f"{i}. {sys_name}" )
			native_file = get_native_struct_file(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name,
			)

			if sys_name not in self.logs["tm"]["init"]:
				rmsd, tm = self.compute_tm_init_struct( sys_name = sys_name, native_file = native_file )
				self.logs["tm"]["init"][sys_name] = tm
				self.logs["rmsd"]["init"][sys_name] = rmsd

			if self.model_type["imp_dl"]:
				if sys_name not in self.logs["tm"][self.imp_dl]:
					rmsd, tm = self.compute_per_sys_tm( sys_name = sys_name, native_file = native_file )
					self.logs["tm"][self.imp_dl][sys_name] = tm
					self.logs["rmsd"][self.imp_dl][sys_name] = rmsd

			if self.model_type["grasp"]:
				if sys_name not in self.logs["tm"]["grasp"]:
					rmsd, tm = self.compute_per_sys_tm_grasp( sys_name = sys_name, native_file = native_file )
					self.logs["tm"]["grasp"][sys_name] = tm
					self.logs["rmsd"]["grasp"][sys_name] = rmsd

			if self.model_type["alphalink2"]:
				if sys_name not in self.logs["tm"]["alphalink2"]:
					rmsd, tm = self.compute_per_sys_tm_alphalink2( sys_name = sys_name, native_file = native_file )
					self.logs["tm"]["alphalink2"][sys_name] = tm
					self.logs["rmsd"]["alphalink2"][sys_name] = rmsd

			if self.model_type["boltz2"]:
				if sys_name not in self.logs["tm"]["boltz2"]:
					rmsd, tm = self.compute_per_sys_tm_boltz2( sys_name = sys_name, native_file = native_file )
					self.logs["tm"]["boltz2"][sys_name] = tm
					self.logs["rmsd"]["boltz2"][sys_name] = rmsd

			write_json( self.logs, self.logs_file )


	def compute_tm_init_struct( self, sys_name: str, native_file: str ):
		"""
		Compute TM-score for the initial OpenFold predicted structure.
		"""
		init_struct_file = get_init_struct_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
		)
		rmsd, tm = self.run_usalign(
			native_file = native_file,
			model_id2 = 00,  # Arbitrary model_id for the init struct.
			model2_file = init_struct_file
		)
		return rmsd, tm


	def compute_per_sys_tm( self,
		sys_name: str,
		native_file: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute TM-score wrt the native structure for the given
			system for all predictions from our method.
		"""
		# stats_file = get_stat_file_path(
		# 	base_dir = self.base_dir,
		# 	sys_name = sys_name,
		# 	modeling_dir_name = self.modeling_dir_name,
		# 	modeling_version = version
		# )
		# stats_dict = np.load( stats_file, allow_pickle = True ).item()
		# model_ids = stats_dict["model_id"]
		# del stats_dict
		analysis_dict_file = get_analysis_dict_path(
			base_dir = self.base_dir,
			sys_name = sys_name,
			modeling_dir_name = self.modeling_dir_name,
			modeling_version = self.modeling_version
		)
		analysis_dict = np.load( analysis_dict_file, allow_pickle = True ).item()
		model_ids = analysis_dict["selected_good_models"]
		del analysis_dict

		rmsd, tm = [], []
		for model_id in model_ids:
			model_file = get_relaxed_model_file(
				base_dir = self.base_dir,
				sys_name = sys_name,
				model_id = model_id,
				modeling_dir_name = self.modeling_dir_name,
				modeling_version = self.modeling_version
			)
			r, t = self.run_usalign(
				native_file = native_file,
				model_id2 = model_id,
				model2_file = model_file
			)
			rmsd.append( r )
			tm.append( t )
		return rmsd, tm


	def compute_per_sys_tm_grasp( self,
		sys_name: str,
		native_file: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute TM-score wrt the native structure for the given
			system for all predictions from GRASP.
		GRASP predicts 25 models by default with 0-indexed model IDs.
		"""
		rmsd, tm = [], []
		for model_id, model_file in self.get_grasp_model_file( sys_name = sys_name ):
			if model_id not in self.selected_models["grasp"][sys_name]:
				continue
			r, t = self.run_usalign(
				native_file = native_file,
				model_id2 = model_id,
				model2_file = model_file
			)
			rmsd.append( r )
			tm.append( t )
		return rmsd, tm


	def compute_per_sys_tm_alphalink2( self,
		sys_name: str,
		native_file: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute TM-score wrt the native structure for the given
			system for all predictions from AlphaLink2.
		AlphaLink2 predicts 25 models by default with 0-indexed model IDs.
		"""
		rmsd, tm = [], []
		for model_id, model_file in self.get_alphalink2_model_file( sys_name = sys_name ):
			if model_id not in self.selected_models["alphalink2"][sys_name]:
				continue
			r, t = self.run_usalign(
				native_file = native_file,
				model_id2 = model_id,
				model2_file = model_file
			)
			rmsd.append( r )
			tm.append( t )
		return rmsd, tm


	def compute_per_sys_tm_boltz2( self,
		sys_name: str,
		native_file: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute TM-score wrt the native structure for the given
			system for all predictions from Boltz2.
		We used Boltz2 for predicting 50 structures.
		"""
		rmsd, tm = [], []
		for model_id, model_file in self.get_boltz2_model_file( sys_name = sys_name ):
			if model_id not in self.selected_models["boltz2"][sys_name]:
				continue
			r, t = self.run_usalign(
				native_file = native_file,
				model_id2 = model_id,
				model2_file = model_file
			)
			rmsd.append( r )
			tm.append( t )
		return rmsd, tm


	def run_usalign( self,
		native_file: str,
		model_id2: int,
		model2_file: str
		) -> Tuple[float, float]:
		"""
		Run USalign to compute TM-score for the predicted
			model wrt native structure.
		"""
		stdout_file = usalign(
			usalign_script = self.usalign_script,
			model_id1 = 00,  # Arbitrary model_id for the native struct.
			model1_file = native_file,
			model_id2 = model_id2,
			model2_file = model2_file,
			tmp_dir = self.tm_tmp_dir,
			mol = "prot",
			mm = 1,
			ter = 1
			)
		rmsd, tm = get_alignment_score( stdout_file )
		return rmsd, tm

	################################################################################
	################################################################################
	def get_chain_mapping( self, native_chain_ids: str ) -> Dict[str, str]:
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


	def remap_chains_in_native( self ):
		"""
		For running DockQ, the chain IDs in the native
			and predicted structures must be the same.
		We remap chains in the native structure as its cheaper
			than doing so for all predicted structures.
		"""
		print( "Remapping chain IDs in the native structure..." )
		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			native_chain_ids = self.benchmark.loc[i, "Auth Asym ID"]
			print( f"{i}. {sys_name}:: {native_chain_ids}" )
			native_struct_file = get_native_struct_file(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name )

			base, ext = os.path.splitext( native_struct_file )
			native_remapped_file = base + "_remapped" + ext

			if not os.path.exists( native_remapped_file ):
				map_dict = self.get_chain_mapping( native_chain_ids )
				remap_chains_cif( struct_file = native_struct_file, map_dict = map_dict )


	def remap_chains_in_preds( self ):
		"""
		AlphaLink2 and GRASP provide a .pdb file as output.
		The chain numbering may not be consistent across all predicted
			structures (especially for GRASP).
		Remap all predicted structures for AlphaLink2 and GRASP.
		"""
		print( "Remapping chain IDs in GRASP and AlphaLink2 predicted structures..." )
		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( f"{i}. {sys_name}" )
			for model_id, model_file in self.get_grasp_model_file( sys_name = sys_name ):
				if sys_name not in self.selected_models["grasp"]:
					continue
				remapped_file = os.path.join(
					self.dockq_tmp_dir,
					f"{sys_name}_grasp_{model_id}.pdb" )
				if not os.path.exists( remapped_file ):
					remap_chains_pdb(
						struct_file = model_file,
						remapped_file = remapped_file )

			# for model_id, model_file in self.get_alphalink2_model_file( sys_name = sys_name ):
				# if sys_name not in self.selected_models["alphalink2"]:
				# 	continue
			# 	remapped_file = os.path.join(
			# 		self.dockq_tmp_dir,
			# 		f"{sys_name}_alphalink2_{model_id}.pdb" )
			# 	if not os.path.exists( remapped_file ):
			# 		remap_chains_pdb(
			# 			struct_file = model_file,
			# 			remapped_file = remapped_file )

	################################################################################
	################################################################################
	def compute_dockq_for_benchmark( self ):
		"""
		Compute the DockQ score wrt the native structure across the
			entire benchmark.
			Initial OpenFold structure
			IMP DL
			GRASP
			AlphaLink2
			Boltz2
		"""
		print( "\n" + "-"*70 +
			"\n\t\033[1m--> Computing DockQ wrt the ground truth structure <--\033[0m\n" +
			"-"*70 )
		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( f"{i}. {sys_name}" )
			if sys_name not in self.logs["dockq"]["init"]:
				dockq = self.compute_dockq_init_struct( sys_name = sys_name )
				self.logs["dockq"]["init"][sys_name] = dockq

			if self.model_type["imp_dl"]:
				if sys_name not in self.logs["dockq"][self.imp_dl]:
					dockq = self.compute_per_sys_dockq( sys_name = sys_name )
					self.logs["dockq"][self.imp_dl][sys_name] = dockq

			if self.model_type["grasp"]:
				if sys_name not in self.logs["dockq"]["grasp"]:
					dockq = self.compute_per_sys_dockq_grasp( sys_name = sys_name )
					self.logs["dockq"]["grasp"][sys_name] = dockq

			if self.model_type["alphalink2"]:
				if sys_name not in self.logs["dockq"]["alphalink2"]:
					dockq = self.compute_per_sys_tm_alphalink2( sys_name = sys_name )
					self.logs["dockq"]["alphalink2"][sys_name] = dockq

			if self.model_type["boltz2"]:
				if sys_name not in self.logs["dockq"]["boltz2"]:
					dockq = self.compute_per_sys_dockq_boltz2( sys_name = sys_name )
					self.logs["dockq"]["boltz2"][sys_name] = dockq

			write_json( self.logs, self.logs_file )


	def compute_dockq_init_struct( self, sys_name: str ):
		"""
		Compute DockQ for the initial OpenFold predicted structure.
		"""
		init_struct_file = get_init_struct_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
		)
		dockq_score = self.run_dockq( sys_name = sys_name, model_file = init_struct_file )
		return dockq_score


	def compute_per_sys_dockq( self,
		sys_name: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute DockQ wrt the native structure for the given
			system for all predictions from our method.
		"""
		analysis_dict_file = get_analysis_dict_path(
			base_dir = self.base_dir,
			sys_name = sys_name,
			modeling_dir_name = self.modeling_dir_name,
			modeling_version = self.modeling_version
		)
		analysis_dict = np.load( analysis_dict_file, allow_pickle = True ).item()
		model_ids = analysis_dict["selected_good_models"]
		del analysis_dict

		dockq_score = []
		for model_id in model_ids:
			model_file = get_relaxed_model_file(
				base_dir = self.base_dir,
				sys_name = sys_name,
				modeling_dir_name = self.modeling_dir_name,
				modeling_version = self.modeling_version,
				model_id = model_id
			)
			d = self.run_dockq( sys_name = sys_name, model_file = model_file )
			dockq_score.append( d )
		return dockq_score


	def compute_per_sys_dockq_grasp( self,
		sys_name: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute DockQ wrt the native structure for the given
			system for all predictions from GRASP.
		"""
		dockq_score = []
		for model_id, model_file in self.get_grasp_model_file( sys_name = sys_name, remapped = True ):
			if model_id not in self.selected_models["grasp"][sys_name]:
				continue
			d = self.run_dockq( sys_name = sys_name, model_file = model_file )
			dockq_score.append( d )
		return dockq_score


	def compute_per_sys_dockq_alphalink2( self,
		sys_name: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute DockQ wrt the native structure for the given
			system for all predictions from AlphaLink2.
		"""
		dockq_score = []
		for model_id, model_file in self.get_alphalink2_model_file( sys_name = sys_name, remapped = True ):
			if model_id not in self.selected_models["alphalink2"][sys_name]:
				continue
			d = self.run_dockq( sys_name = sys_name, model_file = model_file )
			dockq_score.append( d )
		return dockq_score


	def compute_per_sys_dockq_boltz2( self,
		sys_name: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute DockQ wrt the native structure for the given
			system for all predictions from Boltz2.
		"""
		dockq_score = []
		for model_id, model_file in self.get_boltz2_model_file( sys_name = sys_name, remapped = False ):
			if model_id not in self.selected_models["boltz2"][sys_name]:
				continue
			d = self.run_dockq( sys_name = sys_name, model_file = model_file )
			dockq_score.append( d )
		return dockq_score


	def run_dockq( self, sys_name: str, model_file: str ):
		"""
		Compute DockQ score for the given system.
		"""
		native_file = get_native_struct_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
		)
		base, ext = os.path.splitext( native_file )
		native_remapped_file = base + "_remapped" + ext
		dockq_score = dockq(
			native_file = native_remapped_file,
			model_file = model_file
		)
		return dockq_score

	################################################################################
	################################################################################
	def compute_data_sat_for_benchmark( self ):
		"""
		Compute data satisfaction for the entire benchmark.
			IMP DL
			GRASP
			AlphaLink2
			Boltz2
		"""
		print( "\n" + "-"*70 +
			"\n\t\033[1m--> Computing data satisfaction <--\033[0m\n" +
			"-"*70 )
		for i, sys_name in enumerate( self.benchmark["PDB ID"] ):
			print( f"{i}. {sys_name}" )

			if self.model_type["imp_dl"]:
				if sys_name not in self.logs["data_satisfaction"][self.imp_dl]:
					dockq = self.compute_per_sys_data_sat( sys_name = sys_name )
					self.logs["data_satisfaction"][self.imp_dl][sys_name] = dockq

			if self.model_type["grasp"]:
				if sys_name not in self.logs["data_satisfaction"]["grasp"]:
					dockq = self.compute_per_sys_data_sat_grasp( sys_name = sys_name )
					self.logs["data_satisfaction"]["grasp"][sys_name] = dockq

			if self.model_type["alphalink2"]:
				if sys_name not in self.logs["data_satisfaction"]["alphalink2"]:
					dockq = self.compute_per_sys_data_sat_alphalink2( sys_name = sys_name )
					self.logs["data_satisfaction"]["alphalink2"][sys_name] = dockq

			if self.model_type["boltz2"]:
				if sys_name not in self.logs["data_satisfaction"]["boltz2"]:
					dockq = self.compute_per_sys_data_sat_boltz2( sys_name = sys_name )
					self.logs["data_satisfaction"]["boltz2"][sys_name] = dockq

			write_json( self.logs, self.logs_file )


	def compute_per_sys_data_sat( self, sys_name: str ):
		"""
		Compute data satisfaction for all predicted models by
			our method for a given system.
		"""
		il_obj, topo_dict = init_integrative_learning_module(
			sys_name = sys_name,
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			modeling_dir_name = self.modeling_dir_name,
			sys_conf_suff = "_tpfp" )
		restraint_features = il_obj.run_data_gathering()

		metrics_func = Metrics( topo_dict["metrics"], restraint_features )

		analysis_dict_file = get_analysis_dict_path(
			base_dir = self.base_dir,
			sys_name = sys_name,
			modeling_dir_name = self.modeling_dir_name,
			modeling_version = self.modeling_version
		)
		analysis_dict = np.load( analysis_dict_file, allow_pickle = True ).item()
		model_ids = analysis_dict["selected_good_models"]
		del analysis_dict

		data_sat = []
		for model_id in model_ids:
			model_file = get_relaxed_model_file(
				base_dir = self.base_dir,
				sys_name = sys_name,
				modeling_dir_name = self.modeling_dir_name,
				modeling_version = self.modeling_version,
				model_id = model_id
			)
			d = self.compute_data_satisfaction_from_model(
				sys_name = sys_name,
				model_file = model_file,
				metrics_func = metrics_func )

			data_sat.append( d )
		return data_sat


	def compute_per_sys_data_sat_grasp( self,
		sys_name: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute data satisfaction for the given system
			for all predictions from GRASP.
		"""
		il_obj, topo_dict = init_integrative_learning_module(
			sys_name = sys_name,
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			modeling_dir_name = self.modeling_dir_name,
			sys_conf_suff = "_tpfp" )
		restraint_features = il_obj.run_data_gathering()
		metrics_func = Metrics( topo_dict["metrics"], restraint_features )

		data_sat = []
		for model_id, model_file in self.get_grasp_model_file( sys_name = sys_name, remapped = False ):
			if model_id not in self.selected_models["grasp"][sys_name]:
				continue
			d = self.compute_data_satisfaction_from_model(
				sys_name = sys_name,
				model_file = model_file,
				metrics_func = metrics_func )
			data_sat.append( d )
		return data_sat


	def compute_per_sys_data_sat_alphalink2( self,
		sys_name: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute data satisfaction for the given system
			for all predictions from AlphaLink2.
		"""
		il_obj, topo_dict = init_integrative_learning_module(
			sys_name = sys_name,
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			modeling_dir_name = self.modeling_dir_name,
			sys_conf_suff = "_tpfp" )
		restraint_features = il_obj.run_data_gathering()
		metrics_func = Metrics( topo_dict["metrics"], restraint_features )

		data_sat = []
		for model_id, model_file in self.get_alphalink2_model_file( sys_name = sys_name, remapped = False ):
			if model_id not in self.selected_models["alphalink2"][sys_name]:
				continue
			d = self.compute_data_satisfaction_from_model(
				sys_name = sys_name,
				model_file = model_file,
				metrics_func = metrics_func )
			data_sat.append( d )
		return data_sat


	def compute_per_sys_data_sat_boltz2( self,
		sys_name: str ) -> Tuple[List[float], List[float]]:
		"""
		Compute data satisfaction for the given system
			for all predictions from Boltz2.
		"""
		il_obj, topo_dict = init_integrative_learning_module(
			sys_name = sys_name,
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			modeling_dir_name = self.modeling_dir_name,
			sys_conf_suff = "_tpfp" )
		restraint_features = il_obj.run_data_gathering()
		metrics_func = Metrics( topo_dict["metrics"], restraint_features )

		data_sat = []
		for model_id, model_file in self.get_grasp_model_file( sys_name = sys_name, remapped = False ):
			if model_id not in self.selected_models["boltz2"][sys_name]:
				continue
			d = self.compute_data_satisfaction_from_model(
				sys_name = sys_name,
				model_file = model_file,
				metrics_func = metrics_func )
			data_sat.append( d )
		return data_sat


	def compute_data_satisfaction_from_model( self,
		sys_name: str,
		model_file: str,
		metrics_func: Metrics ):
		"""
		Given the file path for the predicted structure,
			compute the data satisfaction.
		Obtain the restraint_features for the system.
		Get the coordinates from the predicted structure.
		"""
		p = Parser( model_file )
		for model in p.get_models():
			coords_dict = p.get_coordinates( model )
		coords = np.concatenate( list( coords_dict.values() ), axis = 0 )

		pred = {"final_atom_positions": torch.from_numpy( coords ).unsqueeze( 0 )}
		metrics_dict = metrics_func.forward(
			out = pred, last_epoch = False  )

		return metrics_dict["xlr"].item()

	################################################################################
	################################################################################
	def create_plots( self ):
		"""
		Create the following plots:
		 	Distribution of TM-score and RMSD wrt the native structure.
			 Distribution of DockQ wrt the native structure.
			 Distribution of data satisfaction.
		"""
		for metric in ["tm", "rmsd", "dockq", "data_satisfaction"]:
			records = []
			for sys_name in self.benchmark["PDB ID"]:
				if self.model_type["imp_dl"]:
					for m in self.logs[metric][self.imp_dl][sys_name]:
						records.append( ( sys_name, self.imp_dl, m ) )
				if self.model_type["grasp"]:
					for m in  self.logs[metric]["grasp"][sys_name]:
						records.append( ( sys_name, "Grasp", m ) )
				if self.model_type["alphalink2"]:
					for m in  self.logs[metric]["alphalink2"][sys_name]:
						records.append( ( sys_name, "Alphalink2", m ) )
				if self.model_type["boltz2"]:
					for m in  self.logs[metric]["boltz2"][sys_name]:
						records.append( ( sys_name, "Boltz2", m ) )

			df = pd.DataFrame(
				records,
				columns = ["Complex", "Method", f"{metric.capitalize()}"] )

			plt.figure( figsize = ( 25, 10 ) )
			plt.rcParams["font.family"] = "sans"
			# ax = sns.violinplot(
			# 	data = df,
			# 	x = "Complex",
			# 	y = f"{metric.capitalize()}",
			# 	hue = "Method",
			# 	width = 1.0,
			# 	split = False,
			# 	inner = "quart",
			# 	cut = 0,
			# 	linewidth = 0.1
			# )
			ax = sns.boxenplot(
				data = df,
				x = "Complex",
				y = f"{metric.capitalize()}",
				hue = "Method",
				linewidth = 0.1
			)
			if metric == "dockq":
				ax.axhline( 0.23, color = "red" )

			ax.set_ylabel( f"{metric.capitalize()}", fontsize = 20 )
			ax.set_xlabel( "Complex", fontsize = 20 )
			plt.xticks( rotation = 90 )
			plt.tick_params(
				axis = "both",
				labelsize = 20,
				length = 10,
				width = 4
			)
			plt.legend( title = "Method" )
			plt.tight_layout()
			file = os.path.join( self.output_dir, f"{metric}_plot.png" )
			plt.savefig( file, dpi = 300 )
			plt.close()


if __name__ == "__main__":
	Comparison().forward()
