"""
Script to perform analysis for the set of models obtained after fine-tuning.
Select good-scoring models.
Filter out structurally similar models.
Assess data satisfaction.
Perform AMBER relaxation.
Perform Molprobity validation.
"""
from typing import List, Dict
import os, glob, copy, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ml_collections import ConfigDict

from utils.utils import open_file_handler, run_subprocess
from analysis.clustering import SelectGoodModels
from analysis.rmsd import StructuralSimilarity
from analysis.amber import AmberRelaxation
from analysis.molprobity import Molprobity

from openfold.np import protein

class Assay():
	"""
	Perform analysis and validation of the predicted ensemble of models.
		1. Select good-scoring models.
		2. Filter out structurally similar models.
		3. Assess data satisfaction.
		4. Perform AMBER relaxation.
		5. Perform Molprobity validation.
	"""
	def __init__( self, sys_name: str,
						analysis_config: ConfigDict,
						modeling_output_dir: str,
						analysis_dir: str,
						struct_format: str,
						seed_worker,
						cores: int, prec: int ):
		seed_worker()
		# Name of the system to be modeled.
		self.sys_name = sys_name
		# Configd for analysis.
		self.analysis_config = analysis_config
		# Output modeling dir for the system.
		self.modeling_output_dir = modeling_output_dir
		self.analysis_dir = analysis_dir
		self.struct_format = struct_format
		# No. of CPU cores to use.
		self.cores = 5
		# precision for floats.
		self.prec = 4

		self.analysis_dict = {}


	def forward( self ):
		"""
		"""
		ts = time.perf_counter()
		self.create_required_paths()
		self.create_required_dir()
		self.load_analysis_dict()
		self.load_stats_file()
		self.run_analysis_pipeline()
		self.create_analysis_plot()

		time_taken = time.perf_counter() - ts
		self.analysis_dict["time_taken"] = time_taken
		print( f"Time taken for analysis = {time_taken/60} minutes " +
				f"OR {time_taken/3600} hours")
		self.save_analysis_dict()


	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		# Stat file from modeling output.
		self.stats_file = os.path.join( self.modeling_output_dir, "Stats.npy" )
		self.analysis_dict_file = os.path.join( self.analysis_dir,
											"analysis_dict.npy" )

		self.ensemble_dir = os.path.join( self.modeling_output_dir,
										f"{self.sys_name}_ensemble" )
		self.molprobity_csv_file = os.path.join( self.analysis_dir,
										"molprobity_validation.csv" )
		self.plot1_file = os.path.join( self.analysis_dir,
										"analysis_plot1.png" )
		self.plot2_file = os.path.join( self.analysis_dir,
										"analysis_plot2.png" )


	def create_required_dir( self ):
		"""
		Check if required dir exist.
		Create the required directories if not already existing.
		"""
		if not os.path.exists( self.modeling_output_dir ):
			raise FileNotFoundError( "Required modeling dir: " +
								f"{self.modeling_output_dir} does not exist..." )

		os.makedirs( self.analysis_dir, exist_ok = True )

	################################################################################
	################################################################################
	def load_stats_file( self ):
		"""
		Load the stats file.
		"""
		self.stats_dict = np.load( self.stats_file, allow_pickle = True ).item()


	def load_analysis_dict( self ):
		"""
		Load the stats file.
		"""
		if os.path.exists( self.analysis_dict_file ):
			self.analysis_dict = np.load(
				self.analysis_dict_file,
				allow_pickle = True ).item()


	def save_analysis_dict( self ):
		"""
		Save analysis_dict on disk.
		"""
		np.save( self.analysis_dict_file, self.analysis_dict, allow_pickle = True )

	################################################################################
	################################################################################
	def run_analysis_pipeline( self ):
		"""
		Run the different analysis steps.
		"""
		model_ids = np.array( self.stats_dict["model_id"] )
		protein_obj = self.stats_dict["protein"]

		input_dict = self.get_assessment_metrics()

		print( "\n--> Selecting good-scoring models <--" )
		if not "good_models" in self.analysis_dict:
			ts = time.perf_counter()
			good_models_index = self.select_good_scoring_models( input_dict )
			good_models = model_ids[good_models_index]
			self.analysis_dict["good_models_index"] = good_models_index
			self.analysis_dict["good_models"] = good_models
			self.save_analysis_dict()
			time_taken = time.perf_counter() - ts
			print( f"Time taken = {time_taken} seconds" )
		else:
			good_models = self.analysis_dict["good_models"]
			good_models_index = self.analysis_dict["good_models_index"]
		print( "Good-scoring models = ", len( good_models ) )

		#------------------------------------------------------------#
		print( "\n--> Removing structurally similar models <--" )
		if not "selected_good_models" in self.analysis_dict:
			ts = time.perf_counter()
			selected_good_models, rmsd_dict = self.filter_similar_models(
				good_models = good_models
			)
			# selected_good_models = model_ids[selected_model_index]
			# Get the index for the, selected good models, in the full set of models.
			selected_model_index = np.where(
					np.isin(
					np.array( model_ids ), np.array( selected_good_models )
					)
				)[0]
			self.analysis_dict["selected_model_index"] = selected_model_index
			self.analysis_dict["selected_good_models"] = selected_good_models
			self.analysis_dict["rmsd"] = rmsd_dict
			self.save_analysis_dict()
			time_taken = time.perf_counter() - ts
			print( f"Time taken = {time_taken} seconds" )
		else:
			selected_good_models = self.analysis_dict["selected_good_models"]
			selected_model_index = self.analysis_dict["selected_model_index"]
		print( "Selected good models = ", selected_good_models )

		#------------------------------------------------------------#
		print( "\n--> Assessing data satisfaction for good-scoring models <--" )
		self.assess_data_satisfaction( good_models_index = selected_model_index )

		if self.analysis_config.enable_relax_validate:
			#------------------------------------------------------------#
			print( "\n--> Running AMBER relaxation <--" )
			if not "relax" in self.analysis_dict:
				ts = time.perf_counter()
				subset_protein_obj = {k: protein_obj[k] for k in protein_obj if k in selected_model_index}
				relax_dict = self.run_amber_relaxation(
					good_models = selected_good_models,
					protein_obj = subset_protein_obj
				)
				self.analysis_dict["relax"]  = relax_dict
				self.save_analysis_dict()
				time_taken = time.perf_counter() - ts
				print( f"Time taken = {time_taken} seconds" )

			relaxed_model_dir = os.path.join(
				self.analysis_dir,
				self.analysis_config.relax.relaxed_model_dir )

			#------------------------------------------------------------#
			print( "\n--> Running MolProbity validation <--" )
			if not "molprob" in self.analysis_dict:
				ts = time.perf_counter()
				molprob_dict = self.run_molprobity_validation(
					good_models = selected_good_models,
					relaxed_model_dir = relaxed_model_dir
				)
				self.analysis_dict["molprob"] = molprob_dict
				self.save_analysis_dict()
				time_taken = time.perf_counter() - ts
				print( f"Time taken = {time_taken} seconds" )
			else:
				molprob_dict = self.analysis_dict["molprob"]
			self.write_molprobity_output_to_csv( molprob_dict = molprob_dict )
		else:
			print( "AMBER relaxation and Molprobity validation have been disabled..." )

	################################################################################
	################################################################################
	def get_assessment_metrics( self ) -> Dict[str, np.array]:
		"""
		Select the required metrics for downstream assessment.
		Also select the model_ids.
		"""
		input_dict = {
			"model_id": np.array( self.stats_dict["model_id"] ).reshape( -1, 1 )
		}

		assessment_metrics = self.analysis_config.assessment_metrics

		for metric in assessment_metrics:
			category, name = metric.split( "-" )
			input_dict[name] = np.array( self.stats_dict[category][name] ).reshape( -1, 1 )

		return input_dict


	def select_good_scoring_models( self,
		input_dict: Dict[str, List]
		) -> np.array:
		"""
		Given the ensemble of predicted structures, separate good and bad models.
		For this we use the loss trems: violation loss, and
			data satisfaction metrics: xl satisfaction.
		Good models: high data satisfaction and low physical violations.
		"""
		clust = SelectGoodModels(
				config = self.analysis_config.model_selection,
				assessment_metrics = self.analysis_config.assessment_metrics
			)
		clust.forward( input_dict = input_dict )
		good_models_index = clust.good_models_index
		return good_models_index

	################################################################################
	################################################################################
	def filter_similar_models( self, good_models: np.array
					) -> tuple[np.array, Dict[str, float]]:
		"""
		Remove models with high structural similarity (RMSD <= 0.5).
		"""
		struct_sim = StructuralSimilarity(
			sys_name = self.sys_name,
			rmsd_config = self.analysis_config.structural_similarity,
			model_ids = good_models,
			struct_format = self.struct_format,
			analysis_dir = self.analysis_dir,
			ensemble_dir = self.ensemble_dir
			)
		struct_sim.forward()
		selected_model_index = copy.copy( struct_sim.selected_model_index )
		rmsd_dict = copy.deepcopy( struct_sim.rmsd_dict )
		del struct_sim
		return selected_model_index, rmsd_dict

	################################################################################
	################################################################################
	def assess_data_satisfaction( self, good_models_index: np.array ):
		"""
		Assess data satisfaction for the good-scoring models.
		Currently implemented only for XL data.
		"""
		xlr = np.array( self.stats_dict["metrics"]["xlr"] )
		violation = np.array( self.stats_dict["loss"]["violation"] )
		ptm = np.array( self.stats_dict["metrics"]["ptm"] )
		iptm = np.array( self.stats_dict["metrics"]["iptm"] )
		global_satisfaction_array = self.stats_dict["metadata"]["xlr"]["xl_satisfaction_array"]
		global_satisfaction_array = global_satisfaction_array[good_models_index]

		num_xls = global_satisfaction_array.shape[1]
		total_satisfied = np.count_nonzero(
			np.sum( global_satisfaction_array, axis = 0 )
			)
		global_data_sat = total_satisfied/num_xls
		per_model_xl_sat = xlr[good_models_index]
		per_model_viol = violation[good_models_index]
		per_model_ptm = ptm[good_models_index]
		per_model_iptm = iptm[good_models_index]

		self.analysis_dict["per_model_xl_sat"] = per_model_xl_sat
		self.analysis_dict["per_model_viol"] = per_model_viol
		self.analysis_dict["per_model_ptm"] = per_model_ptm
		self.analysis_dict["per_model_iptm"] = per_model_iptm
		self.analysis_dict["global_data_satisfaction"] = global_data_sat
		print( "global data satisfaction = ", global_data_sat )

	################################################################################
	################################################################################
	def run_amber_relaxation( self, good_models: np.array, protein_obj: Dict[int, protein.Protein] ):
		"""
		Run AMBER relaxation for the good-scoring models.
		"""
		relax = AmberRelaxation(
			sys_name = self.sys_name,
			amber_config = self.analysis_config.relax,
			model_ids = good_models,
			protein_obj_dict = protein_obj,
			analysis_dir = self.analysis_dir
		)
		relax.forward()

		relax_dict = copy.deepcopy( relax.relax_dict )
		del relax
		return relax_dict

	################################################################################
	################################################################################
	def run_molprobity_validation( self, good_models: np.array,
									relaxed_model_dir: str ):
		"""
		Run MolProbity validation for the good-scoring models.
		"""
		molprob = Molprobity(
			sys_name = self.sys_name,
			molprob_config = self.analysis_config.molprobity,
			model_ids = good_models,
			cores = self.cores,
			struct_format = self.struct_format,
			analysis_dir = self.analysis_dir,
			relax_ensemble_dir = relaxed_model_dir,
			unrelax_ensemble_dir = self.ensemble_dir
		)
		molprob.forward()

		molprob_dict = copy.deepcopy( molprob.molprob_dict )
		del molprob
		return molprob_dict


	def write_molprobity_output_to_csv( self, molprob_dict: Dict[str, Dict] ):
		"""
		Write the MolProbity validation metrics to a csv file.
		"""
		flat_dict = {k:[] for k in ["model_id", "type"]}
		for model_id in molprob_dict:
			for t in ["unrelaxed","relaxed"]:
				flat_dict["model_id"].append( model_id )
				flat_dict["type"].append( t )
				for k in molprob_dict[model_id]["unrelaxed"]:
					if k not in flat_dict:
						flat_dict[k] = []
					flat_dict[k].append( molprob_dict[model_id][t][k] )
		df = pd.DataFrame( flat_dict )
		df.to_csv( self.molprobity_csv_file, index = False )

	################################################################################
	################################################################################
	def create_violin( self, data: List, ax, r: int, color: str, ylabel: str ):
		"""
		Create a violinplot with the required formatting.
		"""
		vp = ax[r].violinplot( dataset = data, orientation = "vertical",
									showmeans = True, showextrema = True )
		for body in vp["bodies"]:
			# Adjust transparency (transparent: 0; opaque: 1).
			body.set_alpha( 0.7 )
			body.set_facecolor( color )
		# Change color and width of the central line.
		vp["cbars"].set_color( "black" )
		vp["cbars"].set_linewidth( 2 )
		# Change color and width of the minimum line.
		vp["cmins"].set_color( "black" )
		vp["cmins"].set_linewidth( 2 )
		# Change color and width of the maximum line.
		vp["cmaxes"].set_color( "black" )
		vp["cmaxes"].set_linewidth( 2 )
		# Change color and width of the mean line.
		vp["cmeans"].set_color( "blue" )
		vp["cmeans"].set_linewidth( 4 )
		ax[r].tick_params( axis = "both" , labelsize = 25, length = 10, width = 4 )
		ax[r].set_ylabel( ylabel, fontsize = 25 )
		ax[r].set_xticks( [1, 2], ["All sampled", "Good scoring"] )
		return ax


	def create_analysis_plot( self ):
		"""
		Create plots for all sampled vs good-scoring models
			1. Distribution of violations
			2. Distribution of xl satisfaction
			3. istribution of pLDDT.
			4. istribution of ipTM.
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 1, 2, figsize = ( 30, 20 ) )

		as_xl = self.stats_dict["metrics"]["xlr"]
		gs_xl = self.analysis_dict["per_model_xl_sat"]
		ax = self.create_violin( data = [as_xl, gs_xl], ax = ax, r = 0,
							color = "orange", ylabel = "per model XL satisfaction" )
		ax[0].set_ylim( -0.1, 1.1 )

		as_viol = self.stats_dict["loss"]["violation"]
		gs_viol = self.analysis_dict["per_model_viol"]
		self.create_violin( data = [as_viol, gs_viol], ax = ax, r = 1,
							color = "orange", ylabel = "per model Violations" )

		plt.tight_layout()
		plt.savefig( self.plot1_file, dpi = 300 )
		plt.close()

		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 1, 2, figsize = ( 30, 20 ) )

		as_ptm = self.stats_dict["metrics"]["ptm"]
		gs_ptm = self.analysis_dict["per_model_ptm"]
		self.create_violin( data = [as_ptm, gs_ptm], ax = ax, r = 0,
							color = "orange", ylabel = "per model pTM" )
		ax[0].set_ylim( -0.1, 1.1 )

		as_iptm = self.stats_dict["metrics"]["iptm"]
		gs_iptm = self.analysis_dict["per_model_iptm"]
		self.create_violin( data = [as_iptm, gs_iptm], ax = ax, r = 1,
							color = "orange", ylabel = "per model ipTM" )
		ax[1].set_ylim( -0.1, 1.1 )

		plt.tight_layout()
		plt.savefig( self.plot2_file, dpi = 300 )
		plt.close()


		# ax[0].violinplot( dataset = [as_viol, gs_viol], orientation = "vertical",
		# 						showmeans = True, showextrema = True )
		# ax[0].set_title( "Per model Violations", fontsize = 35 )
		# ax[0].set_ylabel( "Per model Violations", fontsize = 35 )
		# ax[0].tick_params( axis = "both" , labelsize = 35, length = 10, width = 4 )
		# ax[0].set_xticks( [1, 2], ["All sampled", "Good scoring"] )

		# as_xl = self.stats_dict["metrics"]["xlr"]
		# as_avg_xl = np.mean( as_xl )
		# gs_xl = self.analysis_dict["per_model_xl_sat"]
		# gs_avg_xl = np.mean( gs_xl )
		# ax[1].violinplot( dataset = [as_xl, gs_xl], orientation = "vertical",
		# 						showmeans = True, showextrema = True )
		# # ax[1].set_title( "Per model XL satisfaction", fontsize = 35 )
		# ax[1].set_ylabel( "Per model XL satisfaction", fontsize = 35 )
		# ax[1].tick_params( axis = "both" , labelsize = 35, length = 10, width = 4 )
		# ax[1].set_xticks( [1, 2], ["All sampled", "Good scoring"] )

