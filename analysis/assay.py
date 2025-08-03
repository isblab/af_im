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
		self.plot_file = os.path.join( self.analysis_dir,
										"analysis_plot1.png" )


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

		print( "\n--> Removing structurally similar models <--" )
		if not "selected_good_models" in self.analysis_dict:
			ts = time.perf_counter()
			selected_model_index, rmsd_dict = self.filter_similar_models(
				good_models = good_models
			)
			selected_good_models = model_ids[selected_model_index]
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

		print( "\n--> Assessing data satisfaction for good-scoring models <--" )
		self.assess_data_satisfaction( good_models_index = selected_model_index )

		if self.analysis_config.enable_relax_validate:
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
		For this we use the loss trems: violation loss, ccom loss, and
			data satisfaction metrics: xl satisfaction.
		Good models: high data satisfaction and low physical violations.
		"""
		clust = SelectGoodModels(
				config = self.analysis_config.model_selection,
				assessment_metrics = self.analysis_config.assessment_metrics
			)
		clust.forward( input_dict = input_dict )
		good_models_index = clust.good_models
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
		ccom = np.array( self.stats_dict["loss"]["chain_center_of_mass"] )
		global_satisfaction_array = self.stats_dict["metadata"]["xlr"]["xl_satisfaction_array"]
		global_satisfaction_array = global_satisfaction_array[good_models_index]

		num_xls = global_satisfaction_array.shape[1]
		total_satisfied = np.count_nonzero(
			np.sum( global_satisfaction_array, axis = 0 )
			)
		global_data_sat = total_satisfied/num_xls
		per_model_xl_sat = xlr[good_models_index]
		per_model_viol = violation[good_models_index]
		per_model_ccom = ccom[good_models_index]

		self.analysis_dict["per_model_xl_sat"] = per_model_xl_sat
		self.analysis_dict["per_model_viol"] = per_model_viol
		self.analysis_dict["per_model_ccom"] = per_model_ccom
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


	def create_analysis_plot( self ):
		"""
		Create plots for all sampled vs good-scoring models
			1. Distribution of violations
			2. Distribution of xl satisfaction
		"""
		plt.rcParams["font.family"] = "sans"
		_, ax = plt.subplots( 1, 2, figsize = ( 30, 20 ) )

		as_xl = self.stats_dict["metrics"]["xlr"]
		gs_xl = self.analysis_dict["per_model_xl_sat"]
		self.create_violin( data = [as_xl, gs_xl], ax = ax, r = 0,
							color = "orange", ylabel = "per model XL satisfaction" )

		as_viol = self.stats_dict["loss"]["violation"]
		gs_viol = self.analysis_dict["per_model_viol"]
		self.create_violin( data = [as_viol, gs_viol], ax = ax, r = 1,
							color = "orange", ylabel = "per model Violations" )


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

		plt.savefig( self.plot_file, dpi = 300 )
		plt.close()




	# def compute_model_rmsd( self, prot1: torch.Tensor, prot2: torch.Tensor ):
	# 	"""
	# 	Given the AF2 predicted atom positions ([N,37,3]) for
	# 		a pair of models, compute the Ca-RMSD.
	# 	"""
	# 	ca_idx = residue_constants.atom_order["CA"]
	# 	prot1_ca = prot2_1[:,ca_idx,:].numpy()
	# 	prot2_ca = prot2_2[:,ca_idx,:].numpy()
	# 	rot, rssd = R.align_vectors( prot1_ca, prot2_ca )
	# 	rmsd = rssd/np.sqrt( prot1_ca.shape[0] )
	# 	return rmsd


	# def assess_structural_similarity( self, good_models: np.array ):
	# 	"""
	# 	Given the good-scoring models, remove
	# 		structurally similar models (RMSD <= 0.5).
	# 	"""
	# 	selected_good_models = np.array( [] )
	# 	ignore_models = []
	# 	if len( good_models ) == 1:
	# 		selected_good_models = copy( good_models )
	# 	else:
	# 		protein_obj = self.stats_dict["protein"]
	# 		for i in good_models:
	# 			if i in ignore_models:
	# 				continue
	# 			for j in good_models[1:]:
	# 				rmsd = self.compute_model_rmsd(
	# 					prot1 = protein_obj[i], prot2 = protein_obj[j]
	# 					)
	# 				if rmsd <= 0.5:
	# 					ignore_models.append( j )
	# 				else:
	# 					selected_good_models = np.append( selected_good_models, i )
	# 	return selected_good_models



	# def create_analysis_dir( self ):
	# 	"""
	# 	Create a dir to store analysis results.
	# 	"""
	# 	self.analysis_dir = os.path.join( self.output_dir, "analysis" )
	# 	if not os.path.exists( self.analysis_dir ):
	# 		os.makedirs( self.analysis_dir )

	# # 	self.tmp_dir = os.path.join( self.analysis_dir, "tmp" )
	# # 	if not os.path.exists( self.tmp_dir ):
	# # 		os.makedirs( self.tmp_dir )



	# def prep_data( self ) -> np.array:
	# 	"""
	# 	Given the dict containing loss and metrics, do
	# 		Select the following:
	# 			Violation loss, CCOM loss, restraint loss.
	# 			Restraint metrics
	# 		Stack together in an array.
	# 	"""
	# 	# Convert all metric values to -- (1 - metric).
	# 	# 	Lower (1 - metric) the better -- same as loss.
	# 	metrics = {}
	# 	for k in self.metrics_dict:
	# 		metrics[k] = 1 - np.array( self.metrics_dict )

	# 	data = np.stack( 
	# 			[
	# 				self.loss_dict["violation"],
	# 				self.loss_dict["chain_center_of_mass"],
	# 				self.loss_dict["xlr"],
	# 				metrics
	# 			],
	# 			axis = 1
	# 	)

	# 	return data


	# def get_scaled_data( self, data: np.array ) -> np.array:
	# 	"""
	# 	Perform standard scaling.
	# 	"""
	# 	scaler = StandardScaler()
	# 	scaled_data = scaler.fit_transform( data )

	# 	return scaled_data


	# def get_hdbscan_clusters( self, data: np.array ) -> Dict[int, List[int]]:
	# 	"""
	# 	Run HDBSCAN clustering the return dict for all clusters.
	# 	"""
	# 	hdb = hdbscan.HDBSCAN( min_cluster_size = self.min_cluster_size,
	# 							min_samples = self.min_samples )
	# 	hdb.fit( scaled_data )

	# 	# Get cluster labels for each data point.
	# 	labels = hdb.labels_

	# 	hdb_clusters = {}
	# 	for label in np.unique(labels):
	# 		# Get indices for all data points having a label.
	# 		hdb_clusters[label] = np.array( 
	# 								np.where( labels == label )[0]
	# 								)

	# 	return hdb_clusters


	# def model_filtering( self ) -> List:
	# 	"""
	# 	Remove outlier models from the given set of models.
	# 	Use HDBSCAN to identify such noise models (unclustered).
	# 		Consider the all except the unclustered models for further analysis.
	# 		Use violation loss, CCOM loss, restraint loss, and 
	# 			restraint metrics for clustering.
	# 	"""
	# 	data = self.prep_data()
	# 	# The model at epoch 0 will be kept.
	# 	scaled_data = self.get_scaled_data( data[1:, :] )

	# 	clusters = self.hdb_clusters( scaled_data )

	# 	# Add the epoch 0 model.
	# 	selected_models = [0]
	# 	# Add 1 to all cluster indices to correct for the 
	# 	# 	numbering due to removing the epoch 0 model.
	# 	for label in clusters:
	# 		clusters[label] += 1

	# 		if label == -1:
	# 			unclustered_models = clusters[label]
	# 		else:
	# 			# Ignore the unclustered models.
	# 			selected_models.extend( clusters[label] )

	# 	selected_models = sorted( selected_models )
	# 	return unclustered_models, selected_models


	# def save_loss_metrics( self, model_ids: List[int], file_name: str ):
	# 	"""
	# 	Select the loss and metric values for the given model IDs and save on disk.
	# 	"""
	# 	df_dict = {"model_id": model_ids}
	# 	for k in self.loss_dict:
	# 		v = np.array( self.loss_dict[k] )
	# 		df_dict[k] = v[model_ids]

	# 	for k in self.metrics_dict:
	# 		v = np.array( self.metrics_dict[k] )
	# 		df_dict[f"{k}_metric"] = np.round( 
	# 										np.mean( v[model_ids]), 
	# 										self.prec
	# 										)
	# 	file_name = os.path.join( self.analysis_dir, f"summary_{file_name}.csv" )
	# 	df.to_csv( file_name, index = False )

