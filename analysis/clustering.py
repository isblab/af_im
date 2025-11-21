"""
Contains a wrapper class for various methods for
	selecting good-scoring models.
"""
from typing import List, Dict
import copy
from ml_collections import ConfigDict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class SelectGoodModels():
	"""
	Contains methods to perform score-based clustering.
	Obtain good scoring models.
	"""
	def __init__( self,
					config: ConfigDict,
					assessment_metrics: List ):
		self.config = config
		self.assessment_metrics = assessment_metrics

		self.good_models = np.array( [] )


	def forward( self, input_dict: Dict[str, np.array] ):
		"""
		Obtain good and bad scoring models using all
			clustering methods specified.
		"""
		self.input_dict = input_dict
		self.check_num_methods_enabled()
		
		self.good_models_index = self.run_clustering()
		if len( self.good_models_index ) == 0:
			raise Valueerror( f"No good-scoring models selected..." )


	def check_num_methods_enabled( self ):
		"""
		Assuming user specifies only one method to be used at a time.
		"""
		enabled = 0
		for clust_method in self.config.method:
			if self.config.method[clust_method].enabled:
				enabled += 1

		if enabled == 0:
			raise ValueError( f"At least one method needs to " +
				"be enabled for selecting good-scoring models..." )
		elif enabled > 1:
			raise ValueError( f"Too many methods enabled at once " +
				"for selecting good-scoring models..." )

	################################################################################
	################################################################################
	def prep_data( self ):
		"""
		Prepare data for downstream model selection.
		Scale all the metrics if specified.
		"""
		if self.config.scale_data:
			scaled_data = self.scale_data( data_dict = self.input_dict )
		else:
			scaled_data = copy.deepcopy( self.input_dict )

		if not self.config.method.quant_filter.enabled:
			stacked_data = self.get_data_stack( data_dict = scaled_data )
		else:
			stacked_data = copy.deepcopy( scaled_data )
			del scaled_data

		return stacked_data


	def scale_data( self, data_dict: Dict[str, np.array] ):
		"""
		Return a dict containing scaled metrics.
		Given an input dict containing the model_num, and the
			metrics to be used for clustering, do:
			1. Scale all the metrics.
			2. Stack them together in an array.
		"""
		scaled_data = {}
		scaler = StandardScaler()

		# Metrics/Loss terms to be used for clustering.
		for metric in self.assessment_metrics:
			category, name = metric.split( "-" )
			# As larger metric values indicate better model, we use the complement
			# 	(1 - metric) for harmony with loss values.
			data = copy( data_dict[name] )
			if category == "metric":
				data = 1 - data
			scaled_data[metric] = scaler.fit_transform( data )

		return scaled_data


	def get_data_stack( self, data_dict: Dict[str, np.array] ):
		"""
		Stack all metrics along the columns.
		"""
		# Metrics/Loss terms to be used for clustering.
		stacked_data  = []
		for metric in self.assessment_metrics:
			category, name = metric.split( "-" )
			data = copy( data_dict[name] )
			stacked_data.append( data )

		stacked_data = np.concatenate( stacked_data, axis = 1 )
		return stacked_data

	################################################################################
	################################################################################
	def run_clustering( self ):
		"""
		Run the specified clustering method and return the output.
		"""
		processed_data = self.prep_data()

		if self.config.method.kmeans.enabled:
			good_models_index = self.kmeans( data = processed_data )
		elif self.config.method.gmm.enabled:
			good_models_index = self.gmms( processed_data )
		elif self.config.method.quant_filter.enabled:
			good_models_index = self.quantile_filtering( processed_data )
		elif self.config.method.nds.enabled:
			good_models_index = self.non_dominated_sorting( processed_data )

		return good_models_index


	def kmeans( self, data: np.array ):
		"""
		Use KMeans for clustering the models into good and bad clusters.
		The clustering is based on the metrics provided in the input dict.
		"""
		config = self.config.method.kmeans

		kmeans = KMeans( n_clusters = config.n_clusters,
							random_state = config.random_state,
							n_init = config.n_init )
		kmeans.fit( data )
		labels = kmeans.labels_

		good_models_index = self.get_good_model_cluster( labels = labels )
		return good_models_index


	def gmms( self, data: np.array ):
		"""
		Use GMM for clustering the models into good and bad clusters.
		The clustering is based on the metrics provided in the input dict.
		"""
		config = self.config.method.gmm

		gm = GaussianMixture( n_components = config.n_components,
								random_state = config.random_state )
		gm.fit( data )
		labels = gm.predict( data )

		good_models_index = self.get_good_model_cluster( labels = labels )
		return good_models_index


	# def hdbscan( self, input_dict: Dict[str, List] ):
	# 	"""
	# 	Use HDBSCAN for identifying good scoring models.
	# 	The clustering is based on the metrics provided in the input dict.
	# 	"""
	# 	config = self.clustering_config.hdbscan

	# 	scaled_data = self.prep_data( input_dict = input_dict )

	# 	hdbscan = HDBSCAN( min_cluster_size = config.min_cluster_size,
	# 						min_samples = config.min_sample,
	# 						store_centers = config.store_centers )
	# 	hdbscan.fit( scaled_data )
	# 	labels = hdbscan.labels_
	# 	return labels


	def get_good_model_cluster( self, labels: List
								) -> np.array:
		"""
		Given a list of cluster labels, segregate into
			clusters and return as a dict.
		We consider the cluster with lower avg. violations as good-scoring models.
		Return the indices for the good scoring models.
		"""
		c1 = np.where( labels == 0 )
		c2 = np.where( labels == 1 )
		violations = self.input_dict["violations"]

		if np.mean( violations[c1] ) < np.mean( violations[c2] ):
			good_models_index = c1[0]
		else:
			good_models_index = c2[0]

		return good_models_index


	def quantile_filtering( self, data_dict: np.array ):
		"""
		Selct good-scoring models uisng a
			quantile-based filtering approach.
		Filter models based on data satisfaction.
		From the subset of models that satisfy data,
			select those that have fewer violations.
		Return the indices for the good scoring models.
		"""
		quantiles = self.config.method.quant_filter.quantiles
		assessment_metrics = self.assessment_metrics

		data_sat, viols = [], []
		for metric in assessment_metrics:
			category, name = metric.split( "-" )
			if category == "metrics":
				data = data_dict[name]
				q = np.quantile( data, quantiles[name] )
				data_sat.append( data >= q )
			elif category == "loss":
				pass
			else:
				raise ValueError( "Incorrect category for the assessment metric specified..." )

		data_sat = np.column_stack( data_sat )
		# Selecting models that satisfy all data types.
		data_sat_mask = np.prod( data_sat, axis = 1 )
		data_sat_idx = np.where( data_sat_mask == 1 )

		for metric in assessment_metrics:
			category, name = metric.split( "-" )
			if category == "metrics":
				pass
			elif category == "loss":
				v = data_dict[name]
				q = np.quantile( v[data_sat_idx], quantiles[name] )
				viols.append( v <= q )
			else:
				raise ValueError( "Incorrect category for the assessment metric specified..." )

		viols = np.column_stack( viols )
		viol_mask = np.prod( viols, axis = 1 )

		# Mask for models with high data satisfaction and low violations.
		mask = data_sat_mask.reshape( -1, 1 ) & viol_mask.reshape( -1, 1 )

		# Return indices for good-scoring models.
		good_models_index = np.where( mask == 1 )[0]
		return good_models_index

		# data_sat = []
		# for i, metric in enumerate( assessment_metrics ):
		# 	category, name = metric.split( "-" )
		# 	if category != "metrics":
		# 		continue
		# 	data = data_dict[name]
		# 	q = np.quantile( data, quantiles[name] )
		# 	data_sat.append( data >= q )

		# data_sat = np.column_stack( data_sat )
		# # Selecting models that satisfy all data types.
		# data_sat_mask = np.prod( data_sat, axis = 1 )
		# data_sat_idx = np.where( data_sat_mask == 1 )

		# violations = data_dict["violation"]
		# # Select those that have fewer violations.
		# q = np.quantile( violations[data_sat_idx], quantiles["violation"] )
		# viol_mask = violations <= q

		# mask = data_sat_mask.reshape( -1, 1 ) & viol_mask.reshape( -1, 1 )

		# # Return indices for good-scoring models.
		# good_models_index = np.where( mask == 1 )[0]
		# return good_models_index


	def nondominant_sorting( self, objectives: np.array ):
		"""
		Use non-dominant sorting to obtain
			the good-scoring models (Pareto set).
		Return the indices for the good scoring models.
		"""
		nds = NonDominatedSorting()
		# Obtain indices for non-dominant models (good-scoring models).
		good_models_index = nds.do( objectives, only_non_dominated_front = True )
		return good_models_index

