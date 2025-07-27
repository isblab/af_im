"""
Contains a wrapper class for various clustering methods.
"""
from typing import List, Dict
from ml_collections import ConfigDict
import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class Clustering():
	"""
	Contains methods to perform score-based clustering.
	Obtain good scoring models.
	"""
	def __init__( self,
					clustering_config: ConfigDict,
					assessment_metrics: List ):
		self.clustering_config = clustering_config
		self.assessment_metrics = assessment_metrics

		self.cluster_dict = {}


	def forward( self, input_dict: Dict[str, np.array] ):
		"""
		Obtain good and bad scoring models using all
			clustering methods specified.
		"""
		for clust_method in self.clustering_config.method:
		
		self.cluster_dict[clust_method] = self.run_clustering(
			clust_method = clust_method,
			input_dict = input_dict
			)


	def prep_data( self, input_dict: Dict[str, np.array] ):
		"""
		Given an input dict containing the model_num, and the
			metrics to be used for clustering, do:
			1. Scale all the metrics.
			2. Stack them together in an array.
		"""
		scaler = StandardScaler()
		processed_data = []

		# Metrics/Loss terms to be used for clustering.
		for metric in self.assessment_metrics:
			category, name = metric.split( "-" )
			# As larger metric values indicate better model, we use the complement
			# 	(1 - metric) for harmony with loss values.
			if category == "metric":
				data = 1 - input_dict[category][name]
			else:
				data = input_dict[category][name]
			if self.clustering_config.scale_data:
				processed_data.append( 
					scaler.fit_transform( data )
				 )
			else:
				processed_data.append( data )
		processed_data = np.concatenate( processed_data, axis = 1 )
		return processed_data


	def run_clustering( slef, clust_method: str, input_dict: Dict[str, np.array] ):
		"""
		Run the specified clustering method and return the output.
		"""
		procesed_data = self.prep_data( input_dict = input_dict )

		if clust_method == "kmeans":
			labels = self.kmeans( data = procesed_data )
		elif clust_method == "gmm":
			labels = self.gmms( procesed_data )
		elif clust_method == "hdbscan":
			labels = self.hdbscan( procesed_data )

		cluster_dict = self.get_good_n_bad_models( labels )

		return cluster_dict


	def kmeans( self, data: np.array ):
		"""
		Use KMeans for clustering the models into good and bad clusters.
		The clustering is based on the metrics provided in the input dict.
		"""
		config = self.clustering_config.kmeans

		kmeans = KMeans( n_clusters = config.n_clusters,
							random_state = config.random_state,
							n_init = config.n_init )
		kmeans.fit( data )
		labels = kmeans.labels_
		return labels


	def gmms( self, data: np.array ):
		"""
		Use GMM for clustering the models into good and bad clusters.
		The clustering is based on the metrics provided in the input dict.
		"""
		config = self.clustering_config.gmm

		gm = GaussianMixture( n_components = config.n_components,
								random_state = config.random_state )
		gm.fit( scaled_data )
		labels = gm.predict( data )
		return labels


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


	def get_good_n_bad_models( self, labels: List,
								input_dict: Dict[str, np.array]
								) -> Dict[str, np.array]:
		"""
		Given a list of cluster labels, segregate into
			clusters and return as a dict.
		We consider the cluster with lower avg. violations as good-scoring models.
		"""
		c1 = np.where( labels == 0 )
		c2 = np.where( labels == 1 )
		violations = input_dict["violations"]

		if np.mean( violations[c1] ) < np.mean( violations[c2] ):
			good_models_idx = c1
			badd_models_idx = c2
		else:
			good_models_idx = c2
			badd_models_idx = c1

		cluster_dict = {
			"good_models": good_models_idx,
			"bad_models": bad_models_idx
		}
		return cluster_dict

