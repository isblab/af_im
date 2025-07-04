"""
Contains a wrapper class for various clustering methods.
"""
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Clustering():
	"""
	Contains methods to perform score-based clustering.
	"""
	def __init__( self ):


	def forward( self ):
		"""
		"""


	def prep_data( self, input_dict: Dict[str, List] ):
		"""
		Given an input dict containing the model_num, and the
			metrics to be used for clustering, do:
			1. Scale all the metrics.
			2. Stack them together in an array.
		"""
		scaler = StandardScaler()
		scaled_data = []

		for name in input_dict:
			# As larger metric values indicate better model, we use (1 - metric)
			# 	for harmony with loss values.
			if "metric" in name:
				data = 1 - input_dict[name]
			else:
				data = input_dict[name]
			scaled_data.append( 
				scaler.fit_transform( data )
			 )
		scaled_data = np.hstack( scaled_data )
		return scaled_data


	def kmeans( self, input_dict: Dict[str, List] ):
		"""
		Use KMeans for clustering the models into good and bad clusters.
		The clustering is based on the metrics provided in the input dict.
		"""
		_, model_id = input_dict.pop( "model_id" )

		scaled_data = self.prep_data( input_dict = input_dict )

		kmeans = KMeans( n_clusters = 2,
							random_state = 0,
							n_init = "auto" )
		kmeans.fit( scaled_data )
		labels = kmeans.labels_



	def gmms( self, input_dict: Dict[str, List] ):
		"""
		Use GMM for clustering the models into good and bad clusters.
		The clustering is based on the metrics provided in the input dict.
		"""
		_, model_id = input_dict.pop( "model_id" )

		scaled_data = self.prep_data( input_dict = input_dict )

		gm = GaussianMixture( n_components = 2, random_state = 0 )
		gm.fit( scaled_data )
		labels = gm.predict()



	def get_good_n_bad_models( self, labels: List ):
		"""
		Given a list of cluster labels, segregate into
			clusters and return as a dict.
		"""



