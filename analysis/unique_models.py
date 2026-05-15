"""
Contains module to obtain the structurally unique models based
	on the specified metric.
"""
from typing import Dict
import networkx as nx


class UniqueStructures():
	"""
	Identify unique structures by clustering models based on a
		pairwise structural similarity metric (TM-score, DockQ).

	The input metric dictionary stores pairwise similarity values
		between models:
	self.metric_dict: {
		model_id1: {
			model_id2: metric_value
		}
	}
	model_id1, model_id2 may not be in order (0-N),
		where, N is the total no. of models.
	"""
	def __init__(
		self,
		metric_dict: Dict[int, Dict[int, float]],
		metric_name: str,
		threshold: float
		):
		"""
		Inputs:
		----------
		metric_dict: nested dictcontaining pairwise metric values.
		threshold: minimum similarity value required to connect
			two models.
		"""
		self.metric_dict = metric_dict
		self.metric_name = metric_name
		self.threshold = threshold



	def forward( self ):
		"""
		Cluster structurally similar models using the specified metric and select
			one representative per cluster.

		We represent models as graph nodes.
		Two models are connected if their similarity metric exceeds a threshold.
		Connected components are interpreted as clusters of structurally
			similar models.

		Returns:
		----------
		representatives: list of model IDs corresponding to one representative
			per cluster.
		"""
		G = self.create_graph()
		clusters = self.find_connected_components( G = G )
		representatives = self.select_cluster_representative( clusters = clusters )
		return representatives


	def create_graph( self ) -> nx.Graph:
		"""
		Create a graph of pairwise structural similarity metric values
			with the models as nodes and the metric value as edge.
		Models with metric value >= threshold are connected.
		We are assuming that model_id1 and model_id2 represent
			the same pool of models i.e. we want to find unique
			models within a pool and not compare two spearate pools
			of models.
		Self-connections are ignored.

		Returns:
		----------
		G: networkX graph where nodes are model IDs and
			edges indicate structurally similar models.
		"""
		G = nx.Graph()
		model_ids = list( self.metric_dict.keys() )
		G.add_nodes_from( model_ids )

		for k_i in self.metric_dict:
			for k_j, metrics in self.metric_dict[k_i].items():

				if k_i == k_j:
					continue

				if self.metric_name == "tm":
					value = metrics[self.metric_name]
				else:
					value = metrics
				if value >= self.threshold:
					G.add_edge( k_i, k_j )

		return G


	def find_connected_components( self, G ):
		"""
		Identify clusters of structurally similar models.
		Connected components define clusters where models are linked
			directly or indirectly through the similarity threshold.

		Returns:
		----------
		clusters: list of sets containing model IDs belonging
			to the same structural cluster.

		For example, if
			A--B (TM=0.8)
			B--C (TM=0.8)
			A--C (TM=0.6)
			then (A, B, C) form a cluster.
		"""
		clusters = list( nx.connected_components( G ) )
		return clusters


	def select_cluster_representative( self, clusters ):
		"""
		Select one representative model from each cluster.
		We choose the 1st model as the representtative based
			on the sorted model_id's.
		"""
		representatives = []
		for cluster in clusters:
			# Sets can be indexed; selecting the min model_id
			# 	corresponds to the taking the first model.
			representatives.append( min( cluster ) )
		return representatives

