"""
A wrapper to parse the confidence metrics for the predictions
	from AlphaLink2, Boltz2, GRASP.
"""
from typing import List, Dict
import os
import pandas as pd

from utils.utils import read_json, read_pkl_gz

class ConfidenceMetadataParser():
	def __init__(
		self,
		model: str,
		model_ids: List[int],
		confidence_files: List[str]
		):
		self.model = model
		self.model_ids = model_ids
		self.confidence_files = confidence_files


	def forward( self ):
		"""
		"""
		if self.model == "alphalink2":
			return self.parse_alphalink2_metrics()
		elif self.model == "boltz2":
			return self.parse_boltz2_metrics()
		elif self.model == "grasp":
			return self.parse_grasp_metrics()
		else:
			raise ValueError( f"Unsupported model: {self.model} specified..." )


	def parse_alphalink2_metrics( self ) -> Dict[int, float]:
		"""
		AlphaLink2 provides the full outputs.pkl.gz file.
		We can thus read any metric from there.
		To keep things in sync with other models, we will
			parse the confidence score (ipTM+pTM).
		"""
		confidence_metrics = {}
		for model_id, out_file in zip( self.model_ids, self.confidence_files ):
			data = read_pkl_gz( file_path = out_file )
			confidence_metrics[model_id] = data["iptm+ptm"]
		return confidence_metrics


	def parse_boltz2_metrics( self ) -> Dict[int, float]:
		"""
		Boltz2 provides a JSON file containing for the
			confidence metrics.
		We use tyhe confidence_score which is similar to the
			AF3 ranking_score.
		"""
		confidence_metrics = {}
		for model_id, out_file in zip( self.model_ids, self.confidence_files ):
			data = read_json( file_path = out_file )
			confidence_metrics[model_id] = data["confidence_score"]
		return confidence_metrics


	def parse_grasp_metrics( self ) -> Dict[int, float]:
		"""
		GRASP provides a TSV file containing for the
			confidence metrics.
		We use tyhe RankScore which is just ipTM+pTM.
		"""
		confidence_metrics = {}
		for model_id, out_file in zip( self.model_ids, self.confidence_files ):
			data = pd.read_csv( out_file, sep = "\t" )
			confidence_metrics[model_id] = data["RankScore"][0]
		return confidence_metrics

