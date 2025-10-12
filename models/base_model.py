"""
Contains a base model class to define attributes and methods for
	all downstream model classes.
"""
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import ml_collections as mlc

import torch
from torch import nn


class Model( nn.Module, ABC ):
	"""
	Base class for all model classes to specify the necessary methods.
	"""
	def __init__( self ):
		super( Model, self ).__init__()
		pass

	@abstractmethod
	def predict( self, evo_output: Dict[str, torch.Tensor],
					gt_features: Dict[str, torch.Tensor],
					batch: Dict[str, torch.Tensor]
					# device: str
			) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
		"""
		Run the structure module and auxillary heads module.

		Input:
		----------
		evo_output --> dict containing Pair, Single representtaions.
		gt_features --> dict contaiining the ground truth features.

		Returns:
		----------
		outputs --> dict containing the output from structure module and auxillary heads module.
		"""


	@abstractmethod
	def params( self ) -> List[nn.Module]:

		"""
		Return a list of models for the optimizer.
		"""