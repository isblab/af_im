import torch
from torch import  nn
from torch import optim


class Optimizer( nn.Module ):
	def __init__( self, config ):
		self.config = config

	def forward( self, model ):
		"""
		Return the specified optimizer object.
		
		Input:
		----------
		model --> a Pytorch model object.
		"""
		if self.config.optimizer.SGD.enabled:
			Optimizer = optim.SGD( params = model.parameters(),
									lr = self.config.optimizer.SGD.lr,
									momentum = self.config.optimizer.SGD.momemtum,
									weight_decay = self.config.optimizer.SGD.weight_decay
									 )

		elif self.config.optimizer.Adam.enabled:
			Optimizer = optim.Adam( params = model.parameters(),
									lr = self.config.optimizer.Adam.lr,
									amsgrad = self.config.optimizer.Adam.amsgrad,
									momentum = self.config.optimizer.Adam.momemtum,
									weight_decay = self.config.optimizer.Adam.weight_decay
									 )


		elif self.config.optimizer.AdamW.enabled:
			Optimizer = optim.SGD( params = model.parameters(),
									lr = self.config.optimizer.AdamW.lr,
									amsgrad = self.config.optimizer.AdamW.amsgrad,
									momentum = self.config.optimizer.AdamW.momemtum,
									weight_decay = self.config.optimizer.AdamW.weight_decay
									 )

		return optimizer
