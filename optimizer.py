import torch
from torch import  nn
from torch import optim


class Optimizer( nn.Module ):
	def __init__( self, config ):
		self.config = config

	def forward( self, models ):
		"""
		Return the specified optimizer object.
		
		Input:
		----------
		model --> a Pytorch model object.
		"""
		params = []
		for m in models:
			params.extend( m.parameters() )
		if self.config.SGD.enabled:
			optimizer = optim.SGD( params = params,
									lr = self.config.SGD.lr,
									momentum = self.config.SGD.momentum,
									weight_decay = self.config.SGD.weight_decay
									 )

		elif self.config.Adam.enabled:
			optimizer = optim.Adam( params = params,
									lr = self.config.Adam.lr,
									amsgrad = self.config.Adam.amsgrad,
									weight_decay = self.config.Adam.weight_decay
									 )


		elif self.config.AdamW.enabled:
			optimizer = optim.SGD( params = params,
									lr = self.config.AdamW.lr,
									amsgrad = self.config.AdamW.amsgrad,
									weight_decay = self.config.AdamW.weight_decay
									 )

		else:
			raise Exception( "No optimizer specified..." )

		return optimizer
