"""
Methods for loading pretrained model blocks from OpenFold
	and initialize with pretrained weights.
"""
import os
from typing import List
from collections import OrderedDict
import ml_collections as mlc

import torch

from openfold.model.embedders import (
    InputEmbedder,
    InputEmbedderMultimer,
    RecyclingEmbedder,
    TemplateEmbedder,
    TemplateEmbedderMultimer,
    ExtraMSAEmbedder,
    PreembeddingEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import StructureModule
from openfold.model.heads import (
	PerResidueLDDTCaPredictor, DistogramHead )


class LoadState():
	"""
	Load the parameters for pretrained models.
	"""
	def __init__( self, ofold_config: mlc.ConfigDict, mode: str, is_multimer: bool, device: str ):
		self.ofold_config = ofold_config
		self.mode = mode
		self.is_multimer = is_multimer
		self.device = device

		self.weights_dict = {}

		if self.mode == "mono":
			self.pretrained_weights = torch.load( "./monomer_params.pt" )

		elif self.mode == "multi":
			self.pretrained_weights = torch.load( os.path.abspath( "./multimer_params.pt" ) )
		else:
			raise ValueError( f"Incorrect mode: {self.mode} specified..." )


	def get_pretrained_weights( self, param_key: str ):
		"""
		Get the weights for the pretrained OpenFold monomer and multimer models.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		return OrderedDict(
			( ".".join( key.split( "." )[1:] ), self.pretrained_weights[key] )
				for key in self.pretrained_weights.keys() if param_key in key
		)

		#for layer in layers:
		#	for key in pretrained_weights.keys():
		#		if layer == "input_embedder":
		#			if self.is_multimer:
		#				self.input_embedder = InputEmbedderMultimer(
		#					**self.ofold_config["model"]["input_embedder"]
		#				)
		#			else:
		#				self.input_embedder = InputEmbedder(
		#					**self.ofold_config["model"]["input_embedder"],
		#				)
		#		if layer == "recycling_embedder":
		#			self.recycling_embedder = RecyclingEmbedder(
		#				**self.ofold_config["model"]["recycling_embedder"],
		#			)

		#		if layer == "template_embedder":
		#			if self.is_multimer:
		#				self.template_embedder = TemplateEmbedderMultimer(
		#					self.ofold_config["model"]["template"],
		#				)
		#			else:
		#				self.template_embedder = TemplateEmbedder(
		#					self.ofold_config["model"]["template"],
		#				)

		#		if layer == "extra_msa":
		#			self.extra_msa_embedder = ExtraMSAEmbedder(
		#				**self.ofold_config["model"]["extra_msa_embedder"],
		#			)
		#			self.extra_msa_stack = ExtraMSAStack(
		#				**self.ofold_config["model"]["extra_msa_stack"],
		#			)

		#		self.evoformer = EvoformerStack(
		#			**self.config["model"]["evoformer_stack"],
		#		)

		#		if layer == "structure_module":
		#			# Instantiate the structure module.
		#			self.structure_module = StructureModule(
		#				is_multimer = self.is_multimer,
		#				**self.ofold_config["model"]["structure_module"]
		#			).to( self.device )
		#			# Initialize the models with the pretrained weights.
		#			self.structure_module.load_state_dict( self.weights_dict["structure_module"] )

		#		if layer == "aux_head":
		#			self.aux_heads = AuxiliaryHeads(
		#				self.ofold_config["model"]["heads"],
		#			)

		#		if layer == "structure_module":
		#			# Obtain weights for the structure module only.
		#			self.weights_dict[layer] = OrderedDict(
		#						( ".".join( key.split( "." )[1:] ), pretrained_weights[key] )
		#						for key in pretrained_weights.keys() if "structure_module" in key
		#						)
		#		if layer == "lddt":
		#			# Obtain weights for the plddt head from auxillary heads module.
		#			self.weights_dict[layer] = OrderedDict(
		#						( ".".join( key.split( "." )[2:] ), pretrained_weights[key] )
		#						for key in pretrained_weights.keys() if "aux_heads.plddt" in key
		#						)
		#		if layer == "distogram":
		#			# Obtain weights for the distogram head from auxillary heads module.
		#			self.weights_dict[layer] = OrderedDict(
		#						( ".".join( key.split( "." )[2:] ), pretrained_weights[key] )
		#						for key in pretrained_weights.keys() if "aux_heads.distogram" in key
		#						)


	def load_pretrained_models( self, layers: List ):
		"""
		Load the required models.
		Intialize with the pretrained weights.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		print( "\nLoading the structure module..." )
		self.get_pretrained_weights( layers )

		for layer in layers:
			if layer == "input_embedder":
				if self.is_multimer:
					self.input_embedder = InputEmbedderMultimer(
						**self.ofold_config["model"]["input_embedder"]
					)
				else:
					self.input_embedder = InputEmbedder(
						**self.ofold_config["model"]["input_embedder"],
					)
				self.input_embedder.load_state_dict(
					self.get_pretrained_weights( param_key = "input_embedder" )
					)
			if layer == "recycling_embedder":
				self.recycling_embedder = RecyclingEmbedder(
					**self.ofold_config["model"]["recycling_embedder"],
				)
				self.recycling_embedder.load_state_dict(
					self.get_pretrained_weights( param_key = "recycling_embedder" )
					)

			if layer == "template_embedder":
				if self.is_multimer:
					self.template_embedder = TemplateEmbedderMultimer(
						self.ofold_config["model"]["template"],
					)
				else:
					self.template_embedder = TemplateEmbedder(
						self.ofold_config["model"]["template"],
					)
				self.template_embedder.load_state_dict(
					self.get_pretrained_weights( param_key = "template_embedder" )
					)

			if layer == "extra_msa":
				self.extra_msa_embedder = ExtraMSAEmbedder(
					**self.ofold_config["model"]["extra_msa_embedder"],
				)
				self.extra_msa_embedder.load_state_dict(
					self.get_pretrained_weights( param_key = "extra_msa_embedder" )
					)

				self.extra_msa_stack = ExtraMSAStack(
					**self.ofold_config["model"]["extra_msa_stack"],
				)
				self.extra_msa_stack.load_state_dict(
					self.get_pretrained_weights( param_key = "extra_msa_stack" )
					)

			if layer == "evoformer":
				self.evoformer = EvoformerStack(
					**self.config["model"]["evoformer_stack"],
				)
				self.evoformer.load_state_dict(
					self.get_pretrained_weights( param_key = "evoformer" )
					)

			if layer == "structure_module":
				# Instantiate the structure module.
				self.structure_module = StructureModule(
					is_multimer = self.is_multimer,
					**self.ofold_config["model"]["structure_module"]
				).to( self.device )
				#self.structure_module.load_state_dict( self.weights_dict["structure_module"] )
				self.structure_module.load_state_dict(
					self.get_pretrained_weights( param_key = "structure_module" )
					)

			if layer == "aux_head":
				self.aux_heads = AuxiliaryHeads(
					self.ofold_config["model"]["heads"],
				)
				self.aux_heads.load_state_dict(
					self.get_pretrained_weights( param_key = "aux_heads" )
					)

			if layer == "lddt":
				self.plddt = PerResidueLDDTCaPredictor(
													**self.ofold_config["model"]["heads"]["lddt"]
													).to( self.device )
				self.plddt.load_state_dict(
					self.get_pretrained_weights( param_key = "aux_heads.plddt" )
					)
				#self.plddt.load_state_dict( self.weights_dict["lddt"] )

			if layer == "distogram":
				self.distogram_head = DistogramHead(
													**self.ofold_config["model"]["heads"]["distogram"]
													).to( self.device )
				#self.distogram_head.load_state_dict( self.weights_dict["distogram"] )
				self.distogram_head.load_state_dict(
					self.get_pretrained_weights( param_key = "aux_heads.distogram" )
					)
