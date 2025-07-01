from typing import Dict, Tuple, Optional
import os, glob, time, copy, random
from collections import ( OrderedDict, defaultdict )
import pickle as pkl
import numpy as np
import pandas as pd
import ml_collections as mlc


import torch
from torch import nn
from torch.cuda.amp import GradScaler

from openfold.data import feature_pipeline
from openfold.model.structure_module import StructureModule
from openfold.model.heads import AuxiliaryHeads
from openfold.utils.multi_chain_permutation import multi_chain_permutation_align, split_ground_truth_labels
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.feats import atom14_to_atom37
from openfold.np import protein
from openfold.np.relax import relax

from model import get_model
from loss import LossFunction
from metrics import Metrics
from optimizer import Optimizer
from utils.pdb_utils import ( prep_protein, SaveModels )



def parse_nested_dict( dict_: Dict, action: str, 
						device: Optional[str] = "cuda" ):
	for k in dict_:
		if isinstance( dict_[k], Dict ):
			dict_[k] = parse_nested_dict( dict_[k], action, device )
		else:
			if isinstance( dict_[k], torch.Tensor ):
				if action == "add_dim":
					dict_[k] = dict_[k].unsqueeze( 0 )
				elif action == "add_to_device":
					dict_[k] = dict_[k].to( device )
				elif action == "detach":
					dict_[k] = dict_[k].detach().cpu()
	return dict_


class FitToData():
	def __init__( self, sys_name: str,
					ofold_config: mlc.ConfigDict,
					topology: mlc.ConfigDict,
					mode: str,
					system_features: Dict,
					ofold_output_dir: str,
					output_dir: str,
					prec: int,
					seed_worker,
					device: str ):
		self.sys_name = sys_name
		self.ofold_config = ofold_config
		self.topology = topology
		self.is_multimer = self.ofold_config.globals.is_multimer,
		self.mode = mode
		self.prec = prec
		self.device = device
		self.system_features = system_features
		self.ofold_output_dir = ofold_output_dir
		self.output_dir = output_dir
		# self.use_relaxation = False
		# PDB file contaiing all predicted models.
		self.ensemble_file = os.path.join( self.output_dir, f"{self.sys_name}_output_models" )
		# Directory to store each predicted model as separate PDB file.
		self.ensemble_dir = os.path.join( self.output_dir, f"{self.sys_name}_ensemble" )
		# Directory to store each relaxed predicted model as separate PDB file.
		# self.relax_ensemble_dir = os.path.join( self.output_dir, f"{self.sys_name}_relax_ensemble" )

		if not os.path.exists( self.ensemble_dir ):
			os.makedirs( self.ensemble_dir )

		# if self.use_relaxation:
		# 	if not os.path.exists( self.relax_ensemble_dir ):
		# 		os.makedirs( self.relax_ensemble_dir )
		
		# Set the seeds.
		seed_worker()

		# Add a singleton batch dim.
		self.add_batch_dim()

		self.stats_dict = defaultdict( dict )
		self.loss_fn = LossFunction( self.topology["loss"], self.device )
		self.loss_dict = {}
		self.metrics_fn = Metrics( self.topology["metrics"], self.system_features["restraint_features"] )
		self.scalar_metric_dict = {}
		self.other_metric_dict = {}



	def forward( self ):
		"""
		"""
		self.load_feature_dict()
		self.fit()


	# def ensemble_exists( self ):
	# 	"""
	# 	Check if the ensemble file already exists.
	# 		If exists --> Do not run the finetuning process
	# 	"""
	# 	return os.path.exists( f"{self.ensemble_file}.pdb" )



	def load_feature_dict( self ) -> None:
		"""
		Load the feature_dict saved as a .pkl file in the system's director.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )
		print( self.ofold_output_dir )
		print( os.getcwd() )
		feature_dict_path = glob.glob( f"{self.ofold_output_dir}/predictions/*feature_dict.pkl" )
		print( feature_dict_path )
		if len( feature_dict_path ) == 0:
			raise FileNotFoundError( f"Incorrect path -- {feature_dict_path}..." )

		with open( feature_dict_path[0], "rb" ) as f:
			self.feature_dict = pkl.load( f )



	def get_system_embeddings( self ) -> Dict[str, torch.Tensor]:
		"""
		Obtain the MSA, Pair, and Single representation from OpenFold.
		These are stored in a .pkl file in the OpenFold output directory.

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		msa_rep --> MSA representation for the system [n, r, 256].
		pair_rep --> Pair representation for the system [r, r, 128].
		single_rep --> Single representation for the system [r, 384].
		( n --> no. of seq in MSA; r --> no. of residues in system. )
		"""
		print( "\nLoding evoformer output MSA, Pair, Single representations..." )
		pkl_path = glob.glob( f"{self.ofold_output_dir}/predictions/*output_dict.pkl" )
		if len( pkl_path ) == 0:
			raise Exception( f"Incorrect path -- {pkl_path}..." )

		pkl_path = pkl_path[0]

		with open( pkl_path, "rb" ) as f:
			ofold_output = pkl.load( f )

		# msa_rep = ofold_output["msa"]
		pair_rep = ofold_output["pair"]
		single_rep = ofold_output["single"]

		self.evo_output = {
		# "msa": torch.from_numpy( msa_rep ),
		"pair": torch.from_numpy( pair_rep ),
		"single": torch.from_numpy( single_rep )
		}
		# print( f"MSA rep: {msa_rep.shape} \t Pair rep: {pair_rep.shape} \t Single rep: {single_rep.shape}" )
		print( f"Pair rep: {pair_rep.shape} \t Single rep: {single_rep.shape}" )

		# return evo_output


	def add_batch_dim( self ) -> None:
		"""
		Adds a singleton batch dimension to all tensors.

		****
		This is needed because downstream functions (multi_chain_permutation_align)
			assume that the tensors always have a batch dimension (which will be there while training).
		The reasoning to add a batch dim is speculative.
			I assume this because, in openfold.utils.multi_chain_permutation.multi_chain_permutation_align(),
				Line 421: anchor_true_pos = torch.index_select(true_ca_poses[anchor_gt_idx], 1, anchor_gt_residue)
			tries selecting the dim=1 (hardcoded) in the tensor true_ca_poses (shape = [Nres]) which does not exist.
			Going through the code the only reason for this to happen can be the existence of a batch dim, 
				that would exist while training in mini-batches but does not exist in our case.
		****
		"""
		print( "\nAdding singleton batch dim to all tensors..." )

		# def parse_nested_dict( dict_: Dict, action: str ):
		# 	for k in dict_:
		# 		if isinstance( dict_[k], Dict ):
		# 			dict_[k] = parse_nested_dict( dict_[k] )
		# 		else:
		# 			if isinstance( dict_[k], torch.Tensor ):
		# 				if action == "add_dim":
		# 					dict_[k] = dict_[k].unsqueeze( 0 )
		# 				elif action == "add_to_device":
		# 					dict_[k] = dict_[k].to( self.device )
		# 				elif action == "detach":
		# 					dict_[k] = dict_[k].detach()
		# 	return dict_
		
		with torch.no_grad():
			self.system_features = parse_nested_dict( self.system_features, "add_dim" )
			# dtype = torch.int64 is needed for torch.nn.functional.one_hot() in violation_loss calculation.
			self.system_features["residue_index"] = self.system_features["residue_index"].to( torch.int64 )


	def add_to_device( self ):
		"""
		Add all tensors to device.
		"""
		self.evo_output = parse_nested_dict( self.evo_output, "add_to_device", self.device )
		self.system_features = parse_nested_dict( self.system_features, "add_to_device", self.device )


	def remove_from_device( self, outputs: Dict[str, torch.Tensor] ):
		"""
		Add all tensors to device.
		"""
		self.evo_output = parse_nested_dict( self.evo_output, "detach" )
		self.system_features = parse_nested_dict( self.system_features, "detach" )

		outputs = parse_nested_dict( outputs, "detach" )

		return outputs


	def fit( self ) -> None:
		"""
		Fine-tune weights for the structure module for fit to data.
		Get the Evoformer output: MSA, Single, Pair representations.
			MSA representation not required though.
		Add a singleton batch dim.
		Initialize:
			Model
			A SaveModel object to add and save predicted models to a CIF file.
			Optimizer
		Run the for max_epochs.
			Get predicted output.
			Calculate loss and update parameters.
			Save model to a PDB file.
		"""
		t = time.time()
		self.get_system_embeddings()

		# Craete a SaveModel object.
		save_model_obj = SaveModels( title = "2ayo", 
									output_format = "pdb",
									ensemble_dir = self.ensemble_dir )
									# output_path = self.ensemble_file
		# Initialize the System object.
		save_model_obj.initialize_system()

		# Change the no. of blocks in SM.
		self.ofold_config.model.structure_module.no_blocks = self.topology.model.update_params.sm_no_blocks

		# Get model.
		# 	mode and is_multimer can be removed as we plan to stick to multimers only.
		model = get_model( self.topology.model,
							self.system_features, 
							self.ofold_config,
							self.mode, self.is_multimer, 
							self.device )
		
		# Initialize the specified optimizer.
		optimizer = Optimizer( self.topology.optimizer ).forward( model.params() )

		# Initialize gradient scaler for AMP.
		# self.scaler = GradScaler("cuda")

		for epoch in range( self.topology.train.max_epochs ):
			t_start = time.time()
			print( f"\nEpoch: {epoch} --------------------------" )

			self.add_to_device()
			
			batch = copy.deepcopy( self.system_features )   # Just to keep in sync with OpenFold implementation.
			# Separate out the ground truth features - as in OpenFold training_step.
			gt_features = batch.pop( "gt_features", None )

			# with torch.autocast( device_type = self.device, dtype = torch.float16 ):
			outputs, batch = model.predict( self.evo_output, gt_features, batch )

			# Separate out the restraint features.
			restraint_features = batch.pop( "restraint_features", None )

			# This was used in training AF2 to permutes chains in ground truth before calculating the loss
			# 	because the mapping between the predicted and ground-truth will become arbitrary.
			# 	The model cannot be assumed to predict chains in the same order as the ground truth.
			if self.is_multimer and self.topology.train.allow_mcpa:
				# mcpa --> multi chain permutation align
				print( "--> Performing multi-chain permutation alignment..." )
				batch = multi_chain_permutation_align( out = outputs,
														features = batch,
														ground_truth = gt_features )
			else:
				with torch.no_grad():
					labels = split_ground_truth_labels( gt_features )
					batch.update( gt_features )

			unrelaxed_protein  = self.get_protein_object( outputs = outputs )
			self.add_protein_obj_to_stat( model_num = epoch,
											unrelaxed_protein = unrelaxed_protein )

			self.add_model( save_model_obj = save_model_obj,
							unrelaxed_protein = unrelaxed_protein,
							model_num = epoch )

			self.step( outputs, batch, restraint_features, optimizer, epoch )
			t_end = time.time()
			t_ = time.time()
			print( f"Time taken: {( t_end - t_start )}  seconds" )

		self.save_model( save_model = save_model_obj )

		t_ = time.time()
		print( f"\n --> Time taken for fitting: {( t_ - t )}  seconds" )


	def compute_loss( self, out: Dict[str, torch.Tensor], 
						batch: Dict[str, torch.Tensor], 
						restraint_features: Dict 
				) -> Tuple[torch.Tensor, Dict[str, float]]:
		"""
		Compute the loss and return the cumulative loss and a dict containing all loss terms per epoch.

		"""
		cum_loss, losses = self.loss_fn.forward( out, batch, restraint_features )
		# print( losses )

		return cum_loss, losses



	def update_loss_dict( self, losses: Dict[str, torch.Tensor] ):
		"""
		Keep a tab on the loss per epoch for all individual loss 
			terms and the cumulative loss.
		"""
		if "loss" not in self.stats_dict:
			self.stats_dict["loss"] = {k: [] for k in losses.keys()}

		str_ = ""
		for k, v in losses.items():
			v = round( v.item(), self.prec )
			str_ += f"{k}: {v} \t"
			self.stats_dict["loss"][k].append( v )
		print( f"Losses: {str_}" )



	def compute_metrics( self,
						out: Dict[str, torch.Tensor],
						batch: Dict[str, torch.Tensor],
						restraint_features: Dict,
						last_epoch: bool
				) -> Dict[str, torch.Tensor]:
		"""
		Compute all the required metrics.
		"""
		metrics_dict = self.metrics_fn.forward( out, last_epoch )

		return metrics_dict


	def update_metric_dict( self, metrics_dict: Dict[str, float] ):
		"""
		Save per epoch metric values for all individual merics in stats_dict.
		"""
		if "metrics" not in self.stats_dict:
			self.stats_dict["metrics"] = {k: [] for k in metrics_dict.keys()}

		str_ = ""		
		for k, v in metrics_dict.items():
			v = round( v.item(), self.prec )
			str_ += f"{k}: {v} \t"

			self.stats_dict["metrics"][k].append( v )

		print( f"Metrics: {str_}" )



	def step( self,
				outputs: Dict[str, torch.Tensor],
				batch: Dict[str, torch.Tensor],
				restraint_features: Dict,
				optimizer,
				epoch: int ) -> None:
		"""
		Compute the loss for the finetuned output (need to add that yet).
		Keep track of per-epoch final loss and for each individual loss terms.
		Update the parameters.
		"""
		cum_loss, losses = self.compute_loss( outputs, batch, restraint_features )
		self.update_loss_dict( losses )

		if self.topology.train.allow_grad_update:
			optimizer.zero_grad()
			# self.scaler.scale( cum_loss ).backward()
			# self.scaler.step( optimizer )
			# self.scaler.update()
			cum_loss.backward()
			optimizer.step()

		# Detach and unload all tensors from device.
		outputs = self.remove_from_device( outputs )
		
		if epoch == self.topology.train.max_epochs-1:
			last_epoch = True
		else:
			last_epoch = False

		metrics_dict = self.compute_metrics( outputs,
											batch,
											restraint_features,
											last_epoch )
		self.update_metric_dict( metrics_dict )



	def get_protein_object( self, outputs: Dict[str, torch.Tensor]
							) -> protein.Protein:
		"""
		Given the structure module output dict, return an object of class Protein.
		"""
		unrelaxed_protein = prep_protein( 
									outputs = outputs, 
									feature_dict = self.feature_dict, 
                            		feature_processor = self.feature_processor )
		return unrelaxed_protein


	def add_protein_obj_to_stat( self, model_num: int,
								unrelaxed_protein: protein.Protein, ):
		"""
		Store the Protein object to stats_dict.
		"""
		self.stats_dict["protein"][model_num] = unrelaxed_protein


	def store_metrics_metadata( self ):
		"""
		Store metadata for all metrics into the stats_dict.
		"""
		metadata = self.metrics_fn.metric_metadata_dict
		self.stats_dict["metadata"] = metadata



	def relaxation( self,
					unrelaxed_protein: protein.Protein,
					model_num: int ):
		"""
		Perform AMBER relaxation for the predicted structure.
		# Taken from openfold.utils.script_utils.py.
		"""
		# Not making too many changes.
		model_device = self.device
		cif_output = False
		output_directory = self.relax_ensemble_dir
		output_name = f"model_{model_num}"
		config = self.ofold_config
		
		amber_relaxer = relax.AmberRelaxation(
			use_gpu=(model_device != "cpu"),
			**config.relax,
		)

		t = time.perf_counter()
		visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
		if "cuda" in model_device:
			device_no = model_device.split(":")[-1]
			os.environ["CUDA_VISIBLE_DEVICES"] = device_no
		# the struct_str will contain either a PDB-format or a ModelCIF format string
		struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein, cif_output=cif_output)
		os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
		relaxation_time = time.perf_counter() - t

		# logger.info(f"Relaxation time: {relaxation_time}")
		# update_timings({"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json"))

		# Save the relaxed PDB.
		suffix = "_relaxed.pdb"
		if cif_output:
			suffix = "_relaxed.cif"
		relaxed_output_path = os.path.join(
			output_directory, f'{output_name}{suffix}'
		)
		with open(relaxed_output_path, 'w') as fp:
			fp.write(struct_str)

		print( f"Relaxed output written to {relaxed_output_path}..." )
		# logger.info(f"Relaxed output written to {relaxed_output_path}...")



	def add_model( self,
					save_model_obj: SaveModels,
					unrelaxed_protein: protein.Protein,
					model_num: int ) -> None:
		"""
		For pdb: write the model as a pdb string.
		For cif: add the predicted structure as a model to a modelcif object.
		"""
		# save_model_obj.add_to_modelcif( unrelaxed_protein, epoch )
		save_model_obj.add_model( prot = unrelaxed_protein, epoch = model_num )



	def save_model( self, save_model: SaveModels ) -> None:
		"""
		Save to PDB or CIF file.
		"""
		save_model.save( save_model.system, self.ensemble_file )


