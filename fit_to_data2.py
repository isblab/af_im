from typing import List, Tuple, Dict, Any
import os, glob, time, copy
from collections import ( defaultdict )
import pickle as pkl
import numpy as np
import ml_collections as mlc

import torch

from openfold.data import feature_pipeline
from openfold.np import protein

from models.rigid_sampler import PoseSampling
from models.recycler import Recycler

from loss import LossFunction
from metrics import Metrics
from optimizer import Optimizer
from utils.utils import parse_nested_dict
from utils.pdb_utils import ( prep_protein, SaveModels )


class FitToData():
	def __init__( self, sys_name: str,
					ofold_config: mlc.ConfigDict,
					topology: mlc.ConfigDict,
					mode: str,
					jax_params_path: str,
					feature_dict: Dict[str, Any],
					processed_feature_dict: Dict[str, Any],
					init_pred_dict: Dict[str, Any],
					#ofold_output_dir: str,
					modeling_output_dir: str,
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
		self.jax_params_path = jax_params_path
		self.feature_dict = feature_dict
		self.processed_feature_dict = processed_feature_dict
		self.init_pred_dict = init_pred_dict
		#self.ofold_output_dir = ofold_output_dir
		self.modeling_output_dir = modeling_output_dir

		# Set the seeds.
		seed_worker()

		# Stats for the full run (pose sampling + recycling).
		self.stats_dict = defaultdict( dict )
		self.loss_fn = LossFunction( self.topology["loss"], self.device )
		self.metrics_fn = Metrics( self.topology["metrics"], self.processed_feature_dict["restraint_features"] )

		# Stats for pose sampling.
		self.stats_dict_pose = defaultdict( dict )
		self.loss_fn_pose = LossFunction( self.topology["loss"], self.device )
		self.metrics_fn_pose = Metrics( self.topology["metrics"], self.processed_feature_dict["restraint_features"] )


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.create_required_dir()

		self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )

		self.fit()


	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		# PDB file contaiing all predicted models.
		self.ensemble_file = os.path.join( self.modeling_output_dir, f"{self.sys_name}_output_models" )
		# Directory to store each predicted model as separate PDB file.
		self.ensemble_dir = os.path.join( self.modeling_output_dir, f"{self.sys_name}_ensemble" )		


	def create_required_dir( self ):
		"""
		Create the required directories if not already existing.
		"""
		os.makedirs( self.ensemble_dir, exist_ok = True )


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
		
		with torch.no_grad():
			self.feature_dict = parse_nested_dict( self.feature_dict, "add_dim" )
			# dtype = torch.int64 is needed for torch.nn.functional.one_hot() in violation_loss calculation.
			self.feature_dict["residue_index"] = self.feature_dict["residue_index"] # .to( torch.int64 )

			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, action = "to_tensor" )
			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, "add_dim" )

			self.init_pred_dict = parse_nested_dict( self.init_pred_dict, action = "to_tensor" )
			# self.init_pred_dict = parse_nested_dict( self.init_pred_dict, "add_dim" )

			for k in self.init_pred_dict:
				if k != "sm":
					self.init_pred_dict[k] = self.init_pred_dict[k].unsqueeze( 0 )
				else:
					for m in self.init_pred_dict["sm"]:
						self.init_pred_dict[k][m] = self.init_pred_dict[k][m].unsqueeze( 1 )


	def add_to_device( self, dict_: Dict ):
		"""
		Add all tensors to device.
		"""
		dict_ = parse_nested_dict( dict_, "add_to_device", self.device )


	def remove_from_device( self, dict_: Dict ):
		"""
		Add all tensors to device.
		"""
		dict_ = parse_nested_dict( dict_, "detach" )


	def load_openfold_configs( self ):
		"""
		Load the OpenFold configs.
		Overwrite the configs with those mentioned in topology file.
		"""
		# Change the no. of blocks in SM.
		self.ofold_config.model.structure_module.no_blocks = self.topology.model.update_params.sm_no_blocks
		# Enable long sequence inference.


	def fit( self ) -> None:
		"""
		Add a singleton batch dim.
		Initialize:
			Models (PoseSampling and Alphafold)
			A SaveModel object to add and save predicted models to a CIF file.
			Optimizer

		For each epoch:
			Perform M rounds of rigid sampling.
				Define rigid bodies (either cy chain or based on pLDDT and/or PAE).
				Predict and apply a rigid transformation.
				Compute loss and backpropagate.
			Recycle final structure as input to AlphaFold (no_grad).
				This gives the predicted structure per epoch.
			Compute the metrics.
			Save model to a PDB file.
			Save all required metadata in stats_dict.
				Protein object.
				pLDDT, PAE, pTM, ipTM, distogram_logits
		"""
		self.stats_dict["model_id"] = []

		# Add batch dim.
		self.add_batch_dim()

		# Craete a SaveModel object.
		save_model_obj = SaveModels( title = self.sys_name, 
									output_format = "pdb",
									ensemble_dir = self.ensemble_dir )

		# Initialize the System object.
		save_model_obj.initialize_system()

		batch = {}
		
		# Remove the restraint features.
		for k in self.processed_feature_dict:
			if k not in ["restraint_features", "connectivity_mask", "intra_ev_mask", "inter_ev_mask"]:
				batch[k] = self.processed_feature_dict[k]
				if isinstance( self.processed_feature_dict[k].dtype, float ):
					batch[k] = self.batch[k].to( dtype = torch.float32 )
		self.add_to_device( batch )

		# Initialize the recycling model.
		recycler_model = Recycler(
			ofold_config = self.ofold_config,
			jax_param_path = self.jax_params_path,
			device = self.device )

		t_start = time.perf_counter()
		for epoch in range( self.topology.train.max_epochs ):
			# Note time for full run (pose sampling+recycling).
			t_s = time.perf_counter()
			print( f"\n\033[1mEpoch: {epoch} \033[0m" + "-"*20 )

			#torch.cuda.reset_peak_memory_stats()

			# Skip pose sampling if specified.
			if self.topology.train.skip_pose_sampling:
				out = self.predict_pose()

			torch.cuda.empty_cache()
			# Note time taken by recycling alone.
			t_r_s = time.perf_counter()

			print( "\n\033[1mRecycling optimized pose...\033[0m" )
			with torch.no_grad():
				outputs = recycler_model.forward(
					out = out,
					batch = batch )
			t_r_e = time.perf_counter()
			print( f"Time taken for recycling {epoch}: {( t_r_e - t_r_s )} seconds" )

			self.remove_from_device( outputs )
			out = copy.deepcopy( outputs )
			del outputs
			torch.cuda.empty_cache()

			if epoch == self.topology.train.max_epochs-1:
				last_epoch = True
			else:
				last_epoch = False

			# Just compute loss but don't backpropagate.
			_, losses = self.loss_fn.forward( out,
				self.processed_feature_dict,
				self.processed_feature_dict["restraint_features"] )

			self.update_loss_dict( losses, update_pose_metrics = False )
			metrics_dict = self.metrics_fn.forward(
				out = out, last_epoch = last_epoch,
				update_pose_metrics = False )
			self.update_metric_dict( metrics_dict )

			# Add predicted model to the ensemble.
			unrelaxed_protein  = self.get_protein_object( outputs = out )
			self.add_protein_obj_to_stat( model_id = epoch,
											unrelaxed_protein = unrelaxed_protein )

			self.add_model( save_model_obj = save_model_obj,
							unrelaxed_protein = unrelaxed_protein,
							model_id = epoch )

			# Keep track of the no. of epochs.
			self.stats_dict["model_id"].append( epoch )
			t_e = time.perf_counter()
			print( f"Time taken for epoch {epoch}: {( t_e - t_s )}  seconds" )
			print( "-"*80 + "\n" + "-"*80 )

		t_end = time.perf_counter()
		print( f"Total Time taken for smapling: {( t_end - t_start )}  seconds" )
		self.save_model( save_model = save_model_obj )


	def predict_pose( self ):
		"""
		For M iterations
			Define rigid bodies (either cy chain or based on pLDDT and/or PAE).
			Predict a rigid tranformation using a neural network.
			Apply the rigid transformation.
			Compute loss.
			Backpropagate.
		"""
		# Get model.
		# 	mode and is_multimer can be removed as we plan to stick to multimers only.
		print( "\n\033[1mInitiate pose sampling now...\033[0m" )
		model = PoseSampling(
			model_config = self.topology.model,
			device = self.device )
		model.to( self.device )

		# Initialize the specified optimizer.
		optimizer = Optimizer( self.topology.optimizer ).forward( model.params() )

		# smo = SaveModels( title = self.sys_name, 
		#							output_format = "pdb",
		#							ensemble_dir = "",
		#							 save_single_model = False )

		# Initialize the System object.
		# smo.initialize_system()

		#u = self.get_protein_object( self.init_pred_dict )
		#smo.add_model( prot = u, model_id = 100 )

		for sub_epoch in range( self.topology.train.max_pose_iters ):
			print( f"\nPose sampling epoch: {sub_epoch} --------------------------" )

			out = copy.deepcopy( self.init_pred_dict )
			self.add_to_device( out )
			with torch.autograd.detect_anomaly():
				out = model.predict( out = out )

				#u = self.get_protein_object( out )

				cum_loss, losses = self.loss_fn.forward( out,
					self.processed_feature_dict,
					self.processed_feature_dict["restraint_features"] )
				self.update_loss_dict( losses, update_pose_metrics = True )

				if sub_epoch == self.topology.train.max_pose_iters-1:
					last_epoch = True
				else:
					last_epoch = False
				metrics_dict = self.metrics_fn.forward(
					out = out, last_epoch = last_epoch,
					update_pose_metrics = False )
				self.update_metric_dict( metrics_dict )

				# smo.add_model( prot = u, model_id = sub_epoch )
				optimizer.zero_grad()
				cum_loss.backward()
				#for name, param in model.named_parameters():
				#	if param.grad is not None:
				#		print( f"{name} grad stats: min = {param.grad.min()}, max = {param.grad.max()}, nan = {torch.isnan( param.grad ).any()}" )
				optimizer.step()

		#smo.save( smo.system, "./dummy" )
		return out


	def update_loss_dict( self, losses: Dict[str, torch.Tensor], update_pose_metrics: bool ):
		"""
		Keep a tab on the loss per epoch for all individual loss 
			terms and the cumulative loss.
		"""
		if update_pose_metrics:
			if "loss" not in self.stats_dict_pose:
				self.stats_dict_pose["loss"] = {k: [] for k in losses.keys()}

			str_ = ""
			for k, v in losses.items():
				v = round( v.item(), self.prec )
				str_ += f"{k}: {v} \t"
				self.stats_dict_pose["loss"][k].append( v )
		else:
			if "loss" not in self.stats_dict:
				self.stats_dict["loss"] = {k: [] for k in losses.keys()}

			str_ = ""
			for k, v in losses.items():
				v = round( v.item(), self.prec )
				str_ += f"{k}: {v} \t"
				self.stats_dict["loss"][k].append( v )

		print( f"Losses: {str_}" )


	def update_metric_dict( self, metrics_dict: Dict[str, float], update_pose_metrics: bool ):
		"""
		Save per epoch metric values for all individual merics in stats_dict.
		"""
		if update_pose_metrics:
			if "metrics" not in self.stats_dict_pose:
				self.stats_dict_pose["metrics"] = {k: [] for k in metrics_dict.keys()}

			str_ = ""		
			for k, v in metrics_dict.items():
				v = round( v.item(), self.prec )
				str_ += f"{k}: {v} \t"

				self.stats_dict_pose["metrics"][k].append( v )
		else:
			if "metrics" not in self.stats_dict:
				self.stats_dict["metrics"] = {k: [] for k in metrics_dict.keys()}

			str_ = ""		
			for k, v in metrics_dict.items():
				v = round( v.item(), self.prec )
				str_ += f"{k}: {v} \t"

				self.stats_dict["metrics"][k].append( v )


		print( f"Metrics: {str_}" )


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


	def add_protein_obj_to_stat( self, model_id: int,
								unrelaxed_protein: protein.Protein, ):
		"""
		Store the Protein object to stats_dict.
		"""
		self.stats_dict["protein"][model_id] = unrelaxed_protein


	def store_metrics_metadata( self ):
		"""
		Store metadata for all metrics into the stats_dict.
		"""
		metadata = self.metrics_fn.metric_metadata_dict
		self.stats_dict["metadata"] = metadata


	def add_model( self,
					save_model_obj: SaveModels,
					unrelaxed_protein: protein.Protein,
					model_id: int ) -> None:
		"""
		For pdb: write the model as a pdb string.
		For cif: add the predicted structure as a model to a modelcif object.
		"""
		# save_model_obj.add_to_modelcif( unrelaxed_protein, epoch )
		save_model_obj.add_model( prot = unrelaxed_protein, model_id = model_id )



	def save_model( self, save_model: SaveModels ) -> None:
		"""
		Save to PDB or CIF file.
		"""
		save_model.save( save_model.system, self.ensemble_file )

