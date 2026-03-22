from typing import List, Tuple, Dict, Any
import os, time, copy, random
from collections import ( defaultdict )
import pickle as pkl
import numpy as np
import ml_collections as mlc

import torch

from openfold.config import model_config
from openfold.data import feature_pipeline
from openfold.np import protein

from models.rigid_sampler import PoseSampling, RandomPoseSampling
from models.rigid_body import get_rigid_body
from loss import LossFunction
from metrics import Metrics
from optimizer import Optimizer
from utils.utils import parse_nested_dict
from utils.pdb_utils import ( prep_protein, SaveModels )


class FitToData():
	def __init__( self, sys_name: str,
					topology: mlc.ConfigDict,
					mode: str,
					feature_dict: Dict[str, np.ndarray],
					processed_feature_dict: Dict[str, torch.Tensor],
					init_pred_dict: Dict[str, Any],
					modeling_output_dir: str,
					prec: int,
					seed_worker,
					device: str ):
		self.sys_name = sys_name
		self.topology = topology
		self.mode = mode
		self.prec = prec
		self.device = device
		self.feature_dict = feature_dict
		self.processed_feature_dict = processed_feature_dict
		self.init_pred_dict = init_pred_dict
		self.modeling_output_dir = modeling_output_dir

		self.seed_worker = seed_worker
		# Set the seeds.
		self.seed_worker()

		# Log the required metrics - loss, restraint satisfaction, etc..
		self.stats_dict = {} # defaultdict( dict )


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.create_required_dir()

		self.init_stats_dict()

		# Load OpenFold configs file.
		self.ofold_config = model_config(
			self.topology.system_representation.config_preset,
			long_sequence_inference = self.topology.system_representation.long_sequence_inference,
			use_deepspeed_evoformer_attention = self.topology.system_representation.use_deepspeed_evoformer_attention,
			)
		self.is_multimer = self.topology.system_representation.is_multimer
		self.ofold_config.globals.is_multimer = self.is_multimer
		self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )

		self.fit()

	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create the required file paths.
		"""
		# PDB file contaiing all predicted models.
		self.ensemble_file_prefix = os.path.join( self.modeling_output_dir, f"{self.sys_name}_output_models" )
		# Directory to store each predicted model as separate PDB file.
		self.ensemble_dir = os.path.join( self.modeling_output_dir, f"{self.sys_name}_ensemble" )		


	def create_required_dir( self ):
		"""
		Create the required directories if not already existing.
		"""
		os.makedirs( self.ensemble_dir, exist_ok = True )


	def init_stats_dict( self ):
		"""
		Stores the following data:
			model_id -> list contaiing the model_id for each conformation.
			time_per_frame: time taken for each frame.
			loss: dict containing per-epoch values for all loss terms.
			metrics: dict containing per-epoch values for all metrics.
			metadata: dict to store metadata for the metrics.
			protein: list of Protein objects obtained per epoch.
			transformations: dict to store the rotation and translation predicted per epoch.
		"""
		for k in ["model_id", "time_per_frame"]:
			self.stats_dict[k] = []

		for k in ["loss", "metrics", "metadata", "protein"]:
			self.stats_dict[k] = {}

		self.stats_dict["transformations"] = {
			"rotation": [], "translation": []
		}


	################################################################################
	################################################################################
	def prep_batch( self ) -> Dict[str, Any]:
		"""
		Adds a singleton batch dimension to all tensors.
		Subset the required restraint and ground truth features.

		****
		Not needed anymore.
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
			# Remove all from device for now.
			self.feature_dict = parse_nested_dict( self.feature_dict, "add_dim" )
			self.remove_from_device( self.feature_dict )

			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, action = "to_tensor" )
			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, "add_dim" )
			self.remove_from_device( self.processed_feature_dict )

			self.init_pred_dict = parse_nested_dict( self.init_pred_dict, action = "to_tensor" )
			self.init_pred_dict = parse_nested_dict( self.init_pred_dict, action = "add_dim" )
			self.remove_from_device( self.init_pred_dict )

		# Create a clone that can be modified every frame as specified.
		batch = {}
		for k in ["aatype", "asym_id", "entity_id"]:
			v = self.processed_feature_dict[k]
			batch[k] = v.clone().to( self.device )
		batch["residue_index"] = self.processed_feature_dict["residue_index"].to(
			dtype = torch.float32
			).to( self.device )
		restraint_features = self.processed_feature_dict.pop( "restraint_features" )
		self.add_to_device( restraint_features )
		for k in restraint_features:
			batch[k] = copy.copy( restraint_features[k] )
		del restraint_features
		return batch


	def add_to_device( self, dict_: Dict ):
		"""
		Add all tensors to device.
		Modifies the input dict inplace.
		"""
		dict_ = parse_nested_dict( dict_, "add_to_device", self.device )


	def remove_from_device( self, dict_: Dict ):
		"""
		Detach all tensors from device.
		Modifies the input dict inplace.
		"""
		dict_ = parse_nested_dict( dict_, "detach" )

	################################################################################
	################################################################################
	def init_loss_n_metrics( self, batch: Dict[str, Any] ):
		"""
		Initialize the loss and metrics modules.
		"""
		self.loss_fn = LossFunction( self.topology["loss"], self.device )
		self.metrics_fn = Metrics( self.topology["metrics"], batch )


	def init_coords( self ):
		"""
		Using final_atom_positions From an
			initial predicted structure.
		"""
		print( f"\tInitializing final_atom-positions..." )
		out = {}
		for k in ["final_atom_positions", "final_atom_mask", "asym_id", "plddt"]:
			out[k] = self.init_pred_dict[k].clone().to( self.device )
		return out


	def reinit_coords( self, out: Dict[str, torch.tensor]
		) -> Dict[str, torch.tensor]:
		"""
		Every frame, one can reuse the prediction in the following ways,
			At every frame,
				previous frame.
				initialize to coordinates from the initial structure.
		
		Inputs:
		----------
		out --> dict output from AF2/OpenFold. See self.init_ccords and SystemRepresentation.

		Returns:
		----------
		out --> out dict with final_atom_positions initialized as specified in the topology.
		"""
		print( f"Reinitializing final_atom-positions: reinit_frame = {self.topology.train.reinit_frame}..." )

		if self.topology.train.reinit_frame == "init":
			del out
			out = self.init_coords()
		elif self.topology.train.reinit_frame == "prev_frame":
			# Return the existing final_atom_positions to be used in the next frame.
			self.remove_from_device( out )
		else:
			raise ValueError( f"Incorrect value for reinit_frame. Allowed init/prev_frame..." )

		self.add_to_device( out )
		return out


	def fit( self ):
		"""
		Prepare inputs for pose sampling.
		Run pose sampling.
		"""
		# TODO: fix redundancy in adding/removing from device.
		# Add batch dim.
		batch = self.prep_batch()

		self.init_loss_n_metrics( batch = batch )

		# Craete a SaveModel object.
		save_model_obj = SaveModels( title = self.sys_name, 
									output_format = "pdb",
									ensemble_dir = self.ensemble_dir )
		# Initialize the System object.
		save_model_obj.initialize_system()

		t_start = time.perf_counter()
		# Keep track of moel_id's.
		self.stats_dict["model_id"] = []
		# Note the time per frame.
		self.stats_dict["time_per_frame"] = []

		out = self.init_coords()

		if self.topology.train.sample_random_pose:
			out = self.predict_pose_random(
				out = out,
				batch = batch,
				save_model_obj = save_model_obj )
		else:
			out = self.predict_pose(
				out = out,
				batch = batch,
				save_model_obj = save_model_obj )

		t_end = time.perf_counter()
		print( f"Total Time taken for smapling: {( t_end - t_start )}  seconds" )

		# Save the ensemble on disk.
		self.save_model( save_model = save_model_obj )

	################################################################################
	################################################################################
	def predict_pose( self,
		out: Dict[str, Any],
		batch: Dict[str, Any],
		save_model_obj: SaveModels ) -> Dict[str, Any]:
		"""
		Run pose sampling for N frames.
		For N iterations (frames)
			Run pose sampler
					Define rigid bodies.
					Predict a rigid tranformation using a neural network.
					Apply the rigid transformation.
				Compute loss.
				Backpropagate.
		"""
		print( "\n\033[1mInitiate pose sampling now...\033[0m" )

		# The feature dim equals the length of the length of coarse-grained fixed rigid body.
		model = PoseSampling(
			model_config = self.topology.model,
			device = self.device )
		model.to( self.device )

		# Initialize the specified optimizer.
		optimizer = Optimizer( self.topology.optimizer ).forward( model.params() )
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		# 	optimizer, mode = "min", factor = 0.5, patience = 500, min_lr = 1e-6)
		# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		# 	optimizer, T_max = self.topology.train.num_frames, eta_min = 1e-6 )

		for frame in range( self.topology.train.num_frames ):
			t_s = time.perf_counter()
			print( f"\nPose sampling frame: {frame} --------------------------" )
			out = self.reinit_coords( out = out )

			out = model.predict( out = out, batch = batch )

			cum_loss, losses = self.loss_fn.forward( out,
				batch )

			optimizer.zero_grad()
			cum_loss.backward()
			# Gradient clipping.
			if self.topology.train.max_norm is not None:
				torch.nn.utils.clip_grad_norm_(
					model.params()[0].parameters(),
					max_norm = self.topology.train.max_norm
					)
			optimizer.step()
			# scheduler.step( losses["unscaled_loss"] )

			if frame == self.topology.train.num_frames-1:
				last_frame = True
			else:
				last_frame = False

			metrics_dict = self.metrics_fn.forward(
				out = out,
				last_epoch = last_frame )

			self.update_loss_dict( losses )
			self.update_grad_norm( model = model )

			self.update_data_metric_dict(
				metrics_dict = metrics_dict )

			self.remove_from_device( out )
			# # Remove computed violations.
			# if "violation" in out:
			# 	out.pop( "violation" )

			# Add predicted model to the ensemble.
			unrelaxed_protein  = self.get_protein_object( outputs = out )
			self.add_protein_obj_to_stat( model_id = frame,
											unrelaxed_protein = unrelaxed_protein )

			self.add_model( save_model_obj = save_model_obj,
							unrelaxed_protein = unrelaxed_protein,
							model_id = frame )

			# using frame as the model_id.
			self.stats_dict["model_id"].append( frame )

			t_e = time.perf_counter()
			print( f"Time taken for Frame {frame}: {( t_e - t_s )}  seconds" )
			self.stats_dict["time_per_frame"].append( t_e - t_s )
			print( "-"*80 + "\n" + "-"*80 )

		# Log all predicted rigid transformations.
		self.store_transformations( transformations_dict = model.transformations )

		return out


	def predict_pose_random( self,
		out: Dict[str, Any],
		batch: Dict[str, Any],
		save_model_obj: SaveModels
		) -> Dict[str, Any]:
		"""
		Predict rigid transformations at random.
		Similar to predict_pose() above.
		"""
		print( "\n\033[1mInitiate random pose sampling now...\033[0m" )
		model = RandomPoseSampling(
			model_config = self.topology.model,
			device = self.device )

		for frame in range( self.topology.train.num_frames ):
			t_s = time.perf_counter()
			print( f"\nPose sampling frame: {frame} --------------------------" )

			out = self.reinit_coords( out = out )

			out = model.predict( out = out )

			cum_loss, losses = self.loss_fn.forward( out,
				# self.gt_feature_dict,
				batch )
			self.update_loss_dict( losses )

			if frame == self.topology.train.num_frames-1:
				last_frame = True
			else:
				last_frame = False
			metrics_dict = self.metrics_fn.forward(
				out = out, last_epoch = last_frame )
			self.update_data_metric_dict(
				metrics_dict = metrics_dict )

			# Remove computed violations.
			if "violation" in out:
				out.pop( "violation" )
			self.remove_from_device( out )

			# Add predicted model to the ensemble.
			unrelaxed_protein  = self.get_protein_object( outputs = out )
			self.add_protein_obj_to_stat( model_id = frame,
											unrelaxed_protein = unrelaxed_protein )

			self.add_model( save_model_obj = save_model_obj,
							unrelaxed_protein = unrelaxed_protein,
							model_id = frame )

			# using frame as the model_id.
			self.stats_dict["model_id"].append( frame )

			t_e = time.perf_counter()
			print( f"Time taken for Frame {frame}: {( t_e - t_s )}  seconds" )
			self.stats_dict["time_per_frame"].append( t_e - t_s )
			print( "-"*80 + "\n" + "-"*80 )

		# Log all predicted rigid transformations.
		self.store_transformations( transformations_dict = model.transformations )

		self.add_to_device( out )
		return out

	################################################################################
	################################################################################
	def update_grad_norm( self, model: PoseSampling ):
		"""
		Keep a tab on the gradient norm.
		"""
		total_norm = 0.0
		for module in model.params():
			for p in module.parameters():
				if p.grad is not None:
					total_norm += p.grad.data.norm( 2 ).item() ** 2
		total_norm = total_norm ** 0.5

		# Given "loss" is updated first, so the key exists.
		if not "grad_norm" in self.stats_dict["loss"]:
			self.stats_dict["loss"]["grad_norm"] = [total_norm]
		else:
			self.stats_dict["loss"]["grad_norm"].append( total_norm )
		print( f"Grad norm: {total_norm:.3f}" )


	def update_loss_dict( self, losses: Dict[str, torch.Tensor] ):
		"""
		Keep a tab on the loss per frame/step for all individual loss 
			terms and the cumulative loss.
		||g|| = sqrt( sum_i L2norm( g_i ) )
			g -> gradient norm; g_i -> gradient for each parameter.
		"""
		if self.stats_dict["loss"] == {}:
			self.stats_dict["loss"] = {k: [] for k in losses.keys()}

		str_ = ""
		for k, v in losses.items():
			v = round( v.item(), self.prec )
			str_ += f"{k}: {v} \t"
			self.stats_dict["loss"][k].append( v )

		print( f"Losses: {str_}" )


	def update_data_metric_dict( self,
		metrics_dict: Dict[str, float] ):
		"""
		Save per frame/step metric values for all individual merics in stats_dict.
		"""
		if self.stats_dict["metrics"] == {}:
			self.stats_dict["metrics"] = {k: [] for k in metrics_dict.keys()}

		str_ = ""
		for k, v in metrics_dict.items():
			v = round( v.item(), self.prec )
			str_ += f"{k}: {v} \t"

			self.stats_dict["metrics"][k].append( v )

		print( f"Metrics: {str_}" )


	def store_transformations( self, transformations_dict: Dict[str, List] ):
		"""
		Store all predicted rigid transformations in the stats_dict every frame.
		"""
		if "transformations" not in self.stats_dict:
			self.stats_dict["transformations"] = {
				"rotation": [], "translation": []
			}

		# For M pose sampling steps -> [M, B, 4].
		rot = torch.stack( transformations_dict["rotation"] ).cpu().numpy()
		# For M pose sampling steps -> [M, B, 3].
		trans = torch.stack( transformations_dict["translation"] ).cpu().numpy()

		self.stats_dict["transformations"]["rotation"].append( rot )
		self.stats_dict["transformations"]["translation"].append( trans )

	################################################################################
	################################################################################
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
		Not storing metadata for pose sampling.
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
		# save_model_obj.add_to_modelcif( unrelaxed_protein, frame )
		save_model_obj.add_model( prot = unrelaxed_protein, model_id = model_id )


	def save_model( self, save_model: SaveModels ) -> None:
		"""
		Save to PDB or CIF file.
		"""
		save_model.save( save_model.system, self.ensemble_file_prefix )

