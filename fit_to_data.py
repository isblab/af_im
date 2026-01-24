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
from models.recycler import Recycler
from models.af_sampling import (
	msa_subsampler, msa_column_masking,
	mask_msa_for_xl_res, noised_structure )
from models.custom_template_feats import update_template_feats

from loss import LossFunction
from metrics import Metrics
from optimizer import Optimizer
from utils.utils import parse_nested_dict
from utils.pdb_utils import ( prep_protein, SaveModels )


class FitToData():
	def __init__( self, sys_name: str,
					topology: mlc.ConfigDict,
					mode: str,
					jax_params_path: str,
					feature_dict: Dict[str, np.ndarray],
					processed_feature_dict: Dict[str, torch.Tensor],
					gt_feature_dict: Dict[str, torch.Tensor],
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
		self.jax_params_path = jax_params_path
		self.feature_dict = feature_dict
		self.processed_feature_dict = processed_feature_dict
		self.gt_feature_dict = gt_feature_dict
		self.init_pred_dict = init_pred_dict
		self.modeling_output_dir = modeling_output_dir

		self.seed_worker = seed_worker
		# Set the seeds.
		self.seed_worker()

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
		self.ensemble_file = os.path.join( self.modeling_output_dir, f"{self.sys_name}_output_models" )
		# Directory to store each predicted model as separate PDB file.
		self.ensemble_dir = os.path.join( self.modeling_output_dir, f"{self.sys_name}_ensemble" )		


	def create_required_dir( self ):
		"""
		Create the required directories if not already existing.
		"""
		os.makedirs( self.ensemble_dir, exist_ok = True )

	################################################################################
	################################################################################
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

			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, action = "to_tensor" )
			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, "add_dim" )

			self.gt_feature_dict = parse_nested_dict( self.gt_feature_dict, action = "to_tensor" )
			self.gt_feature_dict = parse_nested_dict( self.gt_feature_dict, action = "add_dim" )
			self.gt_feature_dict["residue_index"] = self.gt_feature_dict["residue_index"].to( torch.int64 )

			self.init_pred_dict = parse_nested_dict( self.init_pred_dict, action = "to_tensor" )

			for k in self.init_pred_dict:
				if k != "sm":
					self.init_pred_dict[k] = self.init_pred_dict[k].unsqueeze( 0 )
				else:
					# For sm output, 0th dim is the SM recycling dim.
					for m in self.init_pred_dict["sm"]:
						self.init_pred_dict["sm"][m] = self.init_pred_dict["sm"][m].unsqueeze( 1 )


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

	################################################################################
	################################################################################
	def init_representations( self ):
		"""
		Initialize the MSA and Pair representation as specified in the topology file.
		init_rep
			init: same reuse the MSA and Pair rep from the initial predicted structure.
			zero: initialize both to 0's.
			none: initialize both to None. This causes the
				recycling embedder to ignore the x_prev (optimized pose to be used).
		"""
		# n = self.processed_feature_dict["asym_id"].shape[1]
		# msa -> [1, S, N]
		_, s, n = self.processed_feature_dict["msa"].shape
		# MSA and Pair representation channel dim.
		c_m, c_z = 256, 128

		print( f"\tItializing MSA and pair representations: init_rep = {self.topology.train.init_rep}..." )
		if "none" in self.topology.train.init_rep:
			# When None, the optimized structure being input to the recycler is ignored.
			msa, pair = None, None
		else:
			for i, init_ in enumerate( self.topology.train.init_rep ):
				if init_ == "init":
					if i == 0:
						msa = self.init_pred_dict["msa"].clone().to( self.device )
					else:
						pair = self.init_pred_dict["pair"].clone().to( self.device )
				elif init_ == "zero":
					if i == 0:
						msa = torch.zeros( [1, s, n, c_m] ).to( self.device )
					else:
						pair = torch.zeros( [1, n, n, c_z] ).to( self.device )
				else:
					raise ValueError( f"Incorrect value for init_rep. Allowed init/zero/none..." )

		return {"msa": msa, "pair": pair}


	def init_coords( self ):
		"""
		final_atom_positions can be obtained
			From an initial predicted structure.
			An all-0's tensor.
		"""
		n = self.processed_feature_dict["asym_id"].shape[1]

		print( f"\tItializing final_atom-positions: init_coord = {self.topology.train.init_coord}..." )
		# Skip using the initial predicted structure.
		if self.topology.train.init_coord == "zero":
			final_atom_positions = torch.zeros( [1, n, 37, 3] )
			plddt = torch.zeros( [1, n] )
			out = {
				"final_atom_positions": final_atom_positions.to( self.device ),
				# These two are already on device.
				"final_atom_mask": self.processed_feature_dict["atom37_atom_exists"],
				"asym_id": self.processed_feature_dict["asym_id"],
				"plddt": plddt.to( self.device )
			}
		elif self.topology.train.init_coord == "init":
			out = {}
			for k in ["final_atom_positions", "final_atom_mask", "asym_id", "plddt"]:
				out[k] = self.init_pred_dict[k].clone().to( self.device )
			# out = copy.deepcopy( self.init_pred_dict )
		else:
			raise ValueError( f"Incorrect value for init_coord. Allowed zero/init..." )
		# Add the initialized MSA and Pair representations.
		out.update( self.init_representations() )

		# self.add_to_device( out )
		return out


	def reinit_rep( self,
		prev_frame_msa: torch.Tensor,
		prev_frame_pair: torch.Tensor ) -> Dict[str, torch.Tensor]:
		"""
		Every frame, one can reuse the MSA and Pair representtaions in the following ways:
			Use representations from the previous frame.
			Initialize again as specified in init_rep.
		"""
		if "init" in self.topology.train.reinit_rep:
			rep = self.init_representations()
		reinit_msa, reinit_pair = self.topology.train.reinit_rep

		if reinit_msa == "init":
			msa = rep["msa"]
		elif reinit_msa == "prev_frame":
			print( f"Reusing MSA rep from previous frame..." )
			msa = prev_frame_msa
		else:
			raise ValueError( f"Incorrect value specified for reinit_rep -> msa - {reinit_msa}..." )

		if reinit_pair == "init":
			pair = rep["pair"]
		elif reinit_pair == "prev_frame":
			print( f"Reusing Pair rep from previous frame..." )
			pair = prev_frame_pair
		else:
			raise ValueError( f"Incorrect value specified for reinit_rep -> pair - {reinit_pair}..." )
		return {"msa": msa, "pair": pair}


	def reinit_coords( self,
		out: torch.tensor,
		prev_frame_coord: torch.tensor,
		pose_iter: bool
		) -> Dict[str, torch.tensor]:
		"""
		Every frame, one can reuse the prediction in the following ways,
			At every frame,
				previous frame.
				initialize again as specified by init_coord.
			At every pose sampling iteration (step), one can reuse prediction from,
				previous frame.
				previous pose sampling iteration (step).
				initialize again as specified by init_coord.
		
		Inputs:
		----------
		out --> dict output from AF2/OpenFold. See SystemRepresentation for details.
		prev_frame_coord --> final_atom_positions from the previous, (n-1)th, frame.
		pose_iter --> boolean flag to distinguish between the a frame and a step iteration.

		Returns:
		----------
		out --> out dict with final_atom_positions initialized as specified in the topology.
		"""
		# For pose sampling.
		if pose_iter:
			print( f"Reinitializing final_atom-positions: reinit_pose = {self.topology.train.reinit_step}..." )
			if self.topology.train.reinit_step == "prev_frame":
				del out["final_atom_positions"]
				out["final_atom_positions"] = prev_frame_coord
			elif self.topology.train.reinit_step == "prev_step":
				# Return the existing final_atom_positions to be used in the next step.
				pass
			elif self.topology.train.reinit_step == "init":
				out = self.init_coords()
			else:
				raise ValueError( f"Incorrect value for reinit_pose. Allowed prev_frame/prev_step/init..." )
		else:
			print( f"Reinitializing final_atom-positions: reinit_frame = {self.topology.train.reinit_frame}..." )
			if self.topology.train.reinit_frame == "init":
				del out
				out = self.init_coords()
			elif self.topology.train.reinit_frame == "prev_frame":
				# Return the existing final_atom_positions to be used in the next frame.
				pass
			else:
				raise ValueError( f"Incorrect value for reinit_frame. Allowed init/prev_frame..." )

		return out

	def initialize_openfold_model( self ):
		"""
		Initialize pre-trained OpenFold model.
		"""
		# Initialize the OpenFold model predict structures biased by the pose sampled conformation.
		recycler_model = Recycler(
			ofold_config = self.ofold_config,
			jax_param_path = self.jax_params_path,
			num_iters = self.topology.model.num_recycles,
			inference_mode = self.topology.model.inference_mode,
			activate_dropouts = self.topology.model.activate_dropouts,
			device = self.device )

		# Toggle template embedder ON/OFF.
		use_template_embedder = self.topology.model.use_template_embedder
		self.ofold_config.model.template.enabled = use_template_embedder
		if use_template_embedder:
			print( "Template embedder turned ON..." )
		else:
			print( "Template embedder turned OFF..." )
			if self.topology.train.use_as_templates:
				raise RuntimeError(
					f"Cannot inject templates because the template embedder is disabled..." )

		# Toggle extra MSA embedder ON/OFF.
		use_extra_msa = self.topology.model.use_extra_msa
		self.ofold_config.model.template.enabled = use_extra_msa
		if use_extra_msa:
			print( "Extra MSA embedder turned ON..." )
		else:
			print( "Extra MSA embedder turned OFF..." )
		return recycler_model


	def fit( self ) -> None:
		"""
		Each frame of the simulation comprises
			1 cycle of Pose sampling
				M steps
			1 round of OpenFold prediction

		Add a singleton batch dim.
		Initialize:
			A SaveModel object to add and save predicted models to a CIF file.
			Recycler model.

		For each frame (epoch):
			Perform M rounds of rigid sampling (one cycle of M steps).
				Define rigid bodies (either chain or based on pLDDT and/or PAE).
				Predict and apply a rigid transformation.
				Compute loss and backpropagate.
			Recycle final structure as input to AlphaFold (no_grad).
				This gives the predicted structure per frame.
			Compute the metrics.
			Save model to a PDB file.
			Save all required metadata in stats_dict.
				Protein object.
				pLDDT, PAE, pTM, ipTM, distogram_logits
		"""
		# Add batch dim.
		self.add_batch_dim()

		# Craete a SaveModel object.
		save_model_obj = SaveModels( title = self.sys_name, 
									output_format = "pdb",
									ensemble_dir = self.ensemble_dir )
		# Initialize the System object.
		save_model_obj.initialize_system()

		recycler_model = self.initialize_openfold_model()

		self.add_to_device( self.processed_feature_dict )
		restraint_features = self.processed_feature_dict.pop( "restraint_features" )
		# Add gt_features to device.
		self.add_to_device( self.gt_feature_dict )

		# Create a clone that can be modified every frame as specified.
		batch = {}
		for k in self.processed_feature_dict:
			v = self.processed_feature_dict[k]
			batch[k] = v.clone()
			if torch.is_tensor( v ) and torch.is_floating_point( v ):
				batch[k] = batch[k].to( dtype = torch.float32 )

		t_start = time.perf_counter()

		# Keep track of moel_id's.
		self.stats_dict["model_id"] = []
		# Note the time per frame.
		self.stats_dict["time_per_frame"] = []

		out = self.init_coords()
		prev_frame_coord = out["final_atom_positions"]
		prev_frame_msa = out["msa"]
		prev_frame_pair = out["pair"]

		for frame in range( self.topology.train.num_frames ):
			# Note time for a frame (pose sampling+recycling).
			t_s = time.perf_counter()
			print( f"\n\033[1mFrame: {frame} \033[0m" + "-"*20 )

			# Skip pose sampling if specified.
			if not self.topology.train.skip_pose_sampling:
				# Skip pose sampling for 0th frame.
				if frame == 0 and self.topology.train.init_coord == "zero":
					print( f"init_coord = {self.topology.train.init_coord}. Skip pose sampling for frame 0..." )
				else:
					# rng_state = torch.random.get_rng_state()
					if self.topology.train.sample_random_pose:
						out = self.predict_pose_random(
							out = out,
							restraint_features = restraint_features,
							prev_frame_coord = prev_frame_coord )
					else:
						out = self.predict_pose(
							out = out,
							restraint_features = restraint_features,
							prev_frame_coord = prev_frame_coord )
					# torch.random.set_rng_state( rng_state )
			else:
				pass

			out, batch = self.prepare_pose_for_injection(
				out = out,
				batch = batch,
				frame = frame )

			# Note time taken by recycling alone.
			t_r_s = time.perf_counter()
			print( "\n\033[1mRecycling optimized pose...\033[0m" )
			with torch.no_grad():
				out, batch = self.af_sampling( out = out, batch = batch )

				outputs = recycler_model.forward(
					out = out,
					batch = batch )

				out = {k:v.clone() if torch.is_tensor(v) else v for k, v in outputs.items()}
				del outputs
				# Just compute loss but don't backpropagate.
				_, losses = self.loss_fn.forward( out,
					self.gt_feature_dict,
					restraint_features )
				# Remove computed violations.
				if "violation" in out:
					del out["violation"]

			t_r_e = time.perf_counter()
			print( f"Time taken for OpenFold prediction at frame {frame}: {( t_r_e - t_r_s )} seconds" )

			# Clear cache.
			torch.cuda.empty_cache()

			if frame == self.topology.train.num_frames-1:
				last_frame = True
			else:
				last_frame = False

			# Log the required loss and metrics.
			self.update_loss_dict( losses, update_pose_metrics = False )
			metrics_dict = self.metrics_fn.forward(
				out = out, last_epoch = last_frame )
			self.update_data_metric_dict(
				metrics_dict = metrics_dict, update_pose_metrics = False )
			self.update_confidence_metrics(
				out = out )

			# Add predicted model to the ensemble.
			unrelaxed_protein  = self.get_protein_object( outputs = out )
			self.add_protein_obj_to_stat( model_id = frame,
											unrelaxed_protein = unrelaxed_protein )

			self.add_model( save_model_obj = save_model_obj,
							unrelaxed_protein = unrelaxed_protein,
							model_id = frame )

			# Current frame coordinates for use in the nest frame if specified.
			prev_frame_coord = out["final_atom_positions"]
			prev_frame_msa = out["msa"]
			prev_frame_pair = out["pair"]
			# Reinitialize out as specified in topology.
			out = self.reinit_coords(
				out = out,
				prev_frame_coord = prev_frame_coord,
				pose_iter = False )
			out.update( self.reinit_rep(
					prev_frame_msa = prev_frame_msa,
					prev_frame_pair = prev_frame_pair )
				)

			# using frame as the model_id.
			self.stats_dict["model_id"].append( frame )
			# Removed the modified MSA features.
			for k in ["msa", "msa_feat", "msa_mask", "deletion_matrix", "cluster_deletion_mean", "cluster_profile"]:
				del batch[k]
				batch[k] = self.processed_feature_dict[k].clone().to( self.device )
			if self.topology.model.extra_msa_subsampling.enabled:
				# Remove the modified extra MSA features.
				for k in ["extra_msa", "extra_deletion_matrix", "extra_msa_mask"]:
					del batch[k]
					batch[k] = self.processed_feature_dict[k].clone().to( self.device )
			if self.topology.train.add_to_existing_templates:
				for k in ["template_aatype", "template_all_atom_positions", "template_all_atom_mask"]:
					del batch[k]
					batch[k] = self.processed_feature_dict[k].clone().to( self.device )

			t_e = time.perf_counter()
			print( f"Time taken for Frame {frame}: {( t_e - t_s )}  seconds" )
			self.stats_dict["time_per_frame"].append( t_e - t_s )
			print( "-"*80 + "\n" + "-"*80 )

		t_end = time.perf_counter()
		print( f"Total Time taken for smapling: {( t_end - t_start )}  seconds" )

		# Save the ensemble on disk.
		self.save_model( save_model = save_model_obj )


	def prepare_pose_for_injection( self,
		out: Dict[str, Any],
		batch: Dict[str, torch.Tensor],
		frame: int
		) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
		"""
		Modify the batch and out dicts for biaisng OpenFold prediction
			with the pose sampled structure.
		The pose sampled structure can be used to bias the prediction from 2 routes:
			Template embedder
				Create template features from the pose sampled structure
					to replace the existing template features.
			Recycling embedder
				Use the coordinates of the pose sampled structure
					as input for the recycling embedder.
			Or both
		"""
		if self.topology.train.use_as_templates:
			print( "Creating template feats from the pose sampled structures..." )
			unrelaxed_protein  = self.get_protein_object( outputs = out )
			batch = update_template_feats(
				batch = batch,
				sys_name = self.sys_name,
				prot = unrelaxed_protein,
				model_id = frame,
				sys_config = self.topology.system,
				add_to_existing_templates = self.topology.train.add_to_existing_templates,
				device = self.device )
			print( f"New template feat dim: {batch['template_all_atom_positions'].shape}..." )

		if self.topology.train.recycle_pose:
			print( "Using the recycling embedder..." )
		else:
			# This let's the recycling embedder initialize MSA, Pair rep and x_prev to 0.
			out["final_atom_positions"] = None

		return out, batch

	################################################################################
	################################################################################
	def predict_pose( self,
		out: Dict[str, Any],
		restraint_features: Dict[str, Any],
		prev_frame_coord: torch.Tensor ) -> Dict[str, Any]:
		"""
		I cycle of pose sampling comprises of M steps.
		For M iterations (steps)
		Run pose sampler
				Define rigid bodies (either cy chain or based on pLDDT and/or PAE).
				Predict a rigid tranformation using a neural network.
				Apply the rigid transformation.
			Compute loss.
			Backpropagate.
		"""
		print( "\n\033[1mInitiate pose sampling now...\033[0m" )
		model = PoseSampling(
			model_config = self.topology.model,
			device = self.device )
		model.to( self.device )

		# Initialize the specified optimizer.
		optimizer = Optimizer( self.topology.optimizer ).forward( model.params() )
		pose_dict = {}

		track_metric = []
		for step in range( self.topology.train.num_steps ):
			print( f"\nPose sampling step: {step} --------------------------" )

			out = self.reinit_coords(
				out = out,
				prev_frame_coord = prev_frame_coord,
				pose_iter = True )

			# self.add_to_device( out )
			# with torch.autograd.detect_anomaly(): # Use while debugging.
			out = model.predict( out = out )

			cum_loss, losses = self.loss_fn.forward( out,
				self.gt_feature_dict,
				restraint_features )
			self.update_loss_dict( losses, update_pose_metrics = True )

			if step == self.topology.train.num_steps-1:
				last_step = True
			else:
				last_step = False
			metrics_dict = self.metrics_fn_pose.forward(
				out = out, last_epoch = last_step )
			self.update_data_metric_dict(
				metrics_dict = metrics_dict, update_pose_metrics = True )
			track_metric.append( metrics_dict["xlr"].item())

			# Remove computed violations.
			if "violation" in out:
				out.pop( "violation" )

			optimizer.zero_grad()
			cum_loss.backward()
			optimizer.step()
			self.remove_from_device( out )
		# self.add_to_device( out )
		# return out
			pose_dict[step] = {
				"final_atom_positions": out["final_atom_positions"],
				"final_atom_mask": out["final_atom_mask"],
				"asym_id": out["asym_id"],
				"plddt": out["plddt"],
				"msa": out["msa"],
				"pair": out["pair"]
			}
			del out
			out = pose_dict[step]

		# Log all predicted rigid transformations.
		self.store_transformations( transformations_dict = model.transformations )

		if self.topology.train.select_pose == "max":
			print( "Selecting the pose with max data satisfaction...")
			max_idx = np.argmax( track_metric )
			print( max_idx, "  ", track_metric[max_idx] )
			out = pose_dict[max_idx]
		elif self.topology.train.select_pose == "last":
			print( "Selecting the pose from last step..." )
			last_step = self.topology.train.num_steps-1
			out = pose_dict[last_step]
		else:
			raise ValueError( f"Incorrect value for the hyperparameters - last_step - specified. Use last/max..." )
		self.add_to_device( out )
		return out


	def predict_pose_random( self,
		out: Dict[str, Any],
		restraint_features: Dict[str, Any],
		prev_frame_coord: torch.Tensor ) -> Dict[str, Any]:
		"""
		Predict rigid transformations at random.
		Similar to predict_pose() above.
		"""
		print( "\n\033[1mInitiate random pose sampling now...\033[0m" )
		model = RandomPoseSampling(
			model_config = self.topology.model,
			device = self.device )

		pose_dict = {}

		track_metric = []
		for step in range( self.topology.train.num_steps ):
			print( f"\nPose sampling step: {step} --------------------------" )

			# if self.topology.train.reinit_step0 or step != 0:
			out = self.reinit_coords(
				out = out,
				prev_frame_coord = prev_frame_coord,
				pose_iter = True )

			out = model.predict( out = out )

			cum_loss, losses = self.loss_fn.forward( out,
				self.gt_feature_dict,
				restraint_features )
			self.update_loss_dict( losses, update_pose_metrics = True )

			if step == self.topology.train.num_steps-1:
				last_step = True
			else:
				last_step = False
			metrics_dict = self.metrics_fn_pose.forward(
				out = out, last_epoch = last_step )
			self.update_data_metric_dict(
				metrics_dict = metrics_dict, update_pose_metrics = True )
			track_metric.append( metrics_dict["xlr"].item())

			# Remove computed violations.
			if "violation" in out:
				out.pop( "violation" )

			self.remove_from_device( out )

			pose_dict[step] = {
				"final_atom_positions": out["final_atom_positions"],
				"final_atom_mask": out["final_atom_mask"],
				"asym_id": out["asym_id"],
				"plddt": out["plddt"],
				"msa": out["msa"],
				"pair": out["pair"]
			}
			del out
			out = pose_dict[step]
		# Log all predicted rigid transformations.
		self.store_transformations( transformations_dict = model.transformations )

		if self.topology.train.select_pose == "max":
			print( "Selecting the pose with max data satisfaction...")
			max_idx = np.argmax( track_metric )
			print( max_idx, "  ", track_metric[max_idx] )
			out = pose_dict[max_idx]
		elif self.topology.train.select_pose == "last":
			print( "Selecting the pose from last step..." )
			last_step = self.topology.train.num_steps-1
			out = pose_dict[last_step]
		else:
			raise ValueError( f"Incorrect value for the hyperparameters - last_step - specified. Use last/max..." )
		self.add_to_device( out )
		return out


	################################################################################
	################################################################################
	def af_sampling( self,
		out: Dict[str, Any],
		batch: Dict[str, torch.Tensor]
		):
		"""
		Apply the specified AF sampling technique:
			MSA subsampling
			Extra MSA subsampling
			MSA column amsking
			Masking MSA for cross-linked residues
		If using multiple, MSA subsampling will be done first.
		"""
		if self.topology.model.subsampling.enabled:
			orig_shape = batch["msa_feat"].shape
			batch = self.msa_subsampling( batch = batch )
			print( f"Full msa_feat = {orig_shape}" +
				f"\tSubsampled msa_feat = {batch['msa_feat'].shape}.." )
		else:
			print( "MSA subsampling switched off..." )

		if self.topology.model.extra_msa_subsampling.enabled:
			orig_shape = batch["extra_msa"].shape
			batch = self.extra_msa_subsampling( batch = batch )
			print( f"Full extra_msa = {orig_shape}" +
				f"\tSubsampled extra_msa = {batch['extra_msa'].shape}.." )
		else:
			print( "Extra MSA subsampling switched off..." )

		if self.topology.model.column_masking.enabled:
			batch = self.column_masking( batch = batch )
		else:
			print( "MSA column masking switched off..." )

		if self.topology.model.msa_xl_res_mask:
			print( "Masking MSA for XL residues..." )
			xl_res_dict = self.processed_feature_dict["restraint_features"]["xl_restraint"]["xl_res_dict"]
			batch = mask_msa_for_xl_res(
				batch = batch,
				xl_res_dict = xl_res_dict )
		else:
			print( "MSA masking for XL residues switched off..." )

		# if self.topology.model.struct_noising.enabled:
		# 	if not self.topology.train.skip_pose_sampling:
		# 		raise ValueError( "Structure noising not compatible with Pose sampling..." )
		# 	if not self.topology.train.recycle_pose:
		# 		raise ValueError( "Recycling must be allowed for structure noising..." )
		# 	print( "Using structure noising..." )
		# 	params = dict( self.topology.model.struct_noising.params )
		# 	mu_list = params["mu"]
		# 	sigma_list = params["sigma"]
		# 	mu = np.random.choice( mu_list, 1, replace = False )[0]
		# 	sigma = np.random.choice( sigma_list, 1, replace = False )[0]
		# 	params["mu"] = mu
		# 	params["sigma"] = sigma
		# 	print( f"Using mu = {params['mu']} and sigma = {params['sigma']}..." )
		# 	out = noised_structure( out = out, params = params )
		# 	out["final_atom_positions"] = out["final_atom_positions"].to( self.device )
		# else:
		# 	print( "Structure noising switched off..." )
		print( "\n" )

		return out, batch


	def msa_subsampling( self,
			batch: Dict[str, torch.Tensor] ) -> Dict[str, torch.Tensor]:
		"""
		Subsample the MSA and update the msa_feat.
		msa_feat -> [1, S, N, 49]
		deletion_matrix -> [1, S, N]
		cluster_deletion_mean -> [1, S, N]
		cluster_profile -> [1, S, N, 23]
		"""
		params = dict( self.topology.model.subsampling.params )
		neff_list = params["neff"]
		neff = random.sample( neff_list, 1 )[0]
		params["neff"] = int( neff )
		print( f"Using neff = {params['neff']}..." )


		subsampled_idx = msa_subsampler(
			msa = batch["msa"].cpu(),
			subsample_type = self.topology.model.subsampling.type,
			params = params )

		if "subsample" not in self.stats_dict:
			self.stats_dict["subsample"] = {
				"neff": [params["neff"]],
				"subsampled_indices": [subsampled_idx]
			}
		else:
			self.stats_dict["subsample"]["neff"].append( params["neff"] )
			self.stats_dict["subsample"]["subsampled_indices"].append( subsampled_idx )

		print( f"Subsampled MSA indices = {subsampled_idx}" )

		# Select subsampled MSA.
		batch["msa"] = batch["msa"][:,subsampled_idx,:].to( self.device )
		batch["msa_feat"] = batch["msa_feat"][:,subsampled_idx,:, :]
		batch["msa_mask"] = batch["msa_mask"][:,subsampled_idx,:]
		batch["deletion_matrix"] = batch["deletion_matrix"][:,subsampled_idx,:]
		batch['cluster_deletion_mean'] = batch["cluster_deletion_mean"][:,subsampled_idx,:]
		batch["cluster_profile"] = batch["cluster_profile"][:,subsampled_idx,:, :]
		return batch


	def extra_msa_subsampling( self,
			batch: Dict[str, torch.Tensor] ) -> Dict[str, torch.Tensor]:
		"""
		Subsample the extra MSA. The following features must be modified:
			extra_msa -> [1, S, N]
			extra_deletion_matrix -> [1, S, N]
			extra_msa_mask -> [1, S, N]
		Will just randomly sample indices for subsampling extra MSA.
		"""
		params = dict( self.topology.model.extra_msa_subsampling.params )
		neff_list = params["neff"]
		neff = random.sample( neff_list, 1 )[0]
		params["neff"] = int( neff )
		print( f"Using neff = {params['neff']}..." )

		indices = list( np.arange( 1, batch["extra_msa"].shape[1], 1 ) )
		subsampled_idx = random.sample( indices, neff )

		if "extra_msa_subsample" not in self.stats_dict:
			self.stats_dict["extra_subsample"] = {
				"neff": [params["neff"]],
				"subsampled_indices": [subsampled_idx]
			}
		else:
			self.stats_dict["extra_msa_subsample"]["neff"].append( params["neff"] )
			self.stats_dict["extra_msa_subsample"]["subsampled_indices"].append( subsampled_idx )

		# Select subsampled MSA.
		for k in ["extra_msa", "extra_deletion_matrix", "extra_msa_mask"]:
			# No extra-MSA information.
			if subsampled_idx == None:
				batch[k] = torch.zeros( batch[k].shape )[:,0,:]
			else:
				batch[k] = batch[k][:,subsampled_idx,:]
		return batch


	def column_masking( self,
			batch: Dict[str, torch.Tensor] ) -> Dict[str, torch.Tensor]:
		"""
		Apply MSA column masking.
		"""
		params = dict( self.topology.model.column_masking.params )
		mask_frac_list = params["mask_frac"]
		mask_frac = random.sample( mask_frac_list, 1 )[0]
		params["mask_frac"] = mask_frac
		print( f"Using column mask fraction = {params['mask_frac']}..." )

		# TODO: perform masking on CPU.
		batch, masked_idx = msa_column_masking(
			batch = batch,
			params = params )
		for k in ["msa", "deletion_matrix", "msa_feat"]:
			batch[k] = batch[k].to( self.device )

		print( f"Masked MSA columns = {masked_idx}" )
		if "col_mask" not in self.stats_dict:
			self.stats_dict["col_mask"] = {
				"mask_frac": [params["mask_frac"]],
				"masked_idx": [masked_idx]
			}
		else:
			self.stats_dict["col_mask"]["mask_frac"].append( params["mask_frac"] )
			self.stats_dict["col_mask"]["masked_idx"].append( masked_idx )
		return batch

	################################################################################
	################################################################################
	def update_loss_dict( self, losses: Dict[str, torch.Tensor], update_pose_metrics: bool ):
		"""
		Keep a tab on the loss per frame/step for all individual loss 
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


	def update_data_metric_dict( self,
		metrics_dict: Dict[str, float],
		update_pose_metrics: bool ):
		"""
		Save per frame/step metric values for all individual merics in stats_dict.
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


	def update_confidence_metrics( self, out: torch.Tensor ):
		"""
		Save the per frame confidence metrics in the stats_dict.
		Also save the num_recycles.
		"""
		if "confidence" not in self.stats_dict:
			self.stats_dict["confidence"].update( 
				{k: [] for k in ["plddt", "pae", "ptm", "iptm", "num_recycles"]} )

		str_ = ""
		for k1, k2 in zip(
			["plddt", "pae", "ptm", "iptm"],
			["plddt", "predicted_aligned_error", "ptm_score", "iptm_score", "num_recycles"] ):
			v = out[k2].detach().cpu()
			if k1 in ["ptm", "iptm"]:
				self.stats_dict["confidence"][k1].append( v.item() )
				str_ += f"{k1}: {v} \t"
			else:
				self.stats_dict["confidence"][k1].append( v  )
		print( f"Confidence: {str_}" )


	def store_transformations( self, transformations_dict: Dict[str, List] ):
		"""
		Store all predicted rigid transformations in the stats_dict every frame.
		"""
		if "transformations" not in self.stats_dict:
			self.stats_dict["transformations"].update( 
				{k: [] for k in ["rotation", "translation"]} )

		# For M pose sampling steps -> [M, B, 4].
		rot = torch.stack( transformations_dict["rotation"] ).cpu().numpy()
		# For M pose sampling steps -> [M, B, 3].
		trans = torch.stack( transformations_dict["rotation"] ).cpu().numpy()

		self.stats_dict["transformations"]["rotation"].append( rot )
		self.stats_dict["transformations"]["rotation"].append( trans )


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
		save_model.save( save_model.system, self.ensemble_file )

