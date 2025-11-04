from typing import List, Tuple, Dict, Any
import os, glob, time, copy
from collections import ( defaultdict )
import pickle as pkl
import numpy as np
import ml_collections as mlc

import torch

from openfold.config import model_config
from openfold.data import feature_pipeline
from openfold.np import protein

from models.rigid_sampler import PoseSampling
from models.recycler import Recycler
from models.af_sampling import (
	msa_subsampler, msa_column_masking,
	mask_msa_for_xl_res )

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
					feature_dict: Dict[str, np.ndarray],
					processed_feature_dict: Dict[str, torch.Tensor],
					gt_feature_dict: Dict[str, torch.Tensor],
					init_pred_dict: Dict[str, Any],
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
		self.gt_feature_dict = gt_feature_dict
		self.init_pred_dict = init_pred_dict
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
		# Load OpenFold configs file.
		self.ofold_config = model_config(
			self.topology.system_representation.config_preset,
			long_sequence_inference = self.topology.system_representation.long_sequence_inference,
			use_deepspeed_evoformer_attention = self.topology.system_representation.use_deepspeed_evoformer_attention,
			)
		if self.topology.model.no_templates:
			print( "Disabling template embedding..." )
			self.ofold_config.model.template.enabled = False
		else:
			self.ofold_config.model.template.enabled = True

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
			self.feature_dict["residue_index"] = self.feature_dict["residue_index"]

			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, action = "to_tensor" )
			self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, "add_dim" )

			self.gt_feature_dict = parse_nested_dict( self.gt_feature_dict, action = "to_tensor" )
			self.gt_feature_dict = parse_nested_dict( self.gt_feature_dict, "add_dim" )
			# dtype = torch.int64 is needed for torch.nn.functional.one_hot() in violation_loss calculation.
			self.gt_feature_dict["residue_index"] = self.gt_feature_dict["residue_index"].to( torch.int64 )

			self.init_pred_dict = parse_nested_dict( self.init_pred_dict, action = "to_tensor" )
			# self.init_pred_dict = parse_nested_dict( self.init_pred_dict, "add_dim" )

			for k in self.init_pred_dict:
				if k != "sm":
					self.init_pred_dict[k] = self.init_pred_dict[k].unsqueeze( 0 )
				else:
					# For sm output, 0th dim is the recycling dim.
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


	################################################################################
	################################################################################
	def get_final_atom_positions( self ):
		"""
		final_atom_positions can be obtained
			From an initial predicted structure.
			An all-1's tensor.
		"""
		# Skip using the initial predicted structure.
		if self.topology.train.init_zero:
			n = self.processed_feature_dict["asym_id"].shape[1]
			final_atom_positions = torch.zeros( [1, n, 37, 3] )
			out = {
				"msa": None,
				"pair": None,
				"final_atom_positions": final_atom_positions,
				"final_atom_mask": self.processed_feature_dict["atom37_atom_exists"],
				"asym_id": self.processed_feature_dict["asym_id"]
			}
		else:
			out = copy.deepcopy( self.init_pred_dict )
		return out


	def fit( self ) -> None:
		"""
		Add a singleton batch dim.
		Initialize:
			A SaveModel object to add and save predicted models to a CIF file.
			Recycler model.

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
		# Add batch dim.
		self.add_batch_dim()

		# Craete a SaveModel object.
		save_model_obj = SaveModels( title = self.sys_name, 
									output_format = "pdb",
									ensemble_dir = self.ensemble_dir )
		# Initialize the System object.
		save_model_obj.initialize_system()

		# Initialize the recycling model.
		recycler_model = Recycler(
			ofold_config = self.ofold_config,
			jax_param_path = self.jax_params_path,
			device = self.device )

		batch = {}
		for k in self.processed_feature_dict:
			if k == "restraint_features":
				continue
			batch[k] = self.processed_feature_dict[k]
			if isinstance( self.processed_feature_dict[k].dtype, float ):
				batch[k] = self.batch[k].to( dtype = torch.float32 )

		# Add batch and gt_features to device.
		self.add_to_device( batch )
		self.add_to_device( self.gt_feature_dict )

		t_start = time.perf_counter()

		# Keep track of moel_id's.
		self.stats_dict["model_id"] = []
		# Note the time per epoch.
		self.stats_dict["time_per_epoch"] = []

		# Skip using the initial predicted structure.
		# if self.topology.train.init_zero:
		# 	n = self.processed_feature_dict["asym_id"].shape[1]
		# 	final_atom_positions = torch.zeros( [1, n, 37, 3] )
		# 	out = {
		# 		"msa": None,
		# 		"pair": None,
		# 		"final_atom_positions": final_atom_positions,
		# 		"final_atom_mask": self.processed_feature_dict["atom37_atom_exists"],
		# 		"asym_id": self.processed_feature_dict["asym_id"]
		# 	}
		# else:
		# 	out = copy.deepcopy( self.init_pred_dict )
		out = self.get_final_atom_positions()

		# self.add_to_device( out )

		for epoch in range( self.topology.train.max_epochs ):
			# Note time for full run (pose sampling+recycling).
			t_s = time.perf_counter()
			print( f"\n\033[1mEpoch: {epoch} \033[0m" + "-"*20 )

			# If True, out from the previous epoch will be reused.
			if not self.topology.train.reuse_prediction:
				out = self.get_final_atom_positions()
				# out = copy.deepcopy( self.init_pred_dict )
				self.add_to_device( out )

			# Skip pose sampling if specified.
			if not self.topology.train.skip_pose_sampling:
				# # If True, out from the previous epoch will be reused.
				# if not self.topology.train.reuse_prediction:
				# 	out = self.get_final_atom_positions()
				# 	# out = copy.deepcopy( self.init_pred_dict )
				# 	self.add_to_device( out )

				out = self.predict_pose( out = out )
			else:
				if self.topology.train.fill_none:
					out["final_atom_positions"] = None
					# else keep the initial predicted final_atom_positions.
			# else:
			# 	# If True, out from the previous epoch will be reused.
			# 	if not self.topology.train.reuse_prediction:
			# 		out = self.get_final_atom_positions()
			# 		# out = copy.deepcopy( self.init_pred_dict )
			# 		self.add_to_device( out )
				# if self.topology.train.init_zero:
				# 	final_atom_positions = torch.zeros( [1, n, 37, 3] )
				# 	out = {
				# 	"msa": None,
				# 	"pair": None,
				# 	"final_atom_positions": final_atom_positions,
				# 	"final_atom_mask": self.processed_feature_dict["atom37_atom_exists"],
				# 	"asym_id": self.processed_feature_dict["asym_id"]
				# 	}
				# else:
				# 	out = copy.deepcopy( self.init_pred_dict )
			self.add_to_device( out )

			# Clear cache.
			torch.cuda.empty_cache()

			# Note time taken by recycling alone.
			t_r_s = time.perf_counter()
			print( "\n\033[1mRecycling optimized pose...\033[0m" )
			with torch.no_grad():
				batch = self.af_sampling( batch = batch )

				outputs = recycler_model.forward(
					out = out,
					batch = batch )
				out = copy.deepcopy( outputs )
				del outputs
				# Just compute loss but don't backpropagate.
				_, losses = self.loss_fn.forward( out,
					self.gt_feature_dict,
					self.processed_feature_dict["restraint_features"] )
				# Remove computed violations.
				if "violation" in out:
					out.pop( "violation" )

				self.remove_from_device( out )
			t_r_e = time.perf_counter()
			print( f"Time taken for recycling {epoch}: {( t_r_e - t_r_s )} seconds" )

			# Removed the modified MSA features.
			for k in ["msa", "msa_feat", "msa_mask", "deletion_matrix", "cluster_deletion_mean", "cluster_profile"]:
				batch[k] = self.processed_feature_dict[k].to( self.device )
			# Clear cache.
			torch.cuda.empty_cache()

			if epoch == self.topology.train.max_epochs-1:
				last_epoch = True
			else:
				last_epoch = False

			# Log the required loss and metrics.
			self.update_loss_dict( losses, update_pose_metrics = False )
			metrics_dict = self.metrics_fn.forward(
				out = out, last_epoch = last_epoch  )
			self.update_data_metric_dict(
				metrics_dict = metrics_dict, update_pose_metrics = False )
			self.update_confidence_metrics(
				out = out )

			# Add predicted model to the ensemble.
			unrelaxed_protein  = self.get_protein_object( outputs = out )
			self.add_protein_obj_to_stat( model_id = epoch,
											unrelaxed_protein = unrelaxed_protein )

			self.add_model( save_model_obj = save_model_obj,
							unrelaxed_protein = unrelaxed_protein,
							model_id = epoch )

			# using epoch as the model_id.
			self.stats_dict["model_id"].append( epoch )
			t_e = time.perf_counter()
			print( f"Time taken for epoch {epoch}: {( t_e - t_s )}  seconds" )
			self.stats_dict["time_per_epoch"].append( t_e - t_s )
			print( "-"*80 + "\n" + "-"*80 )

		t_end = time.perf_counter()
		print( f"Total Time taken for smapling: {( t_end - t_start )}  seconds" )

		# Save the ensemble on disk.
		self.save_model( save_model = save_model_obj )


	################################################################################
	################################################################################
	def predict_pose( self, out: Dict[str, Any] ):
		"""
		For M iterations
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

		for sub_epoch in range( self.topology.train.max_pose_iters ):
			print( f"\nPose sampling epoch: {sub_epoch} --------------------------" )

			# out = copy.deepcopy( self.init_pred_dict )
			if self.topology.train.reinit_per_pose_iter:
				out = self.get_final_atom_positions()
			self.add_to_device( out )
			# with torch.autograd.detect_anomaly(): # Use while debugging.
			out = model.predict( out = out )

			cum_loss, losses = self.loss_fn.forward( out,
				self.gt_feature_dict,
				self.processed_feature_dict["restraint_features"] )
			self.update_loss_dict( losses, update_pose_metrics = True )

			if sub_epoch == self.topology.train.max_pose_iters-1:
				last_epoch = True
			else:
				last_epoch = False
			metrics_dict = self.metrics_fn.forward(
				out = out, last_epoch = last_epoch )
			self.update_data_metric_dict(
				metrics_dict = metrics_dict, update_pose_metrics = True )

			# Remove computed violations.
			if "violation" in out:
				out.pop( "violation" )

			optimizer.zero_grad()
			cum_loss.backward()
			#for name, param in model.named_parameters():
			#	if param.grad is not None:
			#		print( f"{name} grad stats: min = {param.grad.min()}, max = {param.grad.max()}, nan = {torch.isnan( param.grad ).any()}" )
			optimizer.step()
			self.remove_from_device( out )

		return out


	################################################################################
	################################################################################
	def af_sampling( self,
		batch: Dict[str, torch.Tensor]
		):
		"""
		Apply the specified AF sampling technique:
			MSA subsampling
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

		return batch


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
		neff = np.random.choice( neff_list, 1, replace = False )[0]
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
		batch["msa_feat"] = batch["msa_feat"][:,subsampled_idx,:, :].to( self.device )
		batch["msa_mask"] = batch["msa_mask"][:,subsampled_idx,:].to( self.device )
		batch["deletion_matrix"] = batch["deletion_matrix"][:,subsampled_idx,:].to( self.device )
		batch['cluster_deletion_mean'] = batch["cluster_deletion_mean"][:,subsampled_idx,:].to( self.device )
		batch["cluster_profile"] = batch["cluster_profile"][:,subsampled_idx,:, :].to( self.device )
		return batch


	def column_masking( self,
			batch: Dict[str, torch.Tensor] ) -> Dict[str, torch.Tensor]:
		"""
		Apply MSA column masking.
		"""
		params = dict( self.topology.model.column_masking.params )
		mask_frac_list = params["mask_frac"]
		mask_frac = np.random.choice( mask_frac_list, 1, replace = False )[0]
		params["mask_frac"] = mask_frac
		print( f"Using column mask fraction = {params['mask_frac']}..." )

		batch, masked_idx = msa_column_masking(
			batch = batch,
			params = params )

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


	def update_data_metric_dict( self,
		metrics_dict: Dict[str, float],
		update_pose_metrics: bool ):
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


	def update_confidence_metrics( self, out: torch.Tensor ):
		"""
		Save the per epoch confidence metrics in the stats_dict.
		"""
		if "confidence" not in self.stats_dict:
			self.stats_dict["confidence"].update( 
				{k: [] for k in ["plddt", "pae", "ptm", "iptm"]} )

		str_ = ""
		for k1, k2 in zip(
			["plddt", "pae", "ptm", "iptm"],
			["plddt", "predicted_aligned_error", "ptm_score", "iptm_score"] ):
			v = out[k2]
			if k1 in ["ptm", "iptm"]:
				self.stats_dict["confidence"][k1].append( v.item() )
				str_ += f"{k1}: {v} \t"
			else:
				self.stats_dict["confidence"][k1].append( v  )
		print( f"Confidence: {str_}" )


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

