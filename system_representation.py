"""
Craete input features for running OpenFold.
Obtain initial prediction.
"""
from typing import Tuple, Dict, Any
import io, os, pathlib, shutil, pickle as pkl
import numpy as np
import ml_collections as mlc

import torch

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.np import protein
from openfold.utils.script_utils import (load_models_from_command_line, parse_fasta, run_model,
                                         prep_output, relax_protein)
from openfold.utils.tensor_utils import tensor_tree_map
from mod_openfold import parse, process_mmcif, np_example_to_features

from utils.utils import open_file_handler, parse_nested_dict, run_subprocess
from utils.pdb_utils import prep_output


def list_files_with_extensions( dir, extensions ):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


class RemapCUDAUnpickler(pkl.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
		# Change 'cuda:0' to 'cpu' if you want everything on CPU instead
			return lambda b: torch.load(io.BytesIO(b),
			map_location='cuda:0')
		return super().find_class(module, name)


class SystemRepresentation():
	def __init__( self,
			sys_name: str,
			is_multimer: bool,
			sys_rep_config: mlc.ConfigDict,
			fasta_dir: str,
			alignment_dir: str,
			ofold_output_dir: str, 
			seed_worker,
			device: str = "cpu"
		):
		self.sys_name = sys_name
		self.is_multimer = sys_rep_config.is_multimer
		self.db_dir = sys_rep_config.db_dir
		self.config_preset = sys_rep_config.config_preset
		self.db_preset = sys_rep_config.db_preset
		self.tool_base = sys_rep_config.tool_base
		self.openfold_checkpoint_path = sys_rep_config.model_checkpoint
		self.jax_params_path = sys_rep_config.jax_params_path
		self.save_feature_dicts = sys_rep_config.save_feature_dicts
		self.create_restraint_feats = sys_rep_config.create_restraint_feats

		self.init_model_prefix = sys_rep_config.init_model_prefix
		self.ofold_seed = sys_rep_config.seed
		self.cpu_cores = sys_rep_config.cpu_cores
		self.max_template_date = sys_rep_config.max_template_date
		self.subtract_plddt = sys_rep_config.subtract_plddt
		self.long_sequence_inference = sys_rep_config.long_sequence_inference
		self.use_deepspeed_evoformer_attention = sys_rep_config.use_deepspeed_evoformer_attention
		self.skip_relaxation = sys_rep_config.skip_relaxation

		self.databases_n_tools = sys_rep_config.databases_n_tools

		self.fasta_dir = fasta_dir
		self.alignment_dir = alignment_dir
		self.ofold_output_dir = ofold_output_dir
		self.device = device

		# Initialize a dict to store file paths.
		self.file_paths = {}

		seed_worker()

		#Will be created downstream.
		self.feature_dict = None
		self.processed_feature_dict = None
		self.init_pred_dict = None


	def forward( self ):
		"""
		Obtain the required features for an input required by OpenFold.
		We use the initial predicted structure as a pseudo ground truth structure.
		"""
		# Initialize the dict containing the file paths.
		self.file_paths = self.create_required_paths()
		os.makedirs( self.file_paths["ofold_pred_dir"], exist_ok = True )

		self.init_ofold_config()
		self.init_feature_processor()
		print( "Initialized the feature processor..." )

		# Get path for the initial prediction.
		init_model_path = self.get_init_model_path()

		if not os.path.exists( init_model_path ):
			# Get the input features (See AF2 supplementary).
			print( "Creating processed feature dict..." )
			self.feature_dict, self.processed_feature_dict, tag = self.prepare_input()

			if self.save_feature_dicts:
				w = open_file_handler( self.file_paths["feature_dict"], "wb" )
				pkl.dump( self.feature_dict, w, protocol = pkl.HIGHEST_PROTOCOL )
				w.close()

				w = open_file_handler( self.file_paths["processed_feature_dict"], "wb" )
				pkl.dump( self.processed_feature_dict, w, protocol = pkl.HIGHEST_PROTOCOL )
				w.close()

			self.processed_feature_dict = parse_nested_dict(
				self.processed_feature_dict,
				"add_to_device",
				self.device )

			# Run the prediction.
			print( "Running model prediction..." )
			out, output_directory = self.get_init_pred(
				processed_feature_dict = self.processed_feature_dict,
				tag = tag
			)
			self.init_pred_dict = out
			del out

			w = open_file_handler( self.file_paths["out_dict_path"], "wb" )
			tmp_dict = parse_nested_dict( self.init_pred_dict, action = "detach" )
			pkl.dump( tmp_dict, w, protocol = pkl.HIGHEST_PROTOCOL )
			w.close()

			# Save the prediction on disk.
			self.save_init_pred(
				feature_dict = self.feature_dict,
				processed_feature_dict = self.processed_feature_dict,
				out = self.init_pred_dict,
				output_directory = output_directory )
		else:
			if not self.save_feature_dicts:
				self.feature_dict, self.processed_feature_dict, tag = self.prepare_input()
			else:
				f = open_file_handler( self.file_paths["feature_dict"], "rb" )
				self.feature_dict = RemapCUDAUnpickler( f ).load()
				f.close()

				f = open_file_handler( self.file_paths["processed_feature_dict"], "rb" )
				self.processed_feature_dict = RemapCUDAUnpickler(f).load()
				f.close()

			print( self.processed_feature_dict.keys() )

			f = open_file_handler( self.file_paths["out_dict_path"], "rb" )
			self.init_pred_dict = RemapCUDAUnpickler(f).load()
			f.close()
		self.processed_feature_dict = parse_nested_dict( self.processed_feature_dict, "detach" )
		self.init_pred_dict = parse_nested_dict( self.init_pred_dict, "detach" )

		# Remove all recycling dim.
		cycle_no = 0
		fetch_cur_batch = lambda t: t[..., cycle_no]
		self.processed_feature_dict = tensor_tree_map( fetch_cur_batch, self.processed_feature_dict )

		self.gt_feature_dict = self.get_feature_from_init_struct()
		# if self.create_restraint_feats:
			# Add masks for excluded volume and sequence connectivity.
			# intra_ev_mask, inter_ev_mask = self.get_excluded_volume_feats()
			# connectivity_mask = self.get_sequence_connectivity_feats()

			# Get the gt_features
			# self.gt_feature_dict = self.get_feature_from_init_struct()
			# self.gt_feature_dict["intra_ev_mask"] = intra_ev_mask
			# self.gt_feature_dict["inter_ev_mask"] = inter_ev_mask
			# self.gt_feature_dict["connectivity_mask"] = connectivity_mask

	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create the required files, dir paths.
		"""
		file_paths = {
			"feature_dict": os.path.join(
				self.ofold_output_dir,
				"predictions/feature_dict.pkl" ),
			"processed_feature_dict": os.path.join(
				self.ofold_output_dir,
				"predictions/processed_feature_dict.pkl" ),
			"out_dict_path": os.path.join(
				self.ofold_output_dir,
				"predictions/output_dict.pkl" ),
			"ofold_pred_dir": os.path.join(
				self.ofold_output_dir,
				"predictions/" ),
			"pred_file_prefix": f"{self.sys_name}_init_pred",
		}
		return file_paths

	def init_feature_processor( self ):
		self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )


	def init_ofold_config( self ):
		self.ofold_config = model_config(
			self.config_preset,
			long_sequence_inference = self.long_sequence_inference,
			use_deepspeed_evoformer_attention = self.use_deepspeed_evoformer_attention,
			)

	def get_init_model_path( self ):
		"""
		Get the path for the initial model prediction.
		"""
		if self.skip_relaxation:
			init_model_path = os.path.join(
				self.ofold_output_dir,
				"predictions/" + self.file_paths["pred_file_prefix"] + "_unrelaxed.cif" )
		else:
			init_model_path = os.path.join(
				self.ofold_output_dir,
				"predictions/" + self.file_paths["pred_file_prefix"] + "_relaxed.cif" )
		return init_model_path


	################################################################################
	################################################################################
	def get_alignment( self,
		tmp_fasta_path: str,
		local_alignment_dir: str
	):
		"""
		Perform alignment for the query sequence.
		"""
		if self.is_multimer:
			template_searcher = hmmsearch.Hmmsearch(
				binary_path = self.databases_n_tools.hmmsearch_binary_path,
				hmmbuild_binary_path = self.databases_n_tools.hmmbuild_binary_path,
				database_path = self.databases_n_tools.pdb_seqres_database_path,
			)
		else:
			template_searcher = hhsearch.HHSearch(
				binary_path = self.databases_n_tools.hhsearch_binary_path,
				databases = [self.databases_n_tools.pdb70_database_path],
			)

		alignment_runner = data_pipeline.AlignmentRunner(
			jackhmmer_binary_path = self.databases_n_tools.jackhmmer_binary_path,
			hhblits_binary_path = self.databases_n_tools.hhblits_binary_path,
			uniref90_database_path = self.databases_n_tools.uniref90_database_path,
			mgnify_database_path = self.databases_n_tools.mgnify_database_path,
			bfd_database_path = self.databases_n_tools.bfd_database_path,
			uniref30_database_path = self.databases_n_tools.uniref30_database_path,
			uniclust30_database_path = self.databases_n_tools.uniclust30_database_path,
			uniprot_database_path = self.databases_n_tools.uniprot_database_path,
			template_searcher = template_searcher,
			use_small_bfd = self.databases_n_tools.bfd_database_path is None,
			no_cpus = self.cpu_cores
		)

		alignment_runner.run(
			tmp_fasta_path, local_alignment_dir
		)


	def precompute_alignments( self, tags, seqs, ):
		"""
		Taken from run_pretrained_openfold.py.
		OpenFold inference runs alignment for all copies in case of a homomeric input.
			We reuse the alignment for homomers.
		"""
		# Keep track of the tag and the associated seq.
		tmp_dict = {}
		for tag, seq in zip(tags, seqs):
			tmp_fasta_path = os.path.join(
				self.ofold_output_dir, f"tmp_{os.getpid()}.fasta" )
			w = open_file_handler( tmp_fasta_path, "w" )
			w.write( f">{tag}\n{seq}" )
			w.close()

			local_alignment_dir = os.path.join( self.alignment_dir, tag )

			# if args.use_precomputed_alignments is None:
			if not os.path.exists( local_alignment_dir ):
				print( f"Generating alignments for {tag}..." )

				os.makedirs( local_alignment_dir, exist_ok = True )
				# For identical sequences reuse the alignments.
				if seq in tmp_dict:
					prev_local_alignment_dir = tmp_dict[seq]["local_alignment_dir"]
					src = pathlib.Path( prev_local_alignment_dir )
					dst = pathlib.Path( local_alignment_dir )
					for p in src.iterdir():
						shutil.move( str( p ), str( dst ) )
					print( f"Reusing alignments for {tag} from {tmp_dict[seq]['tag']}..." )
				else:
					self.get_alignment(
						tmp_fasta_path = tmp_fasta_path,
						local_alignment_dir = local_alignment_dir
					)
					tmp_dict[seq] = {
						"tag": tag,
						"local_alignment_dir": local_alignment_dir
					}
			else:
				print( f"Using precomputed alignments for {tag} at {local_alignment_dir}..." )

			# Remove temporary FASTA file
			os.remove( tmp_fasta_path )


	def generate_feature_dict(
			self,
			tags,
			seqs,
			data_processor ):
		"""
		Taken from run_pretrained_openfold.py.
		"""
		tmp_fasta_path = os.path.join( self.ofold_output_dir, f"tmp_{os.getpid()}.fasta" )

		with open(tmp_fasta_path, "w") as fp:
			fp.write(
				'\n'.join([f">{tag}\n{seq}" for tag, seq in zip( tags, seqs)] )
			)
		feature_dict = data_processor.process_fasta(
			fasta_path = tmp_fasta_path, alignment_dir = self.alignment_dir,
		)

		# Remove temporary FASTA file
		os.remove( tmp_fasta_path )

		return feature_dict


	def prepare_input( self ):
		"""
		Create input features for running AF2.
		Taken from run_pretrained_openfold.py -> main().
		feature_dict
			aatype, residue_index, seq_length, msa, num_alignments,
			template_aatype, template_all_atom_mask, template_all_atom_positions,
			asym_id, sym_id, entity_id,
			deletion_matrix, deletion_mean,
			all_atom_mask, all_atom_positions,
			assembly_num_chains, entity_mask, num_templates,
			cluster_bias_mask, bert_mask, seq_mask, msa_mas
		processed_feature_dict
			aatype, residue_index, seq_length, msa, num_alignments,
			template_aatype, template_all_atom_mask, template_all_atom_positions,
			asym_id, sym_id, entity_id,
			deletion_matrix, seq_mask, msa_mask, msa_profile, target_feat,
			atom14_atom_exists, residx_atom14_to_atom37, residx_atom37_to_atom14, atom37_atom_exists,
			extra_msa, extra_deletion_matrix, extra_msa_mask, bert_mask, true_msa,
			cluster_profile, cluster_deletion_mean, msa_feat, use_clamped_fape
		The shapes of all features can be found in openfold config.py.
		"""
		if self.is_multimer:
			template_featurizer = templates.HmmsearchHitFeaturizer(
				mmcif_dir = self.databases_n_tools.template_mmcif_dir,
				max_template_date = self.max_template_date,
				max_hits = self.ofold_config.data.predict.max_templates,
				kalign_binary_path = self.databases_n_tools.kalign_binary_path,
				# path to a file with a mapping from PDB IDs to their release dates.
				#	 Thanks to this we don't have to redundantly parse mmCIF files to get that information.
				release_dates_path = None,
				# contains a mapping from obsolete PDB IDs to the PDB IDs of their replacements.
				obsolete_pdbs_path = None
			)
		else:
			template_featurizer = templates.HhsearchHitFeaturizer(
				mmcif_dir = self.databases_n_tools.template_mmcif_dir,
				max_template_date = self.max_template_date,
				max_hits = self.ofold_config.data.predict.max_templates,
				kalign_binary_path=self.databases_n_tools.kalign_binary_path,
				release_dates_path = None,
				obsolete_pdbs_path = None
			)
		data_processor = data_pipeline.DataPipeline(
			template_featurizer = template_featurizer,
		)
		if self.is_multimer:
			# For multimer.
			data_processor = data_pipeline.DataPipelineMultimer(
				monomer_data_pipeline = data_processor,
			)

		self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )
		
		tag_list = []
		seq_list = []
		for fasta_file in list_files_with_extensions( self.fasta_dir, (".fasta", ".fa") ):
			# Gather input sequences
			fasta_path = os.path.join ( self.fasta_dir, fasta_file )
			with open( fasta_path, "r" ) as fp:
				data = fp.read()

			tags, seqs = parse_fasta( data )

			if not self.is_multimer and len( tags ) != 1:
				print(
					f"{fasta_path} contains more than one sequence but " +
					f"multimer mode is not enabled. Skipping..."
				)
				continue

			tag = '-'.join(tags)

			tag_list.append( ( tag, tags ) )
			seq_list.append( seqs )

		seq_sort_fn = lambda target: sum( [len( s ) for s in target[1]] )
		sorted_targets = sorted( zip( tag_list, seq_list ), key = seq_sort_fn )

		if len( sorted_targets ) == 0:
			print( sorted_targets )
			raise ValueError( f"No fasta files found for {self.sys_name}..." )
		elif len( sorted_targets ) > 1:
			print( sorted_targets )
			raise ValueError( f"Multiple fasta files ({len( sorted_targets )}) detected for {self.sys_name}..." )
		( tag, tags ), seqs = sorted_targets[0]

		# Precompute aignments.
		self.precompute_alignments( tags = tags, seqs = seqs )

		feature_dict = self.generate_feature_dict(
			tags,
			seqs,
			data_processor
		)

		processed_feature_dict = self.feature_processor.process_features(
			feature_dict, mode = "predict", is_multimer = self.is_multimer
		)

		return feature_dict, processed_feature_dict, tag


	################################################################################
	################################################################################
	def get_init_pred(
			self,
			processed_feature_dict: Dict[str, Any],
			tag: str
			) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
		"""
		Obtain the initial predicted structure.
		Taken from run_pretrained_openfold.py -> main().

		output_dict
			msa, pair, single, sm, final_atom_positions, final_atom_mask,
			final_affine_tensor, num_recycles, asym_id,
			lddt_logits, plddt, distogram_logits, masked_msa_logits,
			experimentally_resolved_logits,
			tm_logits, ptm_score, iptm_score,
			weighted_ptm_score, aligned_confidence_probs,
			predicted_aligned_error, max_predicted_aligned_error
		"""
		model_generator = load_models_from_command_line(
			self.ofold_config,
			self.device,
			self.openfold_checkpoint_path,
			self.jax_params_path,
			self.ofold_output_dir )

		model, output_directory = [( a, b ) for a, b in model_generator][0]

		out = run_model( model, processed_feature_dict, tag, self.ofold_output_dir )

		# Toss out the recycling dimensions --- we don't need them anymore
		processed_feature_dict = tensor_tree_map(
			lambda x: np.array( x[..., -1].cpu() ),
			processed_feature_dict )
		out = tensor_tree_map( lambda x: np.array( x.cpu() ), out )

		return out, output_directory


	def get_prot_from_pred( self,
			out: Dict[str, Any],
			feature_dict: Dict[str, Any],
			processed_feature_dict: Dict[str, Any],
		) -> protein.Protein:
		"""
		Create a Protein object from the predcition.
		"""
		# Toss out the recycling dimensions --- we don't need them anymore
		processed_feature_dict = tensor_tree_map(
			lambda x: np.array( x[..., -1].cpu() ),
			processed_feature_dict )

		unrelaxed_protein = prep_output(
			out = out,
			batch = processed_feature_dict,
			feature_dict = feature_dict,
			feature_processor = self.feature_processor,
			config_preset = self.config_preset,
			subtract_plddt = self.subtract_plddt
		)
		return unrelaxed_protein


	################################################################################
	################################################################################
	def save_init_pred(
			self,
			out: Dict[str, Any],
			feature_dict: Dict[str, Any],
			processed_feature_dict: Dict[str, Any],
			output_directory: str ):
		"""
		Save the initial predicted structure.
		"""
		unrelaxed_protein = self.get_prot_from_pred(
			out = out,
			feature_dict = feature_dict,
			processed_feature_dict = processed_feature_dict )

		output_name = self.file_paths["pred_file_prefix"] + "_unrelaxed.cif"
		unrelaxed_output_path = os.path.join(
			self.ofold_output_dir,
			output_name
		)
		print( unrelaxed_output_path )
		w = open( unrelaxed_output_path, "w" )
		w.write( protein.to_modelcif( unrelaxed_protein ) )
		w.close()

		output_name = self.file_paths["pred_file_prefix"] + "_unrelaxed.pdb"
		unrelaxed_output_path = os.path.join(
			self.ofold_output_dir,
			output_name
		)
		print( unrelaxed_output_path )
		w = open( unrelaxed_output_path, "w" )
		w.write( protein.to_pdb( unrelaxed_protein ) )
		w.close()
		if not self.skip_relaxation:
			# Relax the prediction.
			print( f"Running AMBER relaxation..." )
			output_name = self.file_paths["pred_file_prefix"]
			relax_protein(
				self.ofold_config,
				self.device,
				unrelaxed_protein,
				output_directory,
				output_name,
				cif_output = True )


	################################################################################
	################################################################################
	def get_feature_from_init_struct( self ) -> Dict[str, Any]:
		"""
		Using the initial predicted structure as the pseudo ground truth structure
			to obtain structural features (gt_features) as mentioned below:
				all_atom_positions: --> [N, 37, 3]
				all_atom_mask: --> [N, 37]
				asym_id: -->  [N]
				sym_id: --> [N]
				entity_id: --> [N]
				aatype: --> [N]
				atom14_atom_exists: --> [N, 14]
				residx_atom14_to_atom37: --> [N, 14]
				residx_atom37_to_atom14: --> [N, 37]
				atom37_atom_exists: --> [N, 37]
				atom14_gt_exists: --> [N, 14]
				atom14_gt_positions: --> [N, 14, 3]
				atom14_alt_gt_positions  --> [N, 14, 3]
				atom14_alt_gt_exists: --> [N, 14]
				atom14_atom_is_ambiguous: --> [N, 14]
				rigidgroups_gt_frames: --> [N, 8, 4, 4]
				rigidgroups_gt_exists: --> [N, 8]
				rigidgroups_group_exists: --> [N, 8]
				rigidgroups_group_is_ambiguous: --> [N, 8]
				rigidgroups_alt_gt_frames: --> [N, 8, 4, 4]
				torsion_angles_sin_cos: --> [N, 7, 2]
				alt_torsion_angles_sin_cos: --> [N, 7, 2]
				torsion_angles_mask: --> [N, 7]
				pseudo_beta: --> [N, 3])
				pseudo_beta_mask: --> [N]
				backbone_rigid_tensor: --> [N, 4, 4]
				backbone_rigid_mask: --> [N]
				chi_angles_sin_cos: --> [N, 4, 2]
				chi_mask: --> [N, 4]
		Parse the .cif file to get an mmcif_object.
		This mmcif_object is not the same as Biopython structure object.
		"""
		init_model_path = self.get_init_model_path()
		with open( init_model_path, "r" ) as f:
			mmcif_string = f.read()

		mmcif_object = parse( 
			file_id = self.sys_name, mmcif_string = mmcif_string
		).mmcif_object

		# Extract relevant relevant features from the mmcif_object.
		np_example = process_mmcif( mmcif_object )

		# Obtain the ground truth features.
		data = np_example_to_features(
				np_example = np_example,
				config = self.ofold_config["data"],
				mode = "train",
				is_multimer = self.is_multimer )

		gt_feats = data["gt_features"]
		gt_feats["residue_index"] = data["residue_index"]
		return gt_feats


	################################################################################
	################################################################################
	# def get_excluded_volume_feats( self ):
	# 	"""
	# 	Create masks to account for:
	# 		Only intrachain residue pairs.
	# 		Only interchain residue pairs.
	# 	Mask out all diagonal elements.

	# 	intra_ev_mask, inter_ev_mask -> [N, N] 
	# 	"""
	# 	asym_id = torch.from_numpy( self.feature_dict["asym_id"] )

	# 	N = asym_id.shape[0]

	# 	#ignore all diagonal element.
	# 	diagonal_mask = torch.ones( ( N, N) ) - np.eye( N )
	# 	diagonal_mask = diagonal_mask.int()

	# 	intra_ev_mask = ( asym_id[None, :] == asym_id[:, None] ).int()
	# 	intra_ev_mask *= diagonal_mask
	# 	inter_ev_mask = ( asym_id[None, :] != asym_id[:, None] ).int()
	# 	inter_ev_mask *= diagonal_mask

	# 	return intra_ev_mask, inter_ev_mask


	# def get_sequence_connectivity_feats( self ):
	# 	"""
	# 	Create a mask to ignore all but intrachain adjacent residues.
	# 	Using an asymmetric mask to account for only ij pairs.

	# 	connectivity_maks -> [N, N]
	# 	"""
	# 	residue_index = self.processed_feature_dict["residue_index"]
	# 	asym_id = torch.from_numpy( self.feature_dict["asym_id"] )

	# 	N = asym_id.shape[0]

	# 	intra_chain_mask = ( asym_id[None, :] == asym_id[:, None] ).int()

	# 	# Adjacent residues in sequence.
	# 	adjacent_mask = torch.zeros( [N, N] )
	# 	adjacent_mask[residue_index, residue_index+1] = 1

	# 	connectivity_mask = intra_chain_mask*adjacent_mask

	# 	return connectivity_mask

	################################################################################
	################################################################################
	def get_distance_map( self, final_atom_position: torch.Tensor ) -> torch.Tensor:
		"""
		Given the final_atom_position, create a Ca-ca distance map,
			to be used as pseudo ground truth.
		final_atom_position -> [N, 37, 3]
		"""
		ca_coords = final_atom_position[:, 1, :]
		# [N, N, 3]
		diff = ca_coords[:, None, :] - ca_coords[None, :, :]
		# [N, N, 3]
		squared_diff = torch.sum( diff**2, dim = -1 )
		gt_distance_map = torch.sqrt( squared_diff )
		return gt_distance_map


	def get_distogram_bins( self ):
		# These are taken from openFold.utils.loss.distogram_loss().
		min_bin = 2.3125
		max_bin = 21.6875
		no_bins = 64

		# Last bin is a catch all bin.
		boundaries = np.linspace(
			min_bin,
			max_bin,
			no_bins - 1 )
		# Add the last bin.
		bin_width = boundaries[1] - boundaries[0]
		last_bin = boundaries[-1] + bin_width
		boundaries = np.append( boundaries, last_bin )

		return torch.from_numpy( boundaries )


	def get_distance_distribution( self, distogram_logits: np.ndarray ) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Given the predicted distogram, obtain the per-residue pair
			mean and variance.
		"""
		distogram = torch.softmax( distogram_logits, dim = -1 )
			# torch.from_numpy( distogram_logits ), dim = -1
			# ).numpy()

		boundaries = self.get_distogram_bins()

		# E(x) = x*p(x)
		E_x = torch.sum( boundaries[None, None, :]*distogram, dim = -1 )

		E_x2 = torch.sum( distogram*boundaries[None, None, :]**2, dim = -1 )
		# var = E(x**2) - E_x**2
		var = E_x2 - E_x**2

		assert E_x.shape == var.shape, "Shape mismatch: mean and variance matrices..."

		return E_x, var

if __name__ == "__main__":
	SystemRepresentation().forward()

