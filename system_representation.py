import numpy as np
import os
import glob
import Bio
from Bio.PDB import PDBParser, MMCIFParser
import collections
import ml_collections
import copy

import torch

from mod_openfold import parse, process_mmcif, np_example_to_features

from typing import Optional

# from openfold.data import ( data_pipeline, feature_pipeline, mmcif_parsing, msa_pairing )
# from openfold.data.data_pipeline import ( add_assembly_features, make_sequence_features )
# from openfold.data.msa_pairing import ( _correct_post_merged_feats, _merge_homomers_dense_msa,
# 					_merge_features_from_multiple_chains )
# from openfold.data.input_pipeline_multimer import prepare_ground_truth_features
# from openfold.data.feature_processing_multimer import _make_seq_mask
# from openfold.data.feature_pipeline import np_to_tensor_dict
from openfold.config import model_config

from commands import ( monomer_cmd, monomer_precomp_aln_cmd,
						multimer_cmd, multimer_precomp_aln_cmd )
from utils import run_subprocess

# FeatureDict = MutableMapping[str, np.ndarray]


class SystemRepresentation():
	def __init__( self, sys_name: str, ofold_dir: str, ofold_script: str, 
					fasta_dir: str, alignment_dir: str, 
					output_dir: str, config_preset: str, 
					ckpt_path: Optional[str], mode: str,
					# init_struct_pdb: str, init_struct_cif: str,
					cpu_cores: int, device: str = "cpu" ):
		self.sys_name = sys_name
		self.openfold_dir = ofold_dir
		self.script = ofold_script
		self.fasta_dir = fasta_dir 
		self.alignment_dir = alignment_dir
		self.output_dir = output_dir
		self.config_preset = config_preset
		self.ckpt_path = ckpt_path
		self.cpu_cores = cpu_cores
		self.device = device
		self.mode = mode


	def forward( self ):
		if len( glob.glob( f"{self.output_dir}/predictions/*unrelaxed.cif" ) ) != 0:
			print( "\nInitial structure for the system exists..." )
			self.init_struct_cif = glob.glob( f"{self.output_dir}/predictions/*unrelaxed.cif" )[0]
		
		else:
			print( "\nPredicting the initial structure for the system..." )
			if not os.path.exists( os.path.abspath( f"{self.output_dir}/predictions/*unrelaxed.cif" ) ):
				self.get_initial_structure()
			# OpenFold predicted initial structure in CIF format.
			self.init_struct_cif = glob.glob( f"{self.output_dir}/predictions/*unrelaxed.cif" )[0]

		print( "\nCreating ground truth features from the initial structure..." )
		data = self.get_feature_from_init_struct()
		# # Parse the .cif file to get an mmcif_object.
		# # This mmcif_object is not the same as Biopython structure object.
		# with open( self.init_struct_cif, "r" ) as f:
		#     mmcif_string = f.read()

		# mmcif_object = parse( # mmcif_parsing.parse(
		#     file_id = "2ayo", mmcif_string = mmcif_string
		# ).mmcif_object

		# print( os.path.exists( self.init_struct_cif ) )
		# # Extract relevant relevant features from the mmcif_object.
		# np_example = self.process_mmcif( mmcif_object )

		# print( "Load config file..." )
		# config = model_config( "model_1_multimer_v3" )

		# # Obtain the ground truth features.
		# data = self.np_example_to_features(
		# 			np_example = np_example,
		# 			config = config["data"],
		# 			mode = "train",
		# 			is_multimer = True )

		print( data.keys() )
		for k in data["gt_features"].keys():
			print( k, ": --> ", data["gt_features"][k].shape )
		return data


	def get_initial_structure( self ):
		"""
		Run monomer/multimer prediction based on mode.
		openfold/script_utils.run_model has been modified ot save the 
			MSA, Pair, and Single representations on disk.
		**Note: To get the struct with correct residue indices in .cif format, 
				I've modified the function openfold.utils.script_utils.prep_output().

		Input:
		----------
		Does not take any arguments.

		Returns:
		----------
		None
		"""
		if self.mode == 'mono':
			self.run_monomer_prediction()
		elif self.mode == 'multi':
			self.run_multimer_prediction()


	def run_monomer_prediction( self ):
		"""
		Use OpenFold for monomer structure prediction.
			Use precomputed alignments if available.

		Input:
		----------
		training --> (bool) specifies if OpenFold is to be used for training/prediction.

		Returns:
		----------
		None
		"""

		# if training:
		# 	script = "unfrozen_pretrained.py"
		# else:
		# 	script = "run_pretrained_openfold.py"

		# Move to OpenFold dir.
		os.chdir( self.openfold_dir )
		
		if os.path.exists( self.alignment_dir ):
			print( "using precomputed alignments for monomer prediction..." )
			command = monomer_precomp_aln_cmd( self.script, self.fasta_dir, 
											self.alignment_dir, self.output_dir,
											self.config_preset, self.ckpt_path,
											self.cpu_cores, self.device )

		else:
			command = monomer_cmd( self.script, self.fasta_dir, self.output_dir,
									self.config_preset,  self.ckpt_path,
									self.cpu_cores, self.device )

		print( "Running monomer prediction..." )
		run_subprocess( command )



	def run_multimer_prediction( self ):
		"""
		Use OpenFold for multimer structure prediction.
			Use precomputed alignments if available.

		Input:
		----------
		training --> (bool) specifies if OpenFold is to be used for training/prediction.

		Returns:
		----------
		None
		"""
		# Move to OpenFold dir.
		os.chdir( self.openfold_dir )

		if os.path.exists( self.alignment_dir ):
			print( "using precomputed alignments for multimer prediction..." )
			command = multimer_precomp_aln_cmd( self.script, self.fasta_dir, 
												self.alignment_dir, self.output_dir,
												self.config_preset, self.cpu_cores, self.device )

		else:
			command = multimer_cmd( self.script, self.fasta_dir, self.output_dir,
									self.config_preset, self.cpu_cores, self.device )

		print( "Running multimer prediction..." )
		run_subprocess( command )


	def get_feature_from_init_struct( self ):
		# Parse the .cif file to get an mmcif_object.
		# This mmcif_object is not the same as Biopython structure object.
		with open( self.init_struct_cif, "r" ) as f:
		    mmcif_string = f.read()

		mmcif_object = parse( 
		    file_id = "2ayo", mmcif_string = mmcif_string
		).mmcif_object

		print( os.path.exists( self.init_struct_cif ) )
		# Extract relevant relevant features from the mmcif_object.
		np_example = process_mmcif( mmcif_object )

		print( "Load config file..." )
		config = model_config( "model_1_multimer_v3" )

		# Obtain the ground truth features.
		data = np_example_to_features(
				np_example = np_example,
				config = config["data"],
				mode = "train",
				is_multimer = True )

		return data


	# def get_mmcif_features( self,
	#         mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
	# ) -> FeatureDict:
	#     """
	#     Taken from openfold.data.data_pipeline.DataPipelineMultimer()
	#     """
	#     mmcif_feats = {}

	#     all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
	#         mmcif_object=mmcif_object, chain_id=chain_id
	#     )
	#     # All atom coordinates in atom37 representations [L, 37, 3].
	#     mmcif_feats["all_atom_positions"] = all_atom_positions
	#     # Binary all atom mask in atom37 representations  [L, 37, 3].
	#     mmcif_feats["all_atom_mask"] = all_atom_mask

	#     mmcif_feats["resolution"] = np.array(
	#         mmcif_object.header["resolution"], dtype=np.float32
	#     )

	#     # mmcif_feats["release_date"] = np.array(
	#     #     [mmcif_object.header["release_date"].encode("utf-8")], dtype=object
	#     # )

	#     mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

	#     return mmcif_feats


	# def convert_monomer_features( self,
	#     monomer_features: FeatureDict,
	#     chain_id: str
	# 	) -> FeatureDict:
	# 	"""
	# 	Taken from openfold.msa_pairing.py
	# 	"""
	# 	"""Reshapes and modifies monomer features for multimer models."""
	# 	converted = {}
	# 	converted["auth_chain_id"] = np.asarray( chain_id, dtype = object )
	# 	unnecessary_leading_dim_feats = {
	# 		"sequence", "domain_name", "num_alignments", "seq_length"
	# 	}
	# 	for feature_name, feature in monomer_features.items():
	# 		if feature_name in unnecessary_leading_dim_feats:
	# 			# asarray ensures it's a np.ndarray.
	# 			feature = np.asarray( feature[0], dtype = feature.dtype )
	# 		elif feature_name == "aatype":
	# 			# The multimer model performs the one-hot operation itself.
	# 			feature = np.argmax( feature, axis = -1 ).astype( np.int32 )
	# 		# We don't need this, but just left it as is.
	# 		elif feature_name == "template_aatype":
	# 			feature = np.argmax( feature, axis = -1 ).astype( np.int32 )
	# 			new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
	# 			feature = np.take( new_order_list, feature.astype( np.int32 ), axis = 0 )
	# 		# We don't need this, but just left it as is.
	# 		elif feature_name == "template_all_atom_masks":
	# 			feature_name = "template_all_atom_mask"
	# 		converted[feature_name] = feature
	# 	return converted


	# def merge_chain_features( self, np_chains_list: List[Mapping[str, np.ndarray]]
	# 						# pair_msa_sequences: bool, max_templates: int
	# 						) -> Mapping[str, np.ndarray]:
	# 	"""
	# 	Taken from openfold.msa_pairing.py
	# 	"""
	# 	"""Merges features for multiple chains to single FeatureDict.

	# 	Args:
	# 		np_chains_list: List of FeatureDicts for each chain.
	# 		pair_msa_sequences: Whether to merge paired MSAs.
	# 		max_templates: The maximum number of templates to include.

	# 	Returns:
	# 	Single FeatureDict for entire complex.
	# 	"""
	# 	# np_chains_list = _pad_templates(
	# 	# 	np_chains_list, max_templates=max_templates)
	# 	np_chains_list = _merge_homomers_dense_msa( np_chains_list )
	# 	# Unpaired MSA features will be always block-diagonalised; paired MSA
	# 	# features will be concatenated.
	# 	np_example = _merge_features_from_multiple_chains(
	# 					np_chains_list, pair_msa_sequences = False )
	# 	np_example = _make_seq_mask( np_example )
	# 	# if pair_msa_sequences:
	# 	# 	np_example = _concatenate_paired_and_unpaired_features( np_example )
	# 	# np_example = _correct_post_merged_feats(
	# 	# 		np_example = np_example,
	# 	# 		np_chains_list = np_chains_list,
	# 	# 		pair_msa_sequences = pair_msa_sequences )

	# 	return np_example


	# def process_mmcif( self,
	# 		mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
	# 		# alignment_dir: str,
	# 		# alignment_index: Optional[Any] = None,
	# ) -> FeatureDict:
	# 	"""
	# 	Taken from openfold.data.data_pipeline.py
	# 	This function extracts the following features from the structure:
	# 		auth_chain_id, 
	# 		sequence
	# 		aatype
	# 		between_segment_residues
	# 		domain_name
	# 		residue_index
	# 		seq_length
	# 		all_atom_positions
	# 		all_atom_mask
	# 		resolution
	# 		release_date
	# 		is_distillation
	# 		asym_id
	# 		sym_id
	# 		entity_id

	# 	Input:
	# 	----------
	# 	mmcif --> an instance of MmcifObject containing relevant info. extracted from the .cif file.

	# 	Returns:
	# 	----------
	# 	np_example --> (dict) contains all the above listed features.
	# 	"""
	# 	all_chain_features = {}
	# 	sequence_features = {}
	# 	# Check if the input struct is a homomer or monomer.
	# 	is_homomer_or_monomer = len(set(list(mmcif.chain_to_seqres.values()))) == 1
		
	# 	# extract the required features for all chains.
	# 	for chain_id, seq in mmcif.chain_to_seqres.items():
	# 		# identifier for each chain e.g. 2AYO_A.
	# 		desc= "_".join([mmcif.file_id, chain_id])

	# 		chain_features = {"sequence": seq}

	# 		# For homomeric chains, do not extract features multiple times.
	# 		if seq in sequence_features:
	# 		    all_chain_features[desc] = copy.deepcopy(
	# 		        sequence_features[seq]
	# 		    )
	# 		    continue

	# 		# Create a dict of sequence features for each chain: 
	# 		# 		aatype, between_segment_residues, domain_name, residue_index, seq_length, sequence.
	# 		seq_feats = make_sequence_features(
	# 					sequence = seq,
	# 					description = desc,
	# 					num_res = len( seq ),
	# 					)

	# 		chain_features.update( seq_feats )
	# 		# if alignment_index is not None:
	# 		#     chain_alignment_index = alignment_index.get(desc)
	# 		#     chain_alignment_dir = alignment_dir
	# 		# else:
	# 		#     chain_alignment_index = None
	# 		#     chain_alignment_dir = os.path.join(alignment_dir, desc)

	# 		# chain_features = self._process_single_chain(
	# 		#     chain_id=desc,
	# 		#     sequence=seq,
	# 		#     description=desc,
	# 		#     chain_alignment_dir=chain_alignment_dir,
	# 		#     chain_alignment_index=chain_alignment_index,
	# 		#     is_homomer_or_monomer=is_homomer_or_monomer
	# 		# )

	# 		chain_features = self.convert_monomer_features(
	# 		    chain_features,
	# 		    chain_id=desc
	# 		)

	# 		mmcif_feats = self.get_mmcif_features(mmcif, chain_id)
			
	# 		chain_features.update( mmcif_feats )
	# 		all_chain_features[desc] = chain_features
	# 		sequence_features[seq] = chain_features

	# 	all_chain_features = add_assembly_features( all_chain_features )

	# 	# np_example = feature_processing_multimer.pair_and_merge(
	# 	# 	all_chain_features=all_chain_features,
	# 	# )

	# 	np_chains_list = list( all_chain_features.values() )
	# 	# # Pad MSA to avoid zero-sized extra_msa.
	# 	np_example = self.merge_chain_features(
	# 				np_chains_list = np_chains_list
	# 				# pair_msa_sequences = pair_msa_sequences,
	# 				# max_templates = MAX_TEMPLATES
	# 				)
	# 	# np_example = pad_msa(np_example, 512)

	# 	return np_example


	# def make_data_config( self,
	#     config: ml_collections.ConfigDict,
	#     mode: str,
	#     num_res: int,
	# ) -> Tuple[ml_collections.ConfigDict, List[str]]:
	# 	"""
	# 	Taken from openfold.data.feature_pipeline.py
	# 	"""
	# 	cfg = copy.deepcopy(config)
	# 	mode_cfg = cfg[mode]
	# 	with cfg.unlocked():
	# 		if mode_cfg.crop_size is None:
	# 			mode_cfg.crop_size = num_res

	# 	feature_names = cfg.common.unsupervised_features

	#     # Add seqemb related features if using seqemb mode.
	# 	if cfg.seqemb_mode.enabled:
	# 		feature_names += cfg.common.seqemb_features

	# 	if cfg.common.use_templates:
	# 		feature_names += cfg.common.template_features

	# 	if cfg[mode].supervised:
	# 		feature_names += cfg.supervised.supervised_features

	# 	return cfg, feature_names



	# def process_tensors_from_config( self, tensors, common_cfg, mode_cfg ):
	# 	"""
	# 	Taken from openfold.data.feature_pipeline.py
	# 	"""
	# 	"""Based on the config, apply filters and transformations to the data."""

	# 	process_gt_feats = mode_cfg.supervised
	# 	gt_tensors = {}
	# 	# if process_gt_feats:
	# 	gt_tensors = prepare_ground_truth_features( tensors )

	# 	# ensemble_seed = random.randint(0, torch.iinfo(torch.int32).max)
	# 	# tensors['aatype'] = tensors['aatype'].to(torch.long)
	# 	# nonensembled = nonensembled_transform_fns()
	# 	# tensors = compose(nonensembled)(tensors)
	# 	# if("no_recycling_iters" in tensors):
	# 	#     num_recycling = int(tensors["no_recycling_iters"])
	# 	# else:
	# 	#     num_recycling = common_cfg.max_recycling_iters

	# 	# def wrap_ensemble_fn(data, i):
	# 	#     """Function to be mapped over the ensemble dimension."""
	# 	#     d = data.copy()
	# 	#     fns = ensembled_transform_fns(
	# 	#         common_cfg, 
	# 	#         mode_cfg, 
	# 	#         ensemble_seed,
	# 	#     )
	# 	#     fn = compose(fns)
	# 	#     d["ensemble_index"] = i
	# 	#     return fn(d)

	# 	# tensors = map_fn(
	# 	#     lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
	# 	# )

	# 	# if process_gt_feats:
	# 	tensors['gt_features'] = gt_tensors

	# 	return tensors


	# def np_example_to_features( self,
	#     np_example: FeatureDict,
	#     config: ml_collections.ConfigDict,
	#     mode: str,
	#     is_multimer: bool = False ):
	#     np_example = dict( np_example )

	#     seq_length = np_example["seq_length"]
	#     num_res = int( seq_length[0] ) if seq_length.ndim != 0 else int( seq_length )
	#     cfg, feature_names = self.make_data_config( config, mode = mode, num_res = num_res )
	 
	#     # if "deletion_matrix_int" in np_example:
	#     #     np_example["deletion_matrix"] = np_example.pop(
	#     #         "deletion_matrix_int"
	#     #     ).astype(np.float32)

	#     tensor_dict = np_to_tensor_dict(
	#         np_example = np_example, features = feature_names
	#     )

	#     with torch.no_grad():
	#         # if is_multimer:
	#         features = self.process_tensors_from_config(
	# 			            tensor_dict,
	# 			            cfg.common,
	# 			            cfg[mode],
	# 			        )
	#         # else:
	#         #     features = input_pipeline.process_tensors_from_config(
	#         #         tensor_dict,
	#         #         cfg.common,
	#         #         cfg[mode],
	#         #     )

	#     # if mode == "train":
	#     #     p = torch.rand(1).item()
	#     #     use_clamped_fape_value = float(p < cfg.supervised.clamp_prob)
	#     #     features["use_clamped_fape"] = torch.full(
	#     #         size=[cfg.common.max_recycling_iters + 1],
	#     #         fill_value=use_clamped_fape_value,
	#     #         dtype=torch.float32,
	#     #     )
	#     # else:
	#     #     features["use_clamped_fape"] = torch.full(
	#     #         size=[cfg.common.max_recycling_iters + 1],
	#     #         fill_value=0.0,
	#     #         dtype=torch.float32,
	#     #     )

	#     return {k: v for k, v in features.items()}




if __name__ == "__main__":
	SystemRepresentation().forward()

