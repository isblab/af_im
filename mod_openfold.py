import numpy as np
import ml_collections
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFIO
from Bio.Data import PDBData
import io
import copy

import torch

from openfold.data import ( data_pipeline, feature_pipeline, mmcif_parsing, msa_pairing )
from openfold.data.data_pipeline import ( add_assembly_features, make_sequence_features )
from openfold.data.msa_pairing import ( _correct_post_merged_feats, _merge_homomers_dense_msa,
					_merge_features_from_multiple_chains )
from openfold.data.mmcif_parsing import (
		_get_first_model, _get_protein_chains,
		_get_atom_site_list, ParsingResult, MmcifObject,
		ResidueAtPosition, ResiduePosition,
		mmcif_loop_to_list, _is_set )
from openfold.data.input_pipeline_multimer import prepare_ground_truth_features
from openfold.data.feature_processing_multimer import _make_seq_mask
from openfold.data.feature_pipeline import np_to_tensor_dict

from typing import Mapping, Optional, MutableMapping, Dict, List, Tuple, Sequence, Any
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]
FeatureDict = MutableMapping[str, np.ndarray]


# Taken from openfold.data.data_pipeline.DataPipelineMultimer()
######################################################################
def get_mmcif_features( 
        mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
) -> FeatureDict:
    mmcif_feats = {}

    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    # All atom coordinates in atom37 representations [L, 37, 3].
    mmcif_feats["all_atom_positions"] = all_atom_positions
    # Binary all atom mask in atom37 representations  [L, 37, 3].
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        mmcif_object.header["resolution"], dtype=np.float32
    )

    # mmcif_feats["release_date"] = np.array(
    #     [mmcif_object.header["release_date"].encode("utf-8")], dtype=object
    # )

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats



# Taken from openfold.msa_pairing.py
######################################################################
def convert_monomer_features( 
    monomer_features: FeatureDict,
    chain_id: str
	) -> FeatureDict:
	"""Reshapes and modifies monomer features for multimer models."""
	converted = {}
	converted["auth_chain_id"] = np.asarray( chain_id, dtype = object )
	unnecessary_leading_dim_feats = {
		"sequence", "domain_name", "num_alignments", "seq_length"
	}
	for feature_name, feature in monomer_features.items():
		if feature_name in unnecessary_leading_dim_feats:
			# asarray ensures it's a np.ndarray.
			feature = np.asarray( feature[0], dtype = feature.dtype )
		elif feature_name == "aatype":
			# The multimer model performs the one-hot operation itself.
			feature = np.argmax( feature, axis = -1 ).astype( np.int32 )
		# We don't need this, but just left it as is.
		elif feature_name == "template_aatype":
			feature = np.argmax( feature, axis = -1 ).astype( np.int32 )
			new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
			feature = np.take( new_order_list, feature.astype( np.int32 ), axis = 0 )
		# We don't need this, but just left it as is.
		elif feature_name == "template_all_atom_masks":
			feature_name = "template_all_atom_mask"
		converted[feature_name] = feature
	return converted



# Taken from openfold.msa_pairing.py
######################################################################
def merge_chain_features( np_chains_list: List[Mapping[str, np.ndarray]]
						# pair_msa_sequences: bool, max_templates: int
						) -> Mapping[str, np.ndarray]:

	"""Merges features for multiple chains to single FeatureDict.

	Args:
		np_chains_list: List of FeatureDicts for each chain.
		pair_msa_sequences: Whether to merge paired MSAs.
		max_templates: The maximum number of templates to include.

	Returns:
	Single FeatureDict for entire complex.
	"""
	# np_chains_list = _pad_templates(
	# 	np_chains_list, max_templates=max_templates)
	np_chains_list = _merge_homomers_dense_msa( np_chains_list )
	# Unpaired MSA features will be always block-diagonalised; paired MSA
	# features will be concatenated.
	np_example = _merge_features_from_multiple_chains(
					np_chains_list, pair_msa_sequences = False )
	np_example = _make_seq_mask( np_example )
	# if pair_msa_sequences:
	# 	np_example = _concatenate_paired_and_unpaired_features( np_example )
	# np_example = _correct_post_merged_feats(
	# 		np_example = np_example,
	# 		np_chains_list = np_chains_list,
	# 		pair_msa_sequences = pair_msa_sequences )

	return np_example


# Taken from openfold.data.data_pipeline.py
######################################################################
def process_mmcif( 
		mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
		# alignment_dir: str,
		# alignment_index: Optional[Any] = None,
) -> FeatureDict:
	"""
	-Kartik-	
	This function extracts the following features from the structure:
		auth_chain_id, 
		sequence
		aatype
		between_segment_residues
		domain_name
		residue_index
		seq_length
		all_atom_positions
		all_atom_mask
		resolution
		release_date
		is_distillation
		asym_id
		sym_id
		entity_id

	Input:
	----------
	mmcif --> an instance of MmcifObject containing relevant info. extracted from the .cif file.

	Returns:
	----------
	np_example --> (dict) contains all the above listed features.
	"""
	all_chain_features = {}
	sequence_features = {}
	# Check if the input struct is a homomer or monomer.
	is_homomer_or_monomer = len(set(list(mmcif.chain_to_seqres.values()))) == 1
	
	# extract the required features for all chains.
	for chain_id, seq in mmcif.chain_to_seqres.items():
		# identifier for each chain e.g. 2AYO_A.
		desc= "_".join([mmcif.file_id, chain_id])

		chain_features = {"sequence": seq}

		# For homomeric chains, do not extract features multiple times.
		if seq in sequence_features:
		    all_chain_features[desc] = copy.deepcopy(
		        sequence_features[seq]
		    )
		    continue

		# Create a dict of sequence features for each chain: 
		# 		aatype, between_segment_residues, domain_name, residue_index, seq_length, sequence.
		seq_feats = make_sequence_features(
					sequence = seq,
					description = desc,
					num_res = len( seq ),
					)

		chain_features.update( seq_feats )
		# if alignment_index is not None:
		#     chain_alignment_index = alignment_index.get(desc)
		#     chain_alignment_dir = alignment_dir
		# else:
		#     chain_alignment_index = None
		#     chain_alignment_dir = os.path.join(alignment_dir, desc)

		# chain_features = self._process_single_chain(
		#     chain_id=desc,
		#     sequence=seq,
		#     description=desc,
		#     chain_alignment_dir=chain_alignment_dir,
		#     chain_alignment_index=chain_alignment_index,
		#     is_homomer_or_monomer=is_homomer_or_monomer
		# )

		chain_features = convert_monomer_features(
		    chain_features,
		    chain_id=desc
		)

		mmcif_feats = get_mmcif_features(mmcif, chain_id)
		
		chain_features.update( mmcif_feats )
		all_chain_features[desc] = chain_features
		sequence_features[seq] = chain_features

	all_chain_features = add_assembly_features( all_chain_features )

	# np_example = feature_processing_multimer.pair_and_merge(
	# 	all_chain_features=all_chain_features,
	# )

	np_chains_list = list( all_chain_features.values() )
	# # Pad MSA to avoid zero-sized extra_msa.
	np_example = merge_chain_features(
				np_chains_list = np_chains_list
				# pair_msa_sequences = pair_msa_sequences,
				# max_templates = MAX_TEMPLATES
				)
	# np_example = pad_msa(np_example, 512)

	return np_example


# Taken from openfold.data.feature_pipeline.py
######################################################################
def make_data_config( 
    config: ml_collections.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
	cfg = copy.deepcopy(config)
	mode_cfg = cfg[mode]
	with cfg.unlocked():
		if mode_cfg.crop_size is None:
			mode_cfg.crop_size = num_res

	feature_names = cfg.common.unsupervised_features

    # Add seqemb related features if using seqemb mode.
	if cfg.seqemb_mode.enabled:
		feature_names += cfg.common.seqemb_features

	if cfg.common.use_templates:
		feature_names += cfg.common.template_features

	if cfg[mode].supervised:
		feature_names += cfg.supervised.supervised_features

	return cfg, feature_names


# Taken from openfold.data.feature_pipeline.py
######################################################################
def process_tensors_from_config( tensors, common_cfg, mode_cfg ):
	"""Based on the config, apply filters and transformations to the data."""

	process_gt_feats = mode_cfg.supervised
	gt_tensors = {}
	# if process_gt_feats:
	gt_tensors = prepare_ground_truth_features( tensors )

	# ensemble_seed = random.randint(0, torch.iinfo(torch.int32).max)
	# tensors['aatype'] = tensors['aatype'].to(torch.long)
	# nonensembled = nonensembled_transform_fns()
	# tensors = compose(nonensembled)(tensors)
	# if("no_recycling_iters" in tensors):
	#     num_recycling = int(tensors["no_recycling_iters"])
	# else:
	#     num_recycling = common_cfg.max_recycling_iters

	# def wrap_ensemble_fn(data, i):
	#     """Function to be mapped over the ensemble dimension."""
	#     d = data.copy()
	#     fns = ensembled_transform_fns(
	#         common_cfg, 
	#         mode_cfg, 
	#         ensemble_seed,
	#     )
	#     fn = compose(fns)
	#     d["ensemble_index"] = i
	#     return fn(d)

	# tensors = map_fn(
	#     lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
	# )

	# if process_gt_feats:
	tensors['gt_features'] = gt_tensors

	return tensors


# Taken from openfold.data.feature_pipeline.py
######################################################################
def np_example_to_features( 
    np_example: FeatureDict,
    config: ml_collections.ConfigDict,
    mode: str,
    is_multimer: bool = False ):
    np_example = dict( np_example )

    seq_length = np_example["seq_length"]
    num_res = int( seq_length[0] ) if seq_length.ndim != 0 else int( seq_length )
    cfg, feature_names = make_data_config( config, mode = mode, num_res = num_res )
 
    # if "deletion_matrix_int" in np_example:
    #     np_example["deletion_matrix"] = np_example.pop(
    #         "deletion_matrix_int"
    #     ).astype(np.float32)

    tensor_dict = np_to_tensor_dict(
        np_example = np_example, features = feature_names
    )

    with torch.no_grad():
        # if is_multimer:
        features = process_tensors_from_config(
			            tensor_dict,
			            cfg.common,
			            cfg[mode],
			        )
        # else:
        #     features = input_pipeline.process_tensors_from_config(
        #         tensor_dict,
        #         cfg.common,
        #         cfg[mode],
        #     )

    # if mode == "train":
    #     p = torch.rand(1).item()
    #     use_clamped_fape_value = float(p < cfg.supervised.clamp_prob)
    #     features["use_clamped_fape"] = torch.full(
    #         size=[cfg.common.max_recycling_iters + 1],
    #         fill_value=use_clamped_fape_value,
    #         dtype=torch.float32,
    #     )
    # else:
    #     features["use_clamped_fape"] = torch.full(
    #         size=[cfg.common.max_recycling_iters + 1],
    #         fill_value=0.0,
    #         dtype=torch.float32,
    #     )

    return {k: v for k, v in features.items()}




# Taken from openfold.data.mmcif_parsing.py
######################################################################
def parse(
    *, file_id: str, mmcif_string: str, catch_all_errors: bool = True
) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    Returns:
      A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure("", handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(
                None, {(file_id, ""): "No protein chains found in this file."}
            )
        seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, seq in valid_chains.items()
        }

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in valid_chains:
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                position = ResiduePosition(
                    chain_id=atom.author_chain_id,
                    residue_number=int(atom.author_seq_num),
                    insertion_code=insertion_code,
                )
                seq_idx = (
                    int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                )
                current = seq_to_structure_mappings.get(
                    atom.author_chain_id, {}
                )
                current[seq_idx] = ResidueAtPosition(
                    position=position,
                    name=atom.residue_name,
                    is_missing=False,
                    hetflag=hetflag,
                )
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(
                        position=None,
                        name=monomer.id,
                        is_missing=True,
                        hetflag=" ",
                    )

        author_chain_to_sequence = {}
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            seq = []
            for monomer in seq_info:
                code = PDBData.protein_letters_3to1.get(monomer.id, "X")
                seq.append(code if len(code) == 1 else "X")
            seq = "".join(seq)
            author_chain_to_sequence[author_chain] = seq

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            raw_string=parsed_info,
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)



# Taken from openfold.data.mmcif_parsing.py
######################################################################
def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list("_exptl.", parsed_info)
    header["structure_method"] = ",".join(
        [experiment["_exptl.method"].lower() for experiment in experiments]
    )

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    # if "_pdbx_audit_revision_history.revision_date" in parsed_info:
    #     header["release_date"] = get_release_date(parsed_info)
    # else:
    #     logging.warning(
    #         "Could not determine release_date: %s", parsed_info["_entry.id"]
    #     )

    header["resolution"] = 0.00
    # for res_key in (
    #     "_refine.ls_d_res_high",
    #     "_em_3d_reconstruction.resolution",
    #     "_reflns.d_resolution_high",
    # ):
    #     if res_key in parsed_info:
    #         try:
    #             raw_resolution = parsed_info[res_key][0]
    #             header["resolution"] = float(raw_resolution)
    #             break
    #         except ValueError:
    #             logging.debug(
    #                 "Invalid resolution format: %s", parsed_info[res_key]
    #             )

    return header
