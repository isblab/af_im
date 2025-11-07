"""
Contains methods for perturbing AF features or representations
	to enable AF sampling.
It include:
	MSA subsampling
        vanilla: randomly select sequences to be kept.
        random_similarity: randomly subsample sequences weighted by their smilarity.
	MSA Dropout
    MSA Column masking
	Dropouts on MSA representation

"""
from typing import Dict, Any
import numpy as np
from scipy.spatial.distance import pdist, squareform

import torch

from openfold.data.data_transforms_multimer import create_msa_feat


def msa_subsampler( msa: torch.Tensor,
		subsample_type: str,
		params: Dict[str, Any]
		) -> torch.Tensor:
	"""
	Subsample the MSA as per the required approach
		and return the indices of the subampled sequences.
	msa -> [B, S, N]
	"""
    # If only 1 sequence in MSA, return as is.
	if msa.shape[1] == 1:
		return torch.tensor( [0] )
	else:
		if subsample_type == "sequential":
			return subsample_sequentially(
				msa = msa.squeeze( 0 ),
				params = params )
		else:
			raise ValueError( "Incorrect subsampling method specified..." )


def get_eff( msa: torch.Tensor,
        eff_cutoff: float = 0.8
        ) -> torch.Tensor:
    """
    Compute weights for each sequence in the
        MSA based on the sequence similarity.
    Taken from DEERFold/openfold/data/msa_subsampling.py.
    """
    if msa.ndim == 3: msa = msa.argmax( -1 )
    # pairwise identity  
    msa_sm = 1.0 - squareform( pdist( msa, "hamming" ) )
    # weight for each sequence
    msa_w = ( msa_sm >= eff_cutoff ).astype( float )
    msa_w = 1/np.sum( msa_w,-1 )

    return msa_w


# if cap_msa is enabled, we bypass the ExtraMSAStack, helps with determinism for |MSA| < 128
def subsample_sequentially(
        msa: torch.Tensor,
        params: Dict[str, Any]
        ) -> torch.Tensor:
    """
    Subsample the MSA sequentially up to the desired neff
        weighted by their similarity.
    Adapted from DEERFold/openfold/data/msa_subsampling.py.
    """
    neff = params["neff"]
    eff_cutoff = params["eff_cutoff"]
    cap_msa = params["cap_msa"]

    subsampled_idx = [0]

    idx = np.arange( 1, msa.shape[0] )
    np.random.shuffle( idx )

    new = [msa[0,:]]

    for i in idx:
        new.append( msa[i,:] )
        subsampled_idx.append( i )
        neff_ = get_eff(
            np.array( new ),
            eff_cutoff = eff_cutoff ).sum()

        if cap_msa:
            if neff_ > neff or len( new ) > 126:
                new.pop()
                subsampled_idx.pop()
                break
        else:
            if neff_ > neff:
                new.pop()
                subsampled_idx.pop()
                break

    return torch.tensor( subsampled_idx )


def msa_column_masking(
	batch: Dict[str, torch.Tensor],
	params: Dict[str, Any] ):
	"""
	Implement MSA column masking.
	Replace a specified fraction of residues with the unknown token (21).
	Adapted from AlphaLink2.
	msa -> [1, S, N]
	deletion_matrix -> [1, S, N]
	"""
	N = batch["msa"].shape[-1]
	mask_frac = params["mask_frac"]

	num_mask = int( N*mask_frac )

	idx = np.arange( 0, N, 1 )
	mask_idx = torch.randperm( len( idx ) )[:num_mask]

	batch["msa"][:, 1:, mask_idx] = 21
	batch["deletion_matrix"][:, 1:, mask_idx] = 0

	batch = create_msa_feat( batch = batch )
	return batch, mask_idx


def mask_msa_for_xl_res(
	batch: Dict[str, torch.Tensor],
	xl_res_dict: Dict
	) -> Dict[str, torch.Tensor]:
	"""
	Mask cross-linked residues in MSA features.
	Adapter from AlphaLink2.
	msa -> [1, S, N]
	deletion_matrix -> [1, S, N]
	"""
	res_mask = []
	for xl_pair in xl_res_dict:
		res_idx1 = torch.tensor( xl_res_dict[xl_pair]["res1"] )
		res_idx2 = torch.tensor( xl_res_dict[xl_pair]["res2"] )

		res_mask.extend( [res_idx1, res_idx2] )

	res_mask = torch.tensor( res_mask )

	# Replacee cross-linked residue with the unknown token.
	batch["msa"][:, 1:, res_mask] = 21
	batch["deletion_matrix"][:, 1:, res_mask] = 0
	# batch['msa'][1:,j] = 21
	# batch["deletion_matrix"][:, 1:, j, :] = 0

	batch.pop( "msa_feat" )
	batch = create_msa_feat( batch = batch )

	return batch

################################################################################
################################################################################
def af_sampling( self,
	batch: Dict[str, torch.Tensor] ) -> Dict[str, torch.Tensor]:
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
