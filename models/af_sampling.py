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
	params: Dict[str, Any] ) -> Dict[str, torch.Tensor]:
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
	# For small sequences.
	if num_mask <= 1:
		raise ValueError( f"Too few columns selected for masking with mask_frac={mask_frac}." +
			"For small sequences try higher values of mask_frac..." )

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


def noised_structure(
	out: Dict[str, Any],
	params: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Create a noised final_atom_positions by:
		Adding gaussian noise to final_atom_positions.
		Sampling final_atom_positions from a gaussian.
	"""
	final_atom_positions = out.pop( "final_atom_positions" ).cpu()
	mu = torch.full( final_atom_positions.shape, params["mu"] )
	sigma = torch.full( final_atom_positions.shape, params["sigma"] )
	noise = torch.normal( mu, sigma )

	# Add noise to the existing structure.
	if params["noise_struct"]:
		final_atom_positions += noise
	# Pure noise structure.
	else:
		final_atom_positions = noise
	out["final_atom_positions"] = final_atom_positions*out["final_atom_mask"].unsqueeze( -1 ).cpu()
	return out

