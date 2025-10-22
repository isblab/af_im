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
        if subsample_type == "random":
            return subsample_random(
                msa = msa.squeeze( 0 ),
                params = params )
        elif subsample_type == "sequential":
            return subsample_sequentially(
                msa = msa.squeeze( 0 ),
                params = params )


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

    indices = [0]
    subsampled_idx = [0]

    idx = np.arange( msa.shape[1] - 1 ) + 1
    np.random.shuffle( idx )

    new = [msa[0,:]]

    for i in idx:
        new.append( msa[i,:] )
        # indices.append( i )
        subsampled_idx.append( i )
        neff_ = get_eff(
            np.array( new ),
            eff_cutoff = eff_cutoff ).sum()

        if cap_msa:
            if neff_ > neff or len( new ) > 126:
                new.pop()
                # indices.pop()
                subsampled_idx.pop()
                break
        else:
            if neff_ > neff:
                new.pop()
                # indices.pop()
                subsampled_idx.pop()
                break

    # return np.array(indices)
    return torch.tensor( subsampled_idx )


def subsample_random(
        msa: torch.Tensor,
        params: Dict[str, Any]
        ) -> torch.Tensor:
    """
    Subsample the MSA at random up to the desired neff
        weighted by their similarity.
    Adapted from DEERFold/openfold/data/msa_subsampling.py.
    """
    neff = params["neff"]
    eff_cutoff = params["eff_cutoff"]

    weights = get_eff( msa, eff_cutoff = eff_cutoff )    
    current_neff = weights[0]
    
    subsampled_idx = [0]
    
    msa = msa[1:]
    weights = weights[1:]
    
    idx = np.arange( msa.shape[1] )
    np.random.shuffle( idx )
    weights = weights[idx]
    msa = msa[idx]
    
    for i, w in enumerate(weights):
        if w + current_neff > neff:
            break
        subsampled_idx.append( i )
        current_neff += w
        
    return torch.Tensor( subsampled_idx )


# def subample_vanilla(
#         msa: torch.Tensor,
#         params: Dict[str, Any] ) -> torch.Tensor:
#     """
#     Randomly subsample the MSA up to the desired fraction of sequences.
#     """
#     # Keep aside the 1st sequence.
#     idx = np.arange( msa.shape[1] )[1:]

#     # Get the effective no. of sequences to be subsampled.
#     neff = int( msa.shape[1]*params["neff_frac"] )

#     subsampled_idx = np.random.choice( idx, neff, replace = False )
#     # Add the index for the query (first) sequence.
#     subsampled_idx = np.insert( subsampled_idx, 0, 0 )

#     return torch.tensor( subsampled_idx )


# def subsample_weighted_random(
#         msa: torch.Tensor,
#         params: Dict[str, Any] ) -> torch.Tensor:
#     """
#     Subsample the MSA up to the desired neff
#         weighted by their similarity.
#     Adapted from DEERFold/openfold/data/msa_subsampling.py.
#     """
#     # Get the effective no. of sequences to be subsampled.
#     neff = int( msa.shape[1]*params["neff_frac"] )
#     eff_cutoff = params["eff_cutoff"]
#     # Get the weights for each sequence.
#     weights = get_eff( msa, eff_cutoff = eff_cutoff )

#     current_neff = weights[0]
#     # pick = [msa[0]]

#     msa = msa[1:]
#     weights = weights[1:]

#     idx = np.arange( 1, msa.shape[1]+1 )
#     np.random.shuffle( idx )
#     weights = weights[idx]
#     msa = msa[idx]
    
#     subsampled_idx = []
#     for i, w in enumerate( weights ):
#         if w + current_neff > neff:
#             break
#         # pick.append( msa[i] )
#         subsampled_idx.append( i )
#         current_neff += w
        
#     # Add the index for the query (first) sequence.
#     subsampled_idx.insert( 0, 0 )
#     return torch.tensor( subsampled_idx )

