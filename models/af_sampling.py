"""
Contains methods for perturbing AF features or representations
	to enable AF sampling.
It include:
	MSA subsampling
	MSA Column masking
	Dropouts on MSA representation
"""
from typing import List

import torch


def get_eff( msa, eff_cutoff = 0.8): # eff_cutoff=0.62 for metapsicov
     
	if msa.ndim == 3: msa = msa.argmax(-1)
	# pairwise identity  
	msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
	# weight for each sequence
	msa_w = (msa_sm >= eff_cutoff).astype(float)
	msa_w = 1/np.sum(msa_w,-1)

	return msa_w

def subsample_msa_random(
        msa: torch.Tensor,
        neff: int,
        eff_cutoff: float ):
    if msa.shape[0] == 1:
        return msa

    weights = get_eff( msa, eff_cutoff = eff_cutoff )
    
    current_neff = weights[0]
    
    pick = [msa[0]]
    
    msa = msa[1:]
    weights = weights[1:]
    
    idx = np.arange(msa.shape[0])
    np.random.shuffle(idx)
    weights = weights[idx]
    msa = msa[idx]
    
    for i, w in enumerate(weights):
        if w + current_neff > neff:
            break
        pick.append(msa[i])
        current_neff += w
        
    return np.array(pick)