"""
Given an AF2 prediction, define rigid bodies
	Split chains into rigid bodies.
	Based on pLDDT and/or PAE.
"""
from typing import List, Tuple

import torch

def get_rigid_chains(
	final_atom_positions: torch.Tensor,
	asym_id: torch.Tensor
	) -> Tuple[List[torch.Tensor], torch.Tensor]:
	"""
	Split the chains in the predicted
		structure into rigid bodies.
	Return a list of all rigid bodies and
		a tensor with mean positions for the rigid bodies.

	final_atom_posistion -> [1, N, 37, 3]
	"""
	rigid_bodies = []
	unique_asym_ids = torch.unique( asym_id )

	# asym_id -> [B, N]
	for a_id in unique_asym_ids:
		idx = torch.where( a_id == asym_id )[1]
		rb = final_atom_positions[:, idx, :, :]
		rigid_bodies.append( rb )

	init_mean_coords = []
	for rb in rigid_bodies:
		# [B, N, 37, 3] -> [B, 3]
		init_mean_coords.append(
			torch.mean( rb, dim = ( 1, 2 ) )
			)
	init_mean_coords  = torch.cat( init_mean_coords, dim = 0 )
	return rigid_bodies, init_mean_coords

