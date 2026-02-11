"""
Contains accessory functions for computing loss and metrics.
"""
import torch
import openfold.np.residue_constants as rc


def final_pred_to_dist_map(
	final_atom_pos: torch.Tensor,
	eps = float
	) -> torch.Tensor:
	"""
	Compute a distance map from the final_atom_positions.
	Using only Ca-coordinates for distance map calculation.

	Input:
	----------
	final_atom_pos --> xyz coordinates. Following representations are allowed:
				atom37 representtaion: [B,N,37,3] --> For 2ayo: [1,480,37,3]
				per-residue representtaion: [B,N,3] --> For 2ayo: [1,480,3]

	Returns:
	----------
	D --> Ca-distance map [B,N,N].
	"""
	# Extracting Ca-coordinates - index 1.
	# [B,N,3] --> For 2ayo: [1,480,3]
	if len( final_atom_pos.shape ) == 4:
		ca_idx = rc.atom_order["CA"]
		#[B, N, 37, 3] -> [B, N, 3]
		ca_pos = final_atom_pos[:, :, ca_idx, :]
	elif len( final_atom_pos.shape ) == 3:
		ca_pos = final_atom_pos
	else:
		raise ValueError( f"Incorrect shape of the tensor: {final_atom_pos.shape}..." )
	diff = ca_pos.unsqueeze( 2 ) - ca_pos.unsqueeze( 1 )
	D = torch.sqrt( 
					torch.sum( ( diff )**2, dim = -1 ) + eps
					)
	return D
