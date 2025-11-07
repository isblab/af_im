"""
Create template features for custom templates in mmcif format.
"""
from typing import Dict, Any
import os
import numpy as np
import torch
from openfold.data.data_pipeline import convert_monomer_features
from openfold.data.templates import get_custom_template_features
from openfold.np import protein

from utils.utils import read_json, parse_nested_dict, run_subprocess
from utils.pdb_utils import SaveModels, get_chain_id

def get_custom_template_feats_from_pred(
	sys_name: str,
	prot: protein.Protein,
	model_id: int,
	sys_config: Dict[str, str],
	tmp_dir_path: str ) -> Dict[str, torch.Tensor]:
	"""
	Obtain template features for a given prediction.
	Adapted from openfold/data/template -> get_custom_template_features().
	"""
	save = SaveModels(
		title = sys_name,
		output_format = "cif",
		ensemble_dir = tmp_dir_path,
		save_single_model = True )
	save.initialize_system()
	save.add_model( prot = prot, model_id = model_id )
	del save

	mmcif_path = os.path.join( tmp_dir_path, f"model_{model_id}.cif" )

	chain_idx = 0
	template_feats = {}
	for entity in sys_config["entity"]:
		for num in range( entity["copy_num"] ):
			chain_id = get_chain_id( chain_idx )
			query_sequence = entity["sequence"]
			chain_idx += 1

			# This will return a warning: "rrelease date not found" which can be ignored.
			templ_feats = get_custom_template_features(
				mmcif_path = mmcif_path,
				query_sequence = query_sequence,
				pdb_id = sys_name,
				chain_id = chain_id,
				kalign_binary_path = os.path.join( "/home/kartik/miniforge3/envs/il_ofold/bin/kalign" )
			)
			templ_feats = convert_monomer_features( templ_feats.features, chain_id )
			if template_feats == {}:
				template_feats = templ_feats
			else:
				for k in templ_feats:
					template_feats[k] = np.column_stack( [template_feats[k], templ_feats[k]] )

	return template_feats


def update_template_feats(
	batch: Dict[str, torch.Tensor],
	sys_name: str,
	prot: protein.Protein,
	model_id: int,
	sys_config: str,
	add_to_existing_templates: bool,
	device = str,
	tmp_dir_path: str = "./tmp/"
	) -> Dict[str, torch.Tensor]:
	"""
	Obtain template features for the prediction.
	Update the temnplate features in the batch dict.
	"""
	os.makedirs( tmp_dir_path, exist_ok = True )
	template_features = get_custom_template_feats_from_pred(
		sys_name = sys_name,
		prot = prot,
		model_id = model_id,
		sys_config = sys_config,
		tmp_dir_path = tmp_dir_path )
	
	cmd = ["rm", "-r", f"{tmp_dir_path}"]
	run_subprocess( cmd )

	for k in ["template_aatype", "template_all_atom_positions", "template_all_atom_mask"]:
		# Add to existing template feats.
		if add_to_existing_templates:
			t = torch.from_numpy( template_features[k] ).unsqueeze( 1 ).to( device )
			batch[k] = torch.cat( [batch[k], t], dim = 1 ).to( device )
		else:
			batch[k] = torch.from_numpy( template_features[k] ).unsqueeze( 0 ).to( device )
	return batch
