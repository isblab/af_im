"""
"""
from typing import List, Dict, Any
import numpy as np

from utils.utils import read_json
from utils.paths import get_sys_config_path


def get_entities_in_system(
	base_dir: str,
	benchmark_name: str,
	sys_name: str ) -> List[Dict[str, Any]]:
	"""
	Parse the sys_config file and return the List of entities in the system.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	Returns:
	----------
	entities: Aa list of entity dictionaries as defined
		in the system config file.
	"""
	sys_config_path = get_sys_config_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	sys_conf = read_json( sys_config_path )
	# with open( sys_config_path, "r" ) as f:
	# 	sys_conf = json.load( f )
	entities = sys_conf["entity"]
	return entities


def get_entity_chain_mapping(
	base_dir: str,
	benchmark_name: str,
	sys_name: str ) -> Dict[int, Dict]:
	"""
	Map all entities to the corresponding chains.
	Each entity can have multiple chains.
		For each copy a new 1-indexed chain ID is created.
			asym_id in OpenFold feature-dic are 1-indexed.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	Returns:
	----------
	entity_id: {
		seq: str,
		chains: [int],
		residues: np.ndarray,
	}
	start,end residue positions are based on the PDB seq_id numbering.
		May not always have residues from 1.
	"""
	entities = get_entities_in_system(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	# GRASP expects numeric chain IDs.
	chain_id = 1
	entity_chain_map = {}
	for entity_id, entity in enumerate( entities, start = 1 ):
		entity_chain_map[entity_id] = {"seq": "", "chains": [], "residues": []}
		for cp in range( entity["copy_num"] ):
			entity_chain_map[entity_id]["seq"] = entity["sequence"]
			entity_chain_map[entity_id]["chains"].append( chain_id )
			entity_chain_map[entity_id]["residues"] = np.arange(
				entity["start"], entity["end"] + 1
				)
			chain_id += 1
	return entity_chain_map


################################################################################
################################################################################
def map_residue_positions_to_system_indices(
	base_dir: str,
	benchmark_name: str,
	sys_name: str ) -> Dict[str, Dict]:
	"""
	Given a system, create a mapping between chain-specific residue
		positions and globally unique system indices.
	residue positions are tied to each chain.
	system indices are 0-indexed, contiguous, and unique for the entire system range(0-N-1);
		where N is the no. of residues in the system.
	Chains are assumed to be in order: 1,2,..., or A,B,...

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	returns:
	sys_index_res_pos_map = {
		chain:{
			"res_to_ind": {int: int},
			"ind_to_res": {int: int}
		}
	}
	"""
	entity_chain_map = get_entity_chain_mapping(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )

	sys_index_res_pos_map = {}
	sys_ind_start = 0
	for entity_id in entity_chain_map:
		seq = entity_chain_map[entity_id]["seq"]
		residues = entity_chain_map[entity_id]["residues"]
		for chain_id in entity_chain_map[entity_id]["chains"]:
			sys_ind_end = sys_ind_start + len( residues )
			sys_indices = np.arange( sys_ind_start, sys_ind_end, 1 )
			sys_index_res_pos_map[chain_id] = {
				"res_to_ind": dict( zip( residues, sys_indices ) ),
				"ind_to_res": dict( zip( sys_indices, residues ) )
			}
			sys_ind_start = sys_ind_end
	return sys_index_res_pos_map

