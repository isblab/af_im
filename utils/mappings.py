"""
Utilities methods for parsing system configs and constructing mappings
	between entities, chains, and residue indices.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from utils.utils import read_json
from utils.pdb_utils import get_chain_id
from utils.paths import get_sys_config_path



def yield_restraints(
	sys_name: str,
	xl_file: str,
	entity_chain_map: Dict[int, Dict],
	numeric_chain_ids: bool,
	return_seq_numbering: bool = True
	):
	"""
	A generator that yields restrained residue pairs.
	Accounts for ambiguity, by enumerating all chain combinations.

	Note:
	For homomeric complexes, different chains in the experimental
		structure may be missing different sets of residues.
		We select the residues to be modeled from only 1 of the chains.
			As a result some XLs may not be modeled.
			We ignore these XLs here.
	XLs have previously been mapped to the 1-indexed seq_id in .cif files.
		However, the residue positions for the modeled seq may or may not
			start from 1.
		So, we get the index for the the modeled residues.
		residue no = residue index + 1
	Sanity checks if the Xl'd residue is Lys or not.
		JWalk only rturns Lys-Lys XLs.

	Input:
	----------
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.
	xl_file: path to the .csv file containing XLs for the given system.
	entity_chain_map: dict containing a mapping between all
		the corresponding chains along with metadata, including
		entity sequence and residues positions.
	numeric_chain_id: 1-indexed numeric chain identifier.
	return_seq_numbering: if True, returns the XL res numbering based
		on the modeled sequence else returns the XL res numbering as
		in the XL file.

	Returns:
	----------
	A tuple containing entity_id, chain_id, residue position for
		the cross-linked residue pairs.
	"""
	xl_df = pd.read_csv( xl_file )

	for row in xl_df.iterrows():
		p1, p2 = row[1]["prot1"], row[1]["prot2"]
		r1, r2, label = row[1]["res1"], row[1]["res2"], row[1]["label"]
		r1, r2 = int( r1 ), int( r2 )

		entity_id1 = int( p1.split( "_" )[1] )
		entity_id2 = int( p2.split( "_" )[1] )

		# Get the residue indices.
		try:
			r1_idx = np.where( entity_chain_map[entity_id1]["residues"] == r1 )[0][0]
		except:
			continue

		try:
			r2_idx = np.where( entity_chain_map[entity_id2]["residues"] == r2 )[0][0]
		except:
			continue
		if return_seq_numbering:
			res1 = r1_idx + 1
			res2 = r2_idx + 1
		else:
			res1 = r1
			res2 = r2

		seq1 = entity_chain_map[entity_id1]["seq"]
		seq2 = entity_chain_map[entity_id2]["seq"]

		# For ambiguous XLs, we consider all combinations.
		for chain_id1 in entity_chain_map[entity_id1]["chains"]:
			if not numeric_chain_ids:
				chain_id1 = get_chain_id( chain_id1 - 1  ) # 0-indexed.
			for chain_id2 in entity_chain_map[entity_id2]["chains"]:
				if not numeric_chain_ids:
					chain_id2 = get_chain_id( chain_id2 - 1  ) # 0-indexed.
				# Skip intr-chain restraint.
				if chain_id1 == chain_id2:
					continue

				if seq1[r1_idx] != "K":
					raise ValueError( f"{sys_name}: Entity: {entity_id1}; " +
						f"Chain: {chain_id1}; residue {res1} is not a Lys..." )

				if seq2[r2_idx] != "K":
					raise ValueError( f"{sys_name}: Entity: {entity_id2}; " +
						f"Chain: {chain_id2}; residue {res2} is not a Lys..." )

				yield entity_id1, entity_id2, chain_id1, chain_id2, res1, res2, label

################################################################################
################################################################################
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
	entity_chain_map: dict containing a mapping between all
		the corresponding chains along with metadata, including
		entity sequence and residues positions.
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
		for chain in entity_chain_map[entity_id]["chains"]:
			chain_id = get_chain_id( chain-1 )
			sys_ind_end = sys_ind_start + len( residues )
			sys_indices = np.arange( sys_ind_start, sys_ind_end, 1 )
			sys_index_res_pos_map[chain_id] = {
				"res_to_ind": dict( zip( residues, sys_indices ) ),
				"ind_to_res": dict( zip( sys_indices, residues ) )
			}
			sys_ind_start = sys_ind_end
	return sys_index_res_pos_map

