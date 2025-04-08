"""
Prepare the system to be modeled and create ground truth 
	features for the experimental data.
This script assumes a specific directory structure:
	base_dir --> e.g. benchmark/2ayo/
		System specific JSON file.
		Restraint input files --> e.g. For Xl restraint *_xls.csv.
"""
import os
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import ml_collections as mlc

import torch

from utils.utils import ( open_file_handler )


class DataGathering():
	"""
	Prepare the system to be modeled and create restraint features.
	"""
	def __init__( self, sys_name: str, base_dir: str,
						fasta_dir: str, sys_config: mlc.ConfigDict ):
		self.sys_name = sys_name
		# Main directory for the modeled system.
		self.base_dir = base_dir
		# Directory containing a FASTA file for the system to be modeled.
		self.fasta_dir = fasta_dir
		# FASTA file for the system to be modeled.
		self.fasta_file_path = f"{self.fasta_dir}/{self.sys_name}.fasta"

		self.sys_config = sys_config

		# Dict storing all required restraint features.
		self.restraint_features = {}


	def forward( self ):
		"""
		Given the system specific configs,
			1. Create the full system containing all chains to be modeled.
			2. Create a directory containing FASTA file for input to OpenFold.
			3. Create a mapping for residue positions to system indices.
			4. Create a mapping for entity to chains.
			5. Create restraint features dict.
		"""
		system_dict = self.create_full_system()
		self.create_fasta_input( system_dict )
		self.res_idx_map = self.create_residue_index_mapping( system_dict )
		self.entity_chain_map = self.create_entity_chain_mapping( system_dict )
		self.chain_entity_map = self.create_chain_entity_mapping( system_dict )
		self.create_xl_restraint_features( system_dict )


	#====================================================================#
	#====================================================================#
	def get_chain_id( self, idx: int ) -> str:
		"""
		Get a chain ID based on an index.
		"""
		alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

		if idx < len( alphabet ):
			chain_id = alphabet[idx]

		else:
			raise ValueError( "Too many chains..." )
		return chain_id


	def create_full_system( self ) -> Dict[str, Dict]:
		"""
		Given the entities in the system configs and the copy no.,
			create all chains in the system (dict) to be modeled.
		Each chain in the system is identified by a sys_chain_id:
			"{sys_name}_{entity_id}_{chain_id}"
		"""
		system_dict = {}
		idx = 0
		for entity in self.sys_config.entity:
			entity_id = entity["entity_id"]
			copy_num = entity["copy_num"]
			start = entity["start"]
			end = entity["end"]
			seq = entity["sequence"][start - 1: end]

			for i in range( copy_num ):
				chain_id = self.get_chain_id( idx )
				sys_chain_id = f"{self.sys_name}_{entity_id}_{chain_id}"

				system_dict[sys_chain_id] = {
											"seq": seq,
											"positions": [start, end]
				}

				idx += 1
		return system_dict


	#====================================================================#
	#====================================================================#
	def create_fasta_input( self, system_dict: Dict ):
		"""
		Given the full system to be modeled, create a 
			directory containig a FASTA file for OpenFold.
		"""
		if not os.path.exists( self.fasta_dir ):
			os.makedirs( self.fasta_dir, exist_ok = True )

		w = open_file_handler( self.fasta_file_path, "w" )
		for chain in system_dict.keys():
			w.writelines( f">{chain}\n{system_dict[chain]['seq']}\n" )
		w.close()


	#====================================================================#
	#====================================================================#
	def create_residue_index_mapping( self, system_dict: Dict
										) -> Dict[str, Dict[int, int]]:
		"""
		Create a mapping between the residue position to system indices and its inverse.
		Also map the residue positions to the corresponding chains.
		"""
		res_idx_map = {}

		sys_start = 0
		for chain in system_dict:
			# Intrapolate all residue positions between start and end.
			start, end = system_dict[chain]["positions"]
			residues_positions = np.arange( start, end+1, 1 )
			total = len( residues_positions )

			if total != len( system_dict[chain]["seq"] ):
				raise ValueError( "No. of residues and sequence length do not match..." )

			# Create the corresponding indices.
			sys_end = sys_start + total
			system_index = np.arange( sys_start, sys_end, 1 )

			sys_start = sys_end

			# Get the chain IDs for all residues.
			chain = chain.split( "_" )[-1]

			res_idx_map[chain] = {}
			res_idx_map[chain]["res_to_ind"] = dict(
												zip( residues_positions, system_index )
												)
			res_idx_map[chain]["ind_to_res"] = dict(
												zip( system_index, residues_positions )
												)

		return res_idx_map


	def create_entity_chain_mapping( self, system_dict: Dict
									) -> Dict[int, List]:
		"""
		Given the system_dict, get all chains for all entity_ids.
		"""
		entity_chain_map = {}
		for sys_chain_id in system_dict:
			_, entity_id, chain_id = sys_chain_id.split( "_" )
			entity_id = int( entity_id )
			if entity_id in entity_chain_map:
				entity_chain_map[entity_id].append( chain_id )
			else:
				entity_chain_map[entity_id] = [chain_id]

		return entity_chain_map


	def create_chain_entity_mapping( self, system_dict: Dict
									) -> Dict[int, List]:
		"""
		Given the system_dict, create a mapping between chains 
			to their respective entity_id.
		"""
		chain_entity_map = {}
		for sys_chain_id in system_dict:
			_, entity_id, chain_id = sys_chain_id.split( "_" )
			entity_id = int( entity_id )
			chain_entity_map[chain_id] = entity_id

		return chain_entity_map


	#====================================================================#
	#====================================================================#
	def parse_xl_data( self ) -> pd.DataFrame:
		"""
		Parse the .csv file containing the XL data.
		"""
		# xl_df = pd.read_csv( os.path.abspath( "2ayo_interprotein_xls.csv" ) )
		xl_csv_file = self.sys_config["data_gathering"]["xl_restraint"]["file_name"]
		xl_df = pd.read_csv( os.path.abspath( xl_csv_file ) )

		return xl_df


	def get_sys_len( self, system_dict: Dict ):
		"""
		Calculate the system length -- total residues in the system.
		"""
		sys_len = 0
		for chain in system_dict:
			sys_len += len( system_dict[chain]['seq'] )

		return sys_len


	def get_chains_for_entity( self, prot: str ) -> List:
		"""
		Given a protein name ("prot_{entity_id}"), return all chains for the entity.
		"""
		_, entity_id = prot.split( "_" )
		entity_id = int( entity_id )
		chains = self.entity_chain_map[entity_id]
		return chains


	def get_all_combinatorial_pairs( self, chains1: List, chains2: List ) -> List[Tuple]:
		"""
		Given lists of chain IDs for protein 1/2, create all 
			combinatorial pairs to account for ambiguity.
		We only consider inter-chain interactions.
		We assume that a XL between AB is the same as BA.
		"""
		pairs = []
		for c1 in chains1:
			for c2 in chains2:
				if c1 != c2:
					# Assuming the pair AB is the same as the pair BA.
					if ( c2, c1 ) not in pairs:
						pairs.append( ( c1, c2 ) )

		return pairs


	def account_for_ambiguity( self, xl_df: pd.DataFrame ) -> Dict:
		"""
		To account for ambiguity we consider all inter-chain 
			for each ambiguous interacting pair.
		For all XLs
			Get the entity_id for the cross-linked protein 1/2.
			Get all chain_ids belonging to entity_id for protein 1/2.
			Get all combinatorial pairs of inter-chain chain_ids.
			Get the residue position for cross-linked residues.
		The output is a dict contaiing all ambiguous pairs for
			each cross-linked residue pair.
		"""
		# Dict to store all the ambiguous XL pairs.
		# xl_amb = {k:[] for k in xl_df.columns}
		xl_amb_dict = {}
		for i in range( xl_df.shape[0] ):
			prot1 = xl_df.loc[ i, "prot1" ]
			chains1 = self.get_chains_for_entity( prot1 )
			prot2 = xl_df.loc[ i, "prot2" ]
			chains2 = self.get_chains_for_entity( prot2 )

			ambiguous_pairs = self.get_all_combinatorial_pairs( chains1, chains2 )

			res1 = xl_df.loc[ i, "res1" ]
			res2 = xl_df.loc[ i, "res2" ]

			xl_amb_dict[i] = {k:[] for k in xl_df.columns}
			for pair in ambiguous_pairs:
				c1, c2 = pair
				xl_amb_dict[i]["prot1"].append( c1 )
				xl_amb_dict[i]["res1"].append( res1 )
				xl_amb_dict[i]["prot2"].append( c2 )
				xl_amb_dict[i]["res2"].append( res2 )

		# xl_amb_df = pd.DataFrame( xl_amb )
		return xl_amb_dict


	def convert_amb_dict_to_df( sefl, xl_amb_dict: Dict ) -> Dict:
		"""
		Given the dict output by self.account_for_ambiguity(),
			flatten it and create a pd.DataFrame.
		Assuming that the residue positions have been mapped to system indices.
		"""
		flat_dict = {k:[] for k in ["prot1", "res1", "prot2", "res2"]}

		# For all XL'd residue pairs.
		for i in xl_amb_dict:
			amb_pairs = xl_amb_dict[i]
			# For all ambiguous XLs of a residue pair.
			for k in amb_pairs:
				flat_dict[k].extend( amb_pairs[k] )

		xl_amb_df = pd.DataFrame( flat_dict )
		return xl_amb_df


	def map_residue_to_index( self, xl_amb_dict: Dict ) -> Dict:
		"""
		Given the xl_amb_dict, convert all residue positions
			to system indices from 0 to N, where N is the
			total no. of residues in the system.
		"""
		xl_amb_dict_sys = {}
		for i in xl_amb_dict:
			xl_amb_dict_sys[i] = {k:[] for k in xl_amb_dict[i]}
			# For all ambiguous XLs of a residue pair.
			for j in range( len( xl_amb_dict[i]["prot1"] ) ):
				chain1 = xl_amb_dict[i]["prot1"][j]
				chain2 = xl_amb_dict[i]["prot2"][j]
				res1 = xl_amb_dict[i]["res1"][j]
				res2 = xl_amb_dict[i]["res2"][j]
				index1 = self.res_idx_map[chain1]["res_to_ind"][res1]
				index2 = self.res_idx_map[chain2]["res_to_ind"][res2]

				xl_amb_dict_sys[i]["prot1"].append( xl_amb_dict[i]["prot1"][j] )
				xl_amb_dict_sys[i]["res1"].append( index1 )
				xl_amb_dict_sys[i]["prot2"].append( xl_amb_dict[i]["prot2"][j] )
				xl_amb_dict_sys[i]["res2"].append( index2 )

		return xl_amb_dict_sys

		# for i in range( len( df ) ):
		# 	chain1 = df.loc[ i, "prot1" ]
		# 	res1 = df.loc[ i, "res1" ]
		# 	index1 = self.res_idx_map[chain1]["res_to_ind"][res1]
		# 	df.loc[ i, "res1" ] = index1

		# 	chain2 = df.loc[ i, "prot2" ]
		# 	res2 = df.loc[ i, "res2" ]
		# 	index2 = self.res_idx_map[chain2]["res_to_ind"][res2]
		# 	df.loc[ i, "res2" ] = index2

		# return df


	# def map_chain_to_protein( self, df: pd.DataFrame ) -> pd.DataFrame:
	# 	"""
	# 	Given the xl_amb_dict, convert all chain_ids to the
	# 		protein name defined as "prot_{entity_id}".
	# 	Not sure if this is required.
	# 	"""
	# 	for i in range( len( df ) ):
	# 		chain1 = df.loc[ i, "prot1" ]
	# 		entity_id1 = self.chain_entity_map[chain1]
	# 		prot1 = f"prot_{entity_id1}"
	# 		df.loc[ i, "prot1" ] = prot1

	# 		chain2 = df.loc[ i, "prot2" ]
	# 		entity_id2 = self.chain_entity_map[chain2]
	# 		prot2 = f"prot_{entity_id2}"
	# 		df.loc[ i, "prot2" ] = prot2

	# 	return df


	def create_xl_restraint_features( self, system_dict ):
		"""
		Create ground truth contact map (binary) for XL restraint.
			We account for ambiguous cross-links in a single contact map.
		Parse the XLs file.
		Add ambiguous XL pairs where required.
		Map all cross-linkd residue positions to system indices.
		Create a 0s-matrix for the system.
		Add 1s for all cross-linked pairs.
		XL restraint features:
			binary mask for XL'd residues.
			a dict containing system indices for all XL apirs.
			max bound for the cross-linker.
			distogram (not sure if needed).
		"""
		xl_config = self.sys_config.data_gathering.xl_restraint
		xl_max_bound = xl_config.xl_max_bound
		xl_df = self.parse_xl_data()
		print( "Total input XL pairs = ", len( xl_df ) )

		# Account for ambiguity.
		xl_amb_dict = self.account_for_ambiguity( xl_df )

		# Map residue positions to system indices.
		xl_amb_dict_sys = self.map_residue_to_index( xl_amb_dict )

		# Convert dict to df for XL mask creation.
		# xl_amb_df = self.convert_amb_dict_to_df( xl_amb_dict_sys )

		sys_len = self.get_sys_len( system_dict )
		# Create a 0-matrix for the XL-residue mask [r,r].
		# 	r -> total no. of residues.
		# xl_res_mask = torch.zeros( ( sys_len, sys_len ) )

		# r1 = xl_amb_df["res1"]
		# r2 = xl_amb_df["res2"]

		# Above diagonal.
		# xl_res_mask[r1, r2] = 1
		# Below diagonal.
		# xl_res_mask[r2, r1] = 1

		# contact_map = xl_res_mask.clone()
		# distogram = self.get_distogram( contact_map, xl_config.xl_max_bound )
		# distogram = distogram*xl_res_mask.squeeze( 0 ).unsqueeze( -1 )

		self.restraint_features["xl_restraint"] = {}
		# self.restraint_features["xl_restraint"]["xl_res_mask"] = xl_res_mask
		self.restraint_features["xl_restraint"]["xl_res_dict"] = xl_amb_dict_sys
		self.restraint_features["xl_restraint"]["xl_max_bound"] = xl_max_bound
		self.restraint_features["xl_restraint"]["total_xls"] = len( xl_amb_dict_sys )
		# self.restraint_features["xl_restraint"]["gt_distogram"] = distogram


		# # Account for ambiguity.
		# xl_amb_df = self.account_for_ambiguity( xl_df )

		# # Map residue positions to system indices.
		# xl_amb_df = self.map_residue_to_index( xl_amb_df )
		# print( "Total XL pairs (with ambiguity) = ", len( xl_amb_df ) )

		# sys_len = self.get_sys_len( system_dict )
		# # Create a 0-matrix for the XL-residue mask [r,r].
		# # 	r -> total no. of residues.
		# xl_res_mask = torch.zeros( ( sys_len, sys_len ) )

		# r1 = xl_amb_df["res1"]
		# r2 = xl_amb_df["res2"]

		# # Above diagonal.
		# xl_res_mask[r1, r2] = 1
		# # Below diagonal.
		# # xl_res_mask[r2, r1] = 1

		# contact_map = xl_res_mask.clone()
		# distogram = self.get_distogram( contact_map, xl_config.xl_max_bound )
		# distogram = distogram*xl_res_mask.squeeze( 0 ).unsqueeze( -1 )

		# self.restraint_features["xl_restraint"] = {}
		# self.restraint_features["xl_restraint"]["xl_res_mask"] = xl_res_mask
		# self.restraint_features["xl_restraint"]["gt_distogram"] = distogram
		# self.restraint_features["xl_restraint"]["xl_max_bound"] = xl_max_bound


	def get_distogram( self, contact_map: torch.Tensor, xl_max_bound: float ):
		"""
		Given a XL contact map and xl max bound, create a distogram.
		"""
		# These are taken from openFold.utils.loss.distogram_loss().
		min_bin = 2.3125
		max_bin = 21.6875
		no_bins = 64

		boundaries = torch.linspace(
			min_bin,
			max_bin,
			no_bins - 1 )

		idx = torch.where( contact_map == 1 )
		contact_map[idx] = xl_max_bound

		# 63 bins for min to max distance and 1 for the rest.
		true_bins = torch.sum( contact_map.unsqueeze( -1 ) > boundaries, dim = -1 )

		distogram = torch.nn.functional.one_hot( true_bins, no_bins ).float()

		return distogram

