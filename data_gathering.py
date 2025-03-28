import os
from typing import Dict
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import ml_collections as mlc

import torch

from pdb_utils import Parser
from utils import read_json, write_json



"""
This script assumes a specific directory structure:
	base_dir --> e.g. benchmark/2ayo/
		System specific JSON file.
		Restraint input files --> e.g. For Xl restraint *_xls.csv.
"""


class DataGathering():
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
			3. Create restraint features dict.
		"""
		system_dict = self.create_full_system()
		self.create_fasta_input( system_dict )
		self.create_xl_restraint_features( system_dict )



	def get_chain_id( self, idx: str ):
		"""
		Get a chain ID based on an index.
		"""
		alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

		if idx < len( alphabet ):
			chain_id = alphabet[idx]
			return chain_id
		
		else:
			raise Exception( "Too many chains." )


	def create_full_system( self ):
		"""
		Given the entities in the system configs and the copy no.,
			create all chains in the system (dict) to be modeled.
		"""
		system_dict = {}
		idx = 0
		for entity in self.sys_config.entity:
			copy_num = entity["copy_num"]
			start = entity["start"]
			end = entity["end"]
			seq = entity["sequence"][start - 1: end]

			for i in range( copy_num ):
				chain_id = self.get_chain_id( idx )

				system_dict[f"{self.sys_name}_{entity}_{chain_id}"] = {
													"seq": seq,
													"positions": [start, end]
				}

				idx += 1
		return system_dict


	def create_fasta_input( self, system_dict: Dict ):
		"""
		Given the full system to be modeled, create a 
			directory containig a FASTA file for OpenFold.
		"""
		if not os.path.exists( self.fasta_dir ):
			os.makedirs( self.fasta_dir )

		with open( self.fasta_file_path, "w" ) as w:
			for chain in system_dict.keys():
				w.writelines( f">{chain}\n{system_dict[chain]['seq']}\n" )


	def parse_xl_data( self ):
		"""
		Parse the .csv file containing the XL data.
		"""
		# xl_df = pd.read_csv( os.path.abspath( "2ayo_interprotein_xls.csv" ) )
		xl_df = pd.read_csv( os.path.abspath( self.sys_config["data_gathering"]["xl_restraint"]["file_name"] ) )

		return xl_df


	def get_residue_index_map( self, system_dict: Dict ):
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
				raise Exception( "No. of residues and sequence length do not match..." )
			
			# Create the corresponding indices.
			sys_end = sys_start + total
			system_index = np.arange( sys_start, sys_end, 1 )

			sys_start = sys_end

			# Get the chain IDs for all residues.
			chain = chain.split( "_" )[-1]
			
			res_idx_map[chain] = {}
			res_idx_map[chain]["res_to_ind"] = dict( zip( residues_positions, system_index ) )
			res_idx_map[chain]["ind_to_res"] = dict( zip( system_index, residues_positions ) )

		return res_idx_map


	def map_residue_to_index( self, xl_df: pd.DataFrame, res_idx_map: Dict ):
		"""
		Given a DataFrame containing positions for cross-linked residues,
			map all residue positions to system indices from 0 to N,
			where N is the total no. of residues in the system.
		"""
		for i in range( len( xl_df ) ):
			chain1 = xl_df.loc[ i, "prot1" ]
			res1 = xl_df.loc[ i, "res1" ]
			index1 = res_idx_map[chain1]["res_to_ind"][res1]
			xl_df.loc[ i, "res1" ] = index1
			
			chain2 = xl_df.loc[ i, "prot2" ]
			res2 = xl_df.loc[ i, "res2" ]
			index2 = res_idx_map[chain2]["res_to_ind"][res2]
			xl_df.loc[ i, "res2" ] = index2

		return xl_df

;

	def get_sys_len( self, system_dict: Dict ):
		"""
		Calculate the system length -- total residues in the system.
		"""
		sys_len = 0
		for chain in system_dict:
			sys_len += len( system_dict[chain]['seq'] )

		return sys_len


	def create_xl_restraint_features( self, system_dict ):
		"""
		Parse the XLs file.
		Calculate and add offsets to each chain in the system.
		Create a binary mask indicating cross-linked (XL'd) residues.
		XL restraint features:
			binary mask for XL'd residues.
			max bound for the cross-linker.
		"""
		xl_config = self.sys_config.data_gathering.xl_restraint
		xl_max_bound = xl_config.xl_max_bound
		xl_df = self.parse_xl_data()

		# offset_dict, sys_len = self.calculate_offsets( system_dict )
		# xl_df = self.add_offsets( offset_dict, xl_df )

		sys_len = self.get_sys_len( system_dict )
		res_idx_map = self.get_residue_index_map( system_dict )

		xl_df = self.map_residue_to_index( xl_df, res_idx_map )
		print( "Total XL pairs = ", len( xl_df ) )

		# Create a 0-matrix for the XL-residue mask [r,r].
		# 	r -> total no. of residues.
		xl_mask = torch.zeros( ( sys_len, sys_len ) )

		r1 = xl_df["res1"]
		r2 = xl_df["res2"]

		# Above diagonal.
		xl_mask[r1, r2] = 1
		# Below diagonal.
		# xl_mask[r2, r1] = 1

		contact_map = xl_mask.clone()
		distogram = self.get_distogram( contact_map, xl_config.xl_max_bound )
		distogram = distogram*xl_mask.squeeze( 0 ).unsqueeze( -1 )
		
		self.restraint_features["xl_restraint"] = {}
		self.restraint_features["xl_restraint"]["xl_res_mask"] = xl_mask
		self.restraint_features["xl_restraint"]["gt_distogram"] = distogram
		self.restraint_features["xl_restraint"]["xl_max_bound"] = xl_max_bound


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

