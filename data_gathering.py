import numpy as np
import pandas as pd
import ml_collections as mlc
import os

from utils import read_json, write_json

from typing import Dict

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


	def forward( self ):
		"""
		Given the system specific configs,
			1. Create the full system containing all chains to be modeled.
			2. Create a directory containing FASTA file for input to OpenFold.
			3. Create restraint features dict.
		"""
		system = self.create_full_system()
		print( system )
		self.create_fasta_input( system )



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
			print( entity )
			copy_num = entity["copy_num"]
			for i in range( copy_num ):
				start = entity["start"]
				end = entity["end"]
				
				chain_id = self.get_chain_id( idx )
				seq = entity["sequence"]
				seq = seq[start - 1: end]

				system_dict[f"{self.sys_name}_{chain_id}"] = seq

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
				w.writelines( f">{chain}\n{system_dict[chain]}\n" )



	def add_residue_offset( self ):
		"""
		Add the offsets to residue positions based on the chain order in the system.
		"""
		self.topology_file = read_json( self.topology_file_path )



	def parse_xl_data( self, batch ):
		"""
		Parse the .csv file containing the XL data.
		"""
		df = pd.read_csv( os.path.abspath( "2ayo_interprotein_xls.csv" ) )

		r1, r2 = np.array( df["res1"] ), np.array( df["res2"] )
		r1, r2 = r1 - 1, r2 -1
		r2 += 404
		xl_dist = torch.zeros( ( 480, 480 ) )
		xl_mask = torch.zeros( ( 480, 480 ) )

		xl_mask[r1, r2] = 1
		xl_dist[r1, r2] = 35

		xl_mask[r2, r1] = 1
		xl_dist[r2, r1] = 35

		batch["xl_restraint"] = {}
		batch["xl_restraint"]["xl_res_mask"] = xl_mask
		# batch["xl_restraint"]["xl_tgt_mask"] = xl_dist

		return batch		


	def preprocess_xl_data( self, df: pd.DataFrame ):
		"""
		1. Sort the XLs according to chain IDs for prot1.
		2. Convert the residue positions to appropriate indices.
			For all residues in a chain, the index must be shifted by the no. 
				of residues in the previous chain.
		3. Split the indices into column vectors.
		"""



	def create_xl_map( self, xl_res1: np.array, xl_res2: np.array ):
		"""
		Create a zero-matrix with the shape defined by the system length.
		Add 1's for XL'd residues, creating a binary XL-map (essentially a contact map).
		"""


	def make_xl_restraint_features( self ):
		"""
		Given the XL'd residues create a binary mask for XL'd residue pairs (xl_res_mask).
		"""

