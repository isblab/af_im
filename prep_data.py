import numpy as np
import pandas as pd
import subprocess
import os
import glob
import ml_collections as mlc
import re

from typing import List, Dict

from utils import ( run_subprocess, read_json, write_json )
from utils import ( write_to_file, 
					read_fasta_from_response )
from api_utils import get_casp_entry

"""
Using CASP15 dataset as our benchmark.
targetlist.csv for CASP15 must be present in ./raw/.
"""

class CreateBenchmark():
	def __init__( self ):
		self.jwalk_exec = "jwalk"
		self.xl_length = 35

		self.benchmark_config_dict = {}
		self.benchmark_seq_dict = {}
		self.casp_input_csv = "./raw/targetlist_mod.csv"
		self.base_dir = os.path.join( "./benchmark/" )

		if not os.path.exists( self.base_dir ):
			os.makedirs( self.base_dir )


	def forward( self ):
		"""
		For our benchmark we consider only protein multimer entries.
		For each entry we need the:
			Sequence of the constituent proteins.
			Structure file.
			Stoichiometry.
		"""
		casp_dict = self.parse_casp_input_file()
		self.create_system_dir( casp_dict )
		self.get_seq_n_struct( casp_dict )
		self.create_sys_config_dict( casp_dict )
		self.simulate_xl_data( casp_dict )
		exit()
		# for name in ["2ayo"]:
		for name in ["H1129"]:
			sys_dir = os.path.abspath( f"./benchmark/{name}/" )
			# Move to the system dir.
			os.chdir( sys_dir )
			self.simulate_xl_data( name )
			# Back to base.
			os.chdir( self.base_dir )



	def parse_casp_input_file( self ):
		"""
		Parse the CASP csv file and extract relevant details.
		The csv file is not pandas readable so reading as text file.
		It contains the following headers:
			Target;Type;Res;Oligo.State;Entry Date; Server Exp.;Human Exp.;QA Exp.;Cancellation Date;Description
		We need the following:
			Target (0): CASP entry ID
			Type (1): Entry type ("All groups", "RNA", "Ligand", "Server")
			Oligo.State (3): Stoichiometry.
			Description (9): Contains the PDB ID where available.
		"""
		with open( self.casp_input_csv, "r" ) as f:
			casp = f.readlines()

		casp_dict = {}
		stop = 0
		for i in range( 1, len( casp ) ):
			line = casp[i]
			line = line.strip().split( ";" )

			type_ = line[1]
			if all( [x not in type_ for x in ["RNA", "Ligand"]] ):
				stoichiometry = line[3]
				if len( stoichiometry ) > 2:
					if stop > 4:
						break
					stop += 1
					casp_id = line[0]
					casp_dict[casp_id] = {}
					casp_dict[casp_id]["stoichiometry"] = stoichiometry
					casp_dict[casp_id]["pdb_id"] = line[9]

		return casp_dict



	def create_system_dir( self, casp_dict: Dict ):
		"""
		Create directories for all benchmark systems.
		"""
		for casp_id in casp_dict:
			sys_dir = os.path.join( self.base_dir, f"{casp_id}" )
			if not os.path.exists( sys_dir ):
				os.makedirs( sys_dir )



	def get_seq_n_struct( self, casp_dict: Dict ):
		"""
		Get the sequence and structure for the CASP entry.
		"""
		for casp_id in casp_dict:
			seq_file = os.path.join( self.base_dir, f"{casp_id}/{casp_id}_seq.json" )
			pdb_file = os.path.join( self.base_dir, f"{casp_id}/{casp_id}.pdb" )
			if os.path.exists( seq_file ) and os.path.exists( pdb_file ):
				seq_dict = read_json( seq_file )
			
			else:
				fasta_response, struct_response = get_casp_entry( casp_id )

				seq_dict = read_fasta_from_response( fasta_response )
				write_json( seq_dict, seq_file )

				write_to_file( struct_response, pdb_file )

			self.benchmark_seq_dict[casp_id] = seq_dict



	def get_stoichiometry( self, stoichiometry: str ):
		"""
		Given the stoichiometry as an alphanumeric str, extract the chains and their copy numbers.
		"""
		tups = re.findall( r"([A-Z])(\d+)", stoichiometry )

		stoichiometry = [int( v ) for k, v in tups]
		
		return stoichiometry



	def get_entities( self, seq_dict: Dict, stoichiometry: List ):
		"""
		Create all entities for the system.
		"""
		entities = []

		sequences = list( seq_dict.values() )
		
		for i in range( len( stoichiometry ) ):
			copy_num = stoichiometry[i]
			
			seq = sequences[i]

			start = 1
			end = len( seq )

			entities.append( 
				{
					"uni_id": "",
					"copy_num": copy_num,
					"start": start,
					"end": end,
					"sequence": seq
				}
			 )

		return entities



	def create_sys_dict_entry( self, sys_idx: int, sys_name: str, entities: Dict ):
		"""
		Given a CASP entry, create a config dict containing:
			"system_{index}": {
					"name",
					"entity": {
						{},
						{}
					},
					"data_gathering": {
						"xl_restraint": {}
					}
			}
		"""
		sys_dict = {
					f"System_{sys_idx}": {
							"name": sys_name,
							"entity": entities,
							"data_gathering": {
								"xl_restraint": {
									"xl_max_bound": self.xl_length,
									"file_name": f"{sys_name}_interprotein_xls.csv",
								}
							}
						}
					}
		return sys_dict



	def create_sys_config_dict( self, casp_dict: Dict ):
		"""
		For all the benchmark entries create a config dict.
		We use the CASP ID as the system name.
		"""

		for idx, sys_name in enumerate( casp_dict ):
			print( sys_name )
			config_file = os.path.join( self.base_dir, f"{sys_name}/sys_conf_{sys_name}.json" )
			seq_dict = self.benchmark_seq_dict[sys_name]
			stoichiometry = self.get_stoichiometry( casp_dict[sys_name]["stoichiometry"] )
			entities = self.get_entities( seq_dict, stoichiometry )
			sys_dict = self.create_sys_dict_entry( idx, sys_name, entities )

			write_json( sys_dict, config_file )



	def run_jwalk( self, pdb_file: str ):
		"""
		Run Jwalk to obtain XLs given a .pdb file.
	
		Input:
		----------
		pdb_file --> Path to the .pdb file.

		returns:
		----------
		None
		"""
		cmd = [
		f"{self.jwalk_exec}",
		"-i", f"{pdb_file}",
		]

		run_subprocess( cmd )



	def parse_jwalk_output( self, name: str ):
		"""
		Jwalk writes a .txt file containing all the XLs.
		It provides the following info:
			Index, Model (input file name)
			Atom1 --> AA-RES-CHAIN-CA
			Atom2 --> AA-RES-CHAIN-CA
			SASD --> Solvent accessible surface distance.
			Eculidean distance --> distance between CA atoms.
		Fetch all intra and inter-protein XLs for which the Euclidean distance is less than xl_length bound.

		Input:
		----------
		name --> System name.

		Returns:
		----------
		None
		"""
		file_path = glob.glob( f"./Jwalk_results/*.txt" )
		if len( file_path ) == 0:
			raise Exception( f"Jwalk results .txt file does not exist for {name}..." )
		file_path = file_path[0]
		df = pd.read_csv( file_path, sep = "\s+" ) # delim_whitespace = True

		# Remove XLs with Eulcidean distances higher than the XL_length.
		# df = df.loc[df["Euclidean"] <= self.xl_length]
		# Remove XLs with SASD distances higher than the XL_length.
		# 	Using SASD provides more accurate XLs.
		df = df.loc[df["SASD"] <= self.xl_length]

		intra = df[df["Atom1"].str.split( "-" ).str[2] == df["Atom2"].str.split( "-" ).str[2]]
		inter = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( "-" ).str[2]]

		# Extract chain ID and res no.
		intraprotein_xls = pd.DataFrame()
		interprotein_xls = pd.DataFrame()
		for i in [1, 2]:
			intraprotein_xls[f"prot{i}"] = intra[f"Atom{i}"].str.split( "-" ).str[2]
			intraprotein_xls[f"res{i}"] = intra[f"Atom{i}"].str.split( "-" ).str[1]
			interprotein_xls[f"prot{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[2]
			interprotein_xls[f"res{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[1]

		intraprotein_xls = intraprotein_xls.reset_index( drop = True )
		interprotein_xls = interprotein_xls.reset_index( drop = True )

		return intraprotein_xls, interprotein_xls



	def simulate_xl_data( self, casp_dict: Dict ):
		"""
		Simulate XLs for the system uisng Jwalk.
		Save the intraprotein and interprotein XLs as csv files.

		Input:
		----------
		name --> System name.

		Returns:
		----------
		None
		"""
		for casp_id in casp_dict:
			sys_dir = os.path.join( self.base_dir, f"{casp_id}/" )
			os.chdir( sys_dir )
			# Run Jwalk.
			if not os.path.exists( f"./Jwalk_results/" ):
				self.run_jwalk( f"./{casp_id}.pdb" )
			else:
				print( f"Jwalk_results already present in {casp_id} dir..." )

			# Obtain intraprotein and interprotein XLs from Jwalk output.
			intraprotein_xls, interprotein_xls = self.parse_jwalk_output( casp_id )
			print( f"Intra-XLs = {len( intraprotein_xls )} \t Inter-XLs = {len( interprotein_xls )}" )
			
			# Save on disk.
			intraprotein_xls.to_csv( f"{casp_id}_intraprotein_xls.csv", index = False )
			interprotein_xls.to_csv( f"{casp_id}_interprotein_xls.csv", index = False )

			os.chdir( "../../" )


if __name__ == "__main__":
	CreateBenchmark().forward()



