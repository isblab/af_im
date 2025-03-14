import os
import glob
import copy
import re
import subprocess
import warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import ml_collections as mlc

from utils import ( run_subprocess, read_json, write_json )
from utils import ( write_to_file, 
					read_fasta_from_response )
from api_utils import get_casp_entry, download_pdb

"""
Using CASP15 dataset as our benchmark.
targetlist.csv for CASP15 must be present in ./raw/.
"""

class CreateBenchmark():
	def __init__( self ):
		self.jwalk_exec = "jwalk"
		self.xl_length = 35

		self.casp_dict = {}
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
		self.casp_dict = self.parse_casp_input_file()
		for sys_num, casp_id in enumerate( self.casp_dict ):
			print( f"CASP ID: {casp_id}" )
			self.get_entry( sys_num, casp_id )
			print( "\n----------------------------------------------\n" )
		# self.create_system_dir( casp_dict )
		# self.get_seq_n_struct( casp_dict )
		# self.create_sys_config_dict( casp_dict )
		# self.simulate_xl_data( casp_dict )
		
		# exit()
		# # for name in ["2ayo"]:
		# for name in ["H1129"]:
		# 	sys_dir = os.path.abspath( f"./benchmark/{name}/" )
		# 	# Move to the system dir.
		# 	os.chdir( sys_dir )
		# 	self.simulate_xl_data( name )
		# 	# Back to base.
		# 	os.chdir( self.base_dir )



	def get_entry( self, sys_num, casp_id: str ):
		"""
		For each CASP entry:
			Get the sequence and PDB structure.
			Create system config dict.
			Simulate XLs using JWalk.
		"""
		# pdb_file = os.path.join( self.base_dir, f"{casp_id}.pdb" )
		dest_pdb = os.path.join( self.base_dir, f"{casp_id}/{casp_id}.pdb" )

		if os.path.exists( dest_pdb ):
			dwnld_success = True
		else:
			dwnld_success, pdb_file = self.get_struct( casp_id )

		if dwnld_success:
			# Get FASTA seq for entry.
			seq_dict, seq_file = self.get_seq( casp_id )
			# self.benchmark_seq_dict[casp_id] = seq_dict
			
			stoichiometry = self.get_stoichiometry( 
									self.casp_dict[casp_id]["stoichiometry"]
									)
			
			if self.sys_checks:
				self.create_system_dir( casp_id )
				sys_dict, config_file = self.create_sys_config_dict( sys_num,
																	casp_id,
																	seq_dict,
																	stoichiometry )
				
				success_jwalk, interprotein_xls = self.simulate_xl_data( casp_id )

				if success_jwalk:
					if not os.path.exists( dest_pdb ):
						cmd = ["mv", f"{pdb_file}", f"{dest_pdb}"]
						run_subprocess( cmd )

					dest_jwalk = os.path.join( self.base_dir, f"{casp_id}/" )
					cmd = ["mv", f"{self.base_dir}", f"{dest_jwalk}"]

					interprotein_xls.to_csv( f"{casp_id}_interprotein_xls.csv", index = False )

					write_json( seq_dict, seq_file )
					write_json( sys_dict, config_file )


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
		print( "Parsing CASP input file..." )
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
				# No monomers.
				if len( stoichiometry ) > 2:
					if line[9] != "":
						casp_id = line[0]
						casp_dict[casp_id] = {}
						casp_dict[casp_id]["Total_res"] = line[2]
						casp_dict[casp_id]["stoichiometry"] = stoichiometry
						casp_dict[casp_id]["pdb_id"] = line[9]

		return casp_dict



	def create_system_dir( self, casp_id: str ):
		"""
		Create directories for all benchmark systems.
		"""
		# for casp_id in casp_dict:
		sys_dir = os.path.join( self.base_dir, f"{casp_id}" )
		if not os.path.exists( sys_dir ):
			os.makedirs( sys_dir )



	def get_seq( self, casp_id: str ):
		"""
		Get the sequence for the CASP entry.
		"""
		print( "Downloading sequence and structure files for CASP entries..." )
		seq_file = os.path.join( self.base_dir, f"{casp_id}/{casp_id}_seq.json" )
		if os.path.exists( seq_file ):
			seq_dict = read_json( seq_file )
		
		else:
			fasta_response = get_casp_entry( casp_id )

			seq_dict = read_fasta_from_response( fasta_response )
		return seq_dict, seq_file



	def get_struct( self, casp_id: str ) -> Tuple[bool, str]:
		"""
		Download the PDB file for the CASP entry.
		"""
		success = False
		pdb_file = os.path.join( self.base_dir, f"{casp_id}.pdb" )
		if not os.path.exists( pdb_file ):
			pdb_id = self.casp_dict[casp_id]["pdb_id"]

			result = download_pdb( pdb_id, "pdb", pdb_file )
			# If .pdb file doesn't exist, try .cif file.
			if not result:
				result = download_pdb( pdb_id, "cif", pdb_file )
				if not result:
					raise Exception( f"Unable to download PDB: {pdb_id}" )
				else:
					success = True
			else:
				success = True
		else:
			success = True

		return success, pdb_file



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


	def sys_checks( self, seq_dict: Dict, stoichiometry: List ):
		"""
		Check if the no. of chains in FASTA is not the same as stoichiometry specified.
		"""
		success = []
		if len( stoichiometry ) != len( seq_dict.keys() ):
			warnings.warn( f"Warning: The no. of sequences in FASTA file is not the same as" +
							f"the specified stoichiometry for {sys_name}..." )
			success.append( False )

		# Remove all but dimers for now.
		if len( stoichiometry ) > 2:
			print( f"{sys_name}: not a dimer" )
			print( stoichiometry )
			success.append( False )

		# Remove all homomers for now.
		if not all( [s == 1 for s in stoichiometry] ):
			print( f"{sys_name}: homomer" )
			success.append( False )

		return all( success )


	def create_sys_config_dict( self, sys_num: int, casp_id: str, seq_dict: Dict, stoichiometry: List ):
		"""
		For all the benchmark entries create a config dict.
		We use the CASP ID as the system name.
		"""
		sys_name = casp_id
		# casp_dict_copy = copy.deepcopy( casp_dict )
		# for idx, sys_name in enumerate( casp_dict_copy ):
		config_file = os.path.join( self.base_dir, f"{sys_name}/sys_conf_{sys_name}.json" )
		# seq_dict = self.benchmark_seq_dict[sys_name]
		# stoichiometry = self.get_stoichiometry( casp_dict[sys_name]["stoichiometry"] )

		# # If the no. of chains in FASTA is not the same as stoichiometry specified.
		# if len( stoichiometry ) != len( seq_dict.keys() ):
		# 	warnings.warn( f"Warning: The no. of sequences in FASTA file is not the same as" +
		# 					f"the specified stoichiometry for {sys_name}..." )
		# 	run_subprocess( ["rm", "-r", f"{os.path.join( self.base_dir, sys_name )}"] )
		# 	casp_dict.pop( sys_name )
		# 	continue

		# # Remove all but dimers for now.
		# if len( stoichiometry ) > 2:
		# 	print( f"{sys_name}: not a dimer" )
		# 	print( stoichiometry )
		# 	run_subprocess( ["rm", "-r", f"{os.path.join( self.base_dir, sys_name )}"] )
		# 	casp_dict.pop( sys_name )
		# 	continue

		# # Remove all homomers for now.
		# if not all( [s == 1 for s in stoichiometry] ):
		# 	print( f"{sys_name}: homomer" )
		# 	run_subprocess( ["rm", "-r", f"{os.path.join( self.base_dir, sys_name )}"] )
		# 	casp_dict.pop( sys_name )
		# 	continue

		entities = self.get_entities( seq_dict, stoichiometry )
		sys_dict = self.create_sys_dict_entry( sys_num, sys_name, entities )

		return sys_dict, config_file
		# write_json( sys_dict, config_file )



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

		# intra = df[df["Atom1"].str.split( "-" ).str[2] == df["Atom2"].str.split( "-" ).str[2]]
		inter = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( "-" ).str[2]]

		# Extract chain ID and res no.
		# intraprotein_xls = pd.DataFrame()
		interprotein_xls = pd.DataFrame()
		for i in [1, 2]:
			# intraprotein_xls[f"prot{i}"] = intra[f"Atom{i}"].str.split( "-" ).str[2]
			# intraprotein_xls[f"res{i}"] = intra[f"Atom{i}"].str.split( "-" ).str[1]
			interprotein_xls[f"prot{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[2]
			interprotein_xls[f"res{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[1]

		# intraprotein_xls = intraprotein_xls.reset_index( drop = True )
		interprotein_xls = interprotein_xls.reset_index( drop = True )

		# return intraprotein_xls, interprotein_xls
		return interprotein_xls


	def simulate_xl_data( self, casp_id: str ) -> Tuple[bool, pd.DataFrame]:
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
		print( "Simulating XL data..." )
		# for casp_id in casp_dict:
		success = False
		sys_dir = os.path.join( self.base_dir, f"{casp_id}/" )
		os.chdir( sys_dir )
		# sys_dir = os.path.join( self.base_dir )
		if os.path.exists( f"./{casp_id}.pdb" ):
			struct_file = os.path.join( f"./{casp_id}.pdb" )
		elif os.path.exists( f"./{casp_id}.cif" ):
			struct_file = os.path.join( f"./{casp_id}.cif" )
		else:
			raise FileNotFoundError( f"PDB/CIF file not found for entry {casp_id}..." )
		# if not os.path.exists( pdb_file ):
		# 	raise FileNotFoundError( f"PDB/CIF file not found for entry {casp_id}..." )
		
		try:
			# Run Jwalk.
			if not os.path.exists( f"./Jwalk_results/" ):
				self.run_jwalk( f"./{casp_id}.pdb" )
			else:
				print( f"Jwalk_results already present in {casp_id} dir..." )

			# Obtain intraprotein and interprotein XLs from Jwalk output.
			interprotein_xls = self.parse_jwalk_output( casp_id )
			# print( f"Intra-XLs = {len( intraprotein_xls )} \t Inter-XLs = {len( interprotein_xls )}" )
			print( f"Inter-XLs = {len( interprotein_xls )}" )
			
			# Save on disk.
			# intraprotein_xls.to_csv( f"{casp_id}_intraprotein_xls.csv", index = False )
			# interprotein_xls.to_csv( f"{casp_id}_interprotein_xls.csv", index = False )

			success = True
		except:
			print( f"Couldn't run JWalk for {casp_id}..." )
			success = False
		os.chdir( "../../" )
		return success, interprotein_xls


if __name__ == "__main__":
	CreateBenchmark().forward()
