import numpy as np
import pandas as pd
import subprocess
import os
import glob

from utils import run_subprocess


class CreateBenchmark():
	def __init__( self ):
		self.jwalk_exec = "jwalk"
		self.xl_length = 35
		self.base_dir = os.path.abspath( "./" )


	def forward( self ):
		for name in ["2ayo"]:
			sys_dir = os.path.abspath( f"./benchmark/{name}/" )
			# Move to the system dir.
			os.chdir( sys_dir )
			self.simulate_xl_data( name )
			# Back to base.
			os.chdir( self.base_dir )


	def run_jwalk( self, pdb_file: str ):
		"""
		Run Jwalk to obtain XLs given a .pdb file.
	
		Input:
		----------
		pdb_file --> PATh to the .pdb file.

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



	def simulate_xl_data( self, name: str ):
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
		# Run Jwalk.
		if not os.path.exists( f"./Jwalk_results/" ):
			self.run_jwalk( f"./{name}.pdb" )
		else:
			print( f"Jwalk_results already present in {name} dir..." )

		# Obtain intraprotein and interprotein XLs from Jwalk output.
		intraprotein_xls, interprotein_xls = self.parse_jwalk_output( name )
		print( f"Intra-XLs = {len( intraprotein_xls )} \t Inter-XLs = {len( interprotein_xls )}" )
		
		# Save on disk.
		intraprotein_xls.to_csv( f"{name}_intraprotein_xls.csv" )
		interprotein_xls.to_csv( f"{name}_interprotein_xls.csv" )


if __name__ == "__main__":
	CreateBenchmark().forward()


