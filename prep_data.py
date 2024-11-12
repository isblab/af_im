import numpy as np
import pandas as pd
import subprocess
import os
import glob

from utils import run_subprocess


class DataGathering():
	def __init__( self ):
		self.jwalk_exec = "jwalk"
		self.xl_length = 35


	def forward( self ):
		for name in ["2ayo"]:
			self.run_jwalk( f"{name}.pdb" )
			self.move_to_sys_dir( name )
			self.parse_jwalk_output( name )



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


	def move_to_sys_dir( self, name ):
		"""
		Move Jwalk output to the system dir.

		Input:
		----------
		name --> System name.

		Returns:
		----------
		None
		"""
		jwalk_out = os.path.abspath( "./Jwalk_results" )
		sys_dir = os.path.abspath( f"./{name}/" )
		if not os.path.exists( f"{sys_dir}/Jwalk_results/" ):
			cmd = ["mv", f"{jwalk_out}", f"{sys_dir}"]
			run_subprocess( cmd )
		else:
			print( f"Jwalk_results already present in {name} dir..." )



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
		file_path = glob.glob( f"./benchmark/{name}/Jwalk_results/*.txt" )
		if len( file_path ) == 0:
			raise Exception( f"Jwalk_results dir does not exist for {name}..." )
		file_path = file_path[0]
		df = pd.read_csv( file_path, delim_whitespace = True )

		intraprotein_xls = df[df["Atom1"].str.split( "-" ).str[2] == df["Atom2"].str.split( "-" ).str[2]]
		interprotein_xls = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( "-" ).str[2]]

		# Remove XLs with distances higher than the XL_length.
		intraprotein_xls = df.loc[df["Euclidean"] <= self.xl_length]
		interprotein_xls = df.loc[df["Euclidean"] <= self.xl_length]

		# Extract chain ID and res no.
		for i in [1, 2]:
			intraprotein_xls = intraprotein_xls.copy()
			intraprotein_xls.loc[f"prot{i}"] = intraprotein_xls[f"Atom{i}"].str.split( "-" ).str[2]
			interprotein_xls = interprotein_xls.copy()
			interprotein_xls.loc[f"res{i}"] = intraprotein_xls[f"Atom{i}"].str.split( "-" ).str[1]

		# Drop the irrelevant columns.
		drop_cols = ["Index", "Model", "Atom1", "Atom2", "SASD", "Euclidean", "Distance"]
		intraprotein_xls = intraprotein_xls.drop( drop_cols, axis = 1 )
		interprotein_xls = interprotein_xls.drop( drop_cols, axis = 1 )



if __name__ == "__main__":
	DataGathering().forward()

