import numpy as np
import pandas as pd
import subprocess

from utils import run_subprocess


class InteractionData():
	def __init__( self ):
		self.jwalk_exec = "jwalk"
		self.xl_length = 35


	def run_jwalk( self ):
		"""
		Run Jwalk to obtain XLs given a .pdb file.
		"""
		cmd = [
		f"{self.jwalk_exec}",
		"-i", f"{pdb_file}",
		]

		run_subprocess( cmd )


	def parse_jwalk_output( self, file_path ):
		"""
		Jwalk writes a .txt file containing all the XLs.
		It provides the following info:
			Index, Model (input file name)
			Atom1 --> AA-RES-CHAIN-CA
			Atom2 --> AA-RES-CHAIN-CA
			SASD --> Solvent accessible surface distance.
			Eculidean distance --> distance between CA atoms.
		"""
		df = pd.read_csv( file_path, delim_whitespace = True )

		intraprotein_xls = df[df["Atom1"].str.split( "-" ).str[2] == df["Atom2"].str.split( "-" ).str[2]]
		interprotein_xls = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( "-" ).str[2]]

		# Remove XLs with distances higher than the XL_length.
		intraprotein_xls = df.loc[df["Euclidean"] <= self.xl_length]
		interprotein_xls = df.loc[df["Euclidean"] <= self.xl_length]

		# Extract chain ID and res no.
		for i in [1, 2]:
			intraprotein_xls[f"prot{i}"] = intraprotein_xls[f"Atom{i}"].str.split( "-" ).str[2]
			interprotein_xls[f"res{i}"] = intraprotein_xls[f"Atom{i}"].str.split( "-" ).str[1]

		# Drop the irrelevant columns.
		drop_cols = ["Index", "Model", "Atom1", "Atom2" "SASD", "Euclidean", "Distance"]
		intraprotein_xls = intraprotein_xls.drop( drop_cols, axis = 1 )
		interprotein_xls = interprotein_xls.drop( drop_cols, axis = 1 )

