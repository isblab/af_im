"""
Contains classes to obtain simulated experimental data.
"""
import os, glob, copy, time, re
import numpy as np
import pandas as pd
from multiprocessing import Pool
import tqdm

from utils.utils import ( run_subprocess,
						open_file_handler )


class SimulateCrosslinks():
	"""
	Simulate cross-linking data using JWalk.
	"""
	def __init__( self, pdb_ids_list: str,
					pdb_struct_dir: str,
					struct_format: str,
					xl_max_bound: int,
					min_inter_xls: int,
					cores: int ):
		self.jwalk_exec = "jwalk"
		self.pdb_ids_list = pdb_ids_list
		self.pdb_struct_dir = pdb_struct_dir
		self.struct_format = struct_format
		self.xl_max_bound = xl_max_bound
		self.min_inter_xls = min_inter_xls
		self.cores = cores

		self.xls_dict = {}



	def forward( self ):
		"""
		"""
		t_start = time.time()
		self.initialize_logs_dict()

		self.jwalk_logs["Total_pdb_ids"] = len( self.pdb_ids_list )
		self.get_xls_in_parallel()

		t_end = time.time()
		time_taken = t_end - t_start
		self.jwalk_logs["pdb_ids_with_inter_xls"] = len( self.xls_dict )
		self.jwalk_logs["time_taken"] = time_taken



	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.jwalk_logs = {
		k:[[], 0] for k in ["failed_to_run_jwalk", "no_inter_xls",
						"too_few_xls"]
		}



	def run_jwalk( self, entry_id: str ):
		"""
		Run Jwalk to obtain XLs given a .pdb file.
			"""
		cmd = [
		f"{self.jwalk_exec}",
		"-i", f"./{entry_id}.pdb",
		# "-i", f"{struct_file}",
		]

		run_subprocess( cmd )



	def parse_jwalk_output( self, entry_id: str ) -> pd.DataFrame:
		"""
		Jwalk writes a .txt file containing all the XLs within
			the output dir named Jwalk_results.
		Parse the file and return as a pd.DataFrame
		"""
		xl_file_path = glob.glob( f"./Jwalk_results/{entry_id}_*.txt" )
		if len( xl_file_path ) == 0:
			xl_df = None
		else:
			xl_file_path = xl_file_path[0]
			xl_df = pd.read_csv( xl_file_path, sep = "\s+" ) # delim_whitespace = True
		return xl_df


	def get_tp_xls( self, xl_df: pd.DataFrame ) -> pd.DataFrame:
		"""
		Obtain true positive (TP) XLs.
			XLs with distance <= max bound.
		"""
		tp_xl_df = copy.copy( xl_df )
		# Using SASD provides more accurate XLs.
		tp_xl_df = tp_xl_df.loc[tp_xl_df["SASD"] <= self.xl_max_bound]
		return tp_xl_df


	def get_fp_xls( self, xl_df: pd.DataFrame ) -> pd.DataFrame:
		"""
		Obtain false positive (FP) XLs.
			We consider XLs with distance > the (max bound + 10A) as FP.
		"""
		fp_xl_df = copy.copy( xl_df )
		# Using SASD provides more accurate XLs.
		fp_xl_df = fp_xl_df.loc[fp_xl_df["SASD"] > self.xl_max_bound+10.0]
		return fp_xl_df


	def get_interprotein_xls( self, df: pd.DataFrame ) -> pd.DataFrame:
		"""
		It provides the following info:
			Index, Model (input file name)
			Atom1 --> AA-RES-CHAIN-CA
			Atom2 --> AA-RES-CHAIN-CA
			SASD --> Solvent accessible surface distance.
			Eculidean distance --> distance between CA atoms.
		Fetch all inter-protein XLs for which the SASD is less than xl_max_bound.
		Note: In some cases. JWalk adds "--" instead of "-" (e.g. 8sjj.
		"""
		inter = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( r'-{1,2}' ).str[2]]

		# Extract chain ID and res no.
		interprotein_xls = pd.DataFrame()
		for i in [1, 2]:
			interprotein_xls[f"prot{i}"] = inter[f"Atom{i}"].str.split( r'-{1,2}' ).str[2]
			interprotein_xls[f"res{i}"] = inter[f"Atom{i}"].str.split( r'-{1,2}' ).str[1]

		interprotein_xls = interprotein_xls.reset_index( drop = True )

		return interprotein_xls



	def get_xls_for_entry_id( self, entry_id: str ):
		"""
		Given an entry_ids, obtain inter-protein XLs.
		JWalk needs the PDB file in the current dir to run.
		Instead of moving to the PDB dir, I am copying the
			PDB to the current dir and running JWalk.
		This allows to parallelize JWalk.
		"""
		logs = {}
		struct_file = os.path.join( self.pdb_struct_dir,
									f"{entry_id}.{self.struct_format}" )

		# Copy PDB file to current dir.
		run_subprocess( ["cp", f"{struct_file}", "./"] )

		self.run_jwalk( entry_id )
		print( f"JWalk run complete for {entry_id}." )

		xl_df = self.parse_jwalk_output( entry_id )

		if xl_df is None:
			tp_inter_xls, fp_inter_xls = None, None
			logs["failed_to_run_jwalk"] = entry_id
		else:
			tp_xls = self.get_tp_xls( xl_df )
			tp_inter_xls = self.get_interprotein_xls( tp_xls )
			fp_xls = self.get_fp_xls( xl_df )
			fp_inter_xls = self.get_interprotein_xls( fp_xls )

			if tp_inter_xls.shape[0] == 0:
				logs["no_inter_xls"]  = [entry_id]
				tp_inter_xls, fp_inter_xls = None, None

			elif tp_inter_xls.shape[0] < self.min_inter_xls:
				logs["too_few_xls"]  = [entry_id]
				tp_inter_xls, fp_inter_xls = None, None

		# remove PDB file from current dir.
		run_subprocess( ["rm", f"./{entry_id}.{self.struct_format}"] )
		print( f"Completed for {entry_id}." )
		return entry_id, tp_inter_xls, fp_inter_xls, logs



	def get_xls_in_parallel( self ):
		"""
		Obtain XLs serially for the given entry_id's.
		"""
		# curr_dir = os.getcwd()
		# os.chdir( self.pdb_struct_dir )
		for idx, entry_id in enumerate( self.pdb_ids_list ):
		# curr_dir = os.getcwd()

		# with Pool( self.cores ) as p:
		# 	for result in tqdm.tqdm( p.imap_unordered( self.get_xls_for_entry_id,
		# 								self.pdb_ids_list ),
		# 								total = len( self.pdb_ids_list ) ):
		# 		entry_id, tp_inter_xls, fp_inter_xls, logs = result

			entry_id, tp_inter_xls, fp_inter_xls, logs = self.get_xls_for_entry_id( entry_id )
			if tp_inter_xls is None:
				for k in logs:
					self.jwalk_logs[k][0].extend( logs[k] )
					self.jwalk_logs[k][1] += len( logs[k] )
			else:
				print( f"{idx}/{len( self.pdb_ids_list )} --> {entry_id}" )
				print( f"TP inter-protein XLs = {tp_inter_xls.shape[0]}" +
					f"\tFP inter-protein XLs = {fp_inter_xls.shape[0]}" )
				self.xls_dict[entry_id] = {"tp_xls": tp_inter_xls,
											"fp_xls": fp_inter_xls,
											"count_tp_xls": tp_inter_xls.shape[0]}
		run_subprocess( ["rm", "-r", f"./Jwalk_results/"] )

