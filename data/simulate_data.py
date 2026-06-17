"""
Contains classes to obtain simulated experimental data.
"""
from typing import List
import os, glob, copy, time, re
import numpy as np
import pandas as pd
from multiprocessing import Pool
import tqdm

from utils.utils import ( run_subprocess,
						read_json, write_json )


class SimulateCrosslinks():
	"""
	Simulate cross-linking data using JWalk.

	Inputs:
	----------
	jwalk_exec: JWalk executable.
	pdb_ids_list: list of PDB IDs.
	pdb_struct_dir: dir contaiing the structure file for all complexes.
	jwalk_dir: dir for saving the JWalk output.
	struct_format: format for the structure file (.pdb/,cif).
	short_linker: Ca-Ca distance for Xls with short linker.
	long_linker: Ca-Ca distance for Xls with longer linker.
	num_inter_xls: min no. of interprotein XLs needed (both short and long).
	cores: no. of CPU cores to be used for parallelization.
	"""
	def __init__( self,
		jwalk_exec: str,
		pdb_ids_list: List[str],
		pdb_struct_dir: str,
		jwalk_dir: str,
		struct_format: str,
		short_linker: float,
		long_linker: float,
		num_inter_xls: int,
		cores: int,
		aa1: str = "LYS",
		aa2: str = "LYS",
		):
		self.jwalk_exec = jwalk_exec
		self.pdb_ids_list = pdb_ids_list
		self.pdb_struct_dir = pdb_struct_dir
		self.jwalk_dir = jwalk_dir
		self.struct_format = struct_format
		self.short_linker = short_linker
		self.long_linker = long_linker
		self.num_inter_xls = num_inter_xls
		self.cores = cores
		self.aa1 = aa1
		self.aa2 = aa2
		# If true, reinitialize logs even when the file exists.
		self.reinit_logs = False

		self.xl_file_paths = {}
		self.xls_dict = {}


	def forward( self ):
		"""
		Run JWalk to obtain all simulated XLs for each entry.
			Save on disk.
		Select all interprotein XLs.
			Select TP XLS for:
				Short linker
				Long linker
			Also select FP XLs
		Note: True positive (TP/tp); False positive (FP/fp); Cross-link (XL)
		"""
		t_start = time.perf_counter()
		self.initialize_logs_dict()
		self.create_xl_file_paths()

		self.jwalk_logs["Total_pdb_ids"] = len( self.pdb_ids_list )

		self.get_simulated_xls_jwalk()
		self.select_xls_for_benchmark()

		t_end = time.perf_counter()
		time_taken = t_end - t_start
		self.jwalk_logs["selected_entry_ids"] = len( self.xls_dict )
		self.jwalk_logs["time_taken"] = time_taken
		write_json( self.jwalk_logs, self.logs_file )


	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.logs_file = os.path.join( self.jwalk_dir, "Logs_jwalk.json" )
		if os.path.exists( self.logs_file ) and not self.reinit_logs:
			self.jwalk_logs = read_json( self.logs_file )
		else:
			self.jwalk_logs = {
			k:[[], 0] for k in ["failed_to_run_jwalk", "no_xls",
				"entry_id_with_xls", "no_inter_xls", "too_few_short_xls",
				"too_few_long_xls", "no_fp_xls", "selected_entry_ids"
				]
			}

	def create_xl_file_paths( self ):
		"""
		Jwalk writes a .txt file containing all the computed XLs.
		For each entry, create the path for the JWalk XLs .csv file.
		"""
		for entry_id in self.pdb_ids_list:
			if self.aa1 == "LYS" and self.aa2 == "LYS":
				xl_file = f"{entry_id}_jwalk.csv"
			else:
				xl_file = f"{entry_id}_jwalk_{self.aa1}-{self.aa2}.csv"
			self.xl_file_paths[entry_id] = os.path.join(
				self.jwalk_dir,
				xl_file )

	################################################################################
	def run_jwalk( self, entry_id: str ):
		"""
		Run Jwalk to obtain XLs given a .pdb file.
		Remove the structure file once JWalk run is complete.

		Inputs:
		----------
		entry_id: PDB identifier for a system.
		"""
		cmd = [
		f"{self.jwalk_exec}",
		"-i", f"./{entry_id}.pdb",
		"-aa1", f"{self.aa1}",
		"-aa2", f"{self.aa2}"
		# "-i", f"{struct_file}",
		]

		run_subprocess( cmd )
		# Delete the structure file.
		os.remove( f"./{entry_id}.pdb" )


	def parse_jwalk_output( self, entry_id: str ) -> pd.DataFrame:
		"""
		Jwalk writes a .txt file containing all the XLs within
			the output dir named Jwalk_results.
		Parse the file and return as a pd.DataFrame.

		JWalk provides the following info:
			Index, Model (input file name)
			Atom1 --> AA-RES-CHAIN-CA
			Atom2 --> AA-RES-CHAIN-CA
				Atom field may have 1 or 2 hyphens (e.g. 8sjj).
			SASD --> Solvent accessible surface distance.
			Eculidean distance --> distance between CA atoms.

		Inputs:
		----------
		entry_id: PDB identifier for a system.
		"""
		xl_file_path = glob.glob( f"./Jwalk_results/{entry_id}_*.txt" )
		if len( xl_file_path ) == 0:
			xl_df = None
		else:
			xl_file_path = xl_file_path[0]
			xl_df = pd.read_csv( xl_file_path, sep = "\s+" ) # delim_whitespace = True
		return xl_df

	################################################################################
	def get_simulated_xls_jwalk( self ):
		"""
		Run JWalk to obtain XLs for each entry.
		Parse the JWalk output file and convert to a pd.DataFrame.
		Save on disk.
		"""
		for idx, entry_id in enumerate( self.pdb_ids_list ):
			print( f"{idx}/{len( self.pdb_ids_list )} --> {entry_id}" )
			xl_file = self.xl_file_paths[entry_id]
			if entry_id in self.jwalk_logs["entry_id_with_xls"][0] and os.path.exists( xl_file ):
				print( f"JWalk XLs already exist for {entry_id}..." )
				continue
			else:
				xl_df = self.get_xls_for_entry_id( entry_id )
				if xl_df is None:
					self.jwalk_logs["failed_to_run_jwalk"][0].append( entry_id )
					self.jwalk_logs["failed_to_run_jwalk"][1] += 1
				elif len( xl_df ) == 0:
					self.jwalk_logs["no_xls"][0].append( entry_id )
					self.jwalk_logs["no_xls"][1] += 1
				else:
					self.jwalk_logs["entry_id_with_xls"][0].append( entry_id )
					self.jwalk_logs["entry_id_with_xls"][1] += 1
					xl_df.to_csv( xl_file, index = False )
				# Remove the JWalk results dir.
				run_subprocess( ["rm", "-r", f"./Jwalk_results/"] )
				# Log after processing each entry_id.
				write_json( self.jwalk_logs, self.logs_file )


	def get_xls_for_entry_id( self, entry_id: str ) -> pd.DataFrame:
		"""
		Given an entry_ids, obtain inter-protein XLs.
			1. Copy PDB file to JWalk dir.
				JWalk needs the PDB file in the current dir to run.
				Instead of moving to the PDB dir, I am copying the
					PDB to the current dir and running JWalk.
			2. Run Jwalk.
			3. Parse the JWalk output and return a pd.DataFrame.

		Inputs:
		----------
		entry_id: PDB identifier for a system.
		"""
		struct_file = os.path.join( self.pdb_struct_dir,
									f"{entry_id}.{self.struct_format}" )

		# Copy PDB file to current dir.
		run_subprocess( ["cp", f"{struct_file}", "./"] )

		self.run_jwalk( entry_id )
		print( f"JWalk run complete for {entry_id}." )

		xl_df = self.parse_jwalk_output( entry_id )
		return xl_df

	################################################################################
	################################################################################
	def select_xls_for_benchmark( self ):
		"""
		For each entry_id,
			Select all interprotein XLs.
			Select XLs with short linker.
			Select XLs with long linker.
			Also select false positive (FP) XLs.
		An entry_id is selected only if it has short, long and FP XLs.
		"""
		for entry_id in self.jwalk_logs["entry_id_with_xls"][0]:
			# # For reruns, ignore if entry_id was already rejected.
			# skip = any( entry_id in self.jwalk_logs[k][0] 
			# 		for k in [
			# 			"no_inter_xls", "too_few_short_xls",
			# 			"too_few_long_xls", "no_fp_xls"
			# 			]
			# 	)
			# if skip:
			# 	continue

			xl_file = self.xl_file_paths[entry_id]
			df = pd.read_csv( xl_file )
			inter_xls = df[
				df["Atom1"].str.split( r'-{1,2}' ).str[2] !=
				df["Atom2"].str.split( r'-{1,2}' ).str[2]
				]

			# Skip an entry if no interprotein XLs exist.
			if len( inter_xls ) == 0 and entry_id not in self.jwalk_logs["no_inter_xls"][0]:
				self.jwalk_logs["no_inter_xls"][0].append( entry_id )
				self.jwalk_logs["no_inter_xls"][1] += 1
				continue

			short_xl_df = self.select_short_linker_xls( inter_xls = inter_xls )
			# Skip an entry if no short XLs exist.
			if len( short_xl_df ) < self.num_inter_xls:
				self.jwalk_logs["too_few_short_xls"][0].append( entry_id )
				self.jwalk_logs["too_few_short_xls"][1] += 1
				continue

			long_xl_df = self.select_long_linker_xls( inter_xls = inter_xls )
			# Skip an entry if no long XLs exist.
			if len( long_xl_df ) < self.num_inter_xls and entry_id not in self.jwalk_logs["too_few_long_xls"][0]:
				self.jwalk_logs["too_few_long_xls"][0].append( entry_id )
				self.jwalk_logs["too_few_long_xls"][1] += 1
				continue

			fp_xl_df = self.select_fp_xls( inter_xls = inter_xls )
			# Skip an entry if no FP XLs exist.
			if len( fp_xl_df ) == 0 and entry_id not in self.jwalk_logs["no_fp_xls"][0]:
				self.jwalk_logs["no_fp_xls"][0].append( entry_id )
				self.jwalk_logs["no_fp_xls"][1] += 1
				continue

			self.xls_dict[entry_id] = {
				"short_xls": short_xl_df,
				"long_xls": long_xl_df,
				"fp_xls": fp_xl_df
			}


	def select_short_linker_xls( self,
		inter_xls: pd.DataFrame
		) -> pd.DataFrame:
		"""
		Select XLs with SASD <self.short_linker length.
		XL file format: prot1,res1,prot2,res2

		Inputs:
		----------
		inter_xls: dataframe containing JWalk predicted
			interprotein XLs.

		Returns:
		----------
		short_xl_df: dataframe containing XLs with Ca-Ca distance
			<short_linker length.
		"""
		short_xl_df = pd.DataFrame()
		short_xls = inter_xls.loc[inter_xls["SASD"] <= self.short_linker]
		for i in [1, 2]:
			short_xl_df[f"prot{i}"] = short_xls[f"Atom{i}"].str.split( r'-{1,2}' ).str[2]
			short_xl_df[f"res{i}"] = short_xls[f"Atom{i}"].str.split( r'-{1,2}' ).str[1]
		short_xl_df = short_xl_df.reset_index( drop = True )
		
		return short_xl_df


	def select_long_linker_xls( self, inter_xls: pd.DataFrame ) -> pd.DataFrame:
		"""
		Select XLs with SASD <self.long_linker length.
		XL file format: prot1,res1,prot2,res2

		Inputs:
		----------
		inter_xls: dataframe containing JWalk predicted
			interprotein XLs.

		Returns:
		----------
		long_xl_df: dataframe containing XLs with Ca-Ca distance
			>short_linker and <=long_linker length.
		"""
		long_xl_df = pd.DataFrame()
		long_xls = inter_xls.loc[
			( inter_xls["SASD"] > self.short_linker ) &
			( inter_xls["SASD"] <= self.long_linker )
		]
		for i in [1, 2]:
			long_xl_df[f"prot{i}"] = long_xls[f"Atom{i}"].str.split( r'-{1,2}' ).str[2]
			long_xl_df[f"res{i}"] = long_xls[f"Atom{i}"].str.split( r'-{1,2}' ).str[1]

		long_xl_df = long_xl_df.reset_index( drop = True )
		return long_xl_df


	def select_fp_xls( self, inter_xls: pd.DataFrame ) -> pd.DataFrame:
		"""
		Select FP XLs with SASD >self.short_linker length+20 angstorm.
			We consider that XLs greater than the short linker lengtn+20 angstorm
				as false positives.
			These XLs can physically form but are implausible at the short
				linger length.
			Sort the FP XLs in descending order.
		XL file format: prot1,res1,prot2,res2

		Inputs:
		----------
		inter_xls: dataframe containing JWalk predicted
			interprotein XLs.

		Returns:
		----------
		fp_xl_df: dataframe containing FP XLs.
		"""
		fp_xl_df = pd.DataFrame()
		fp_xls = inter_xls.loc[inter_xls["SASD"] > self.short_linker+20]
		fp_xls = fp_xls.sort_values( by = "SASD", ascending = False )
		for i in [1, 2]:
			fp_xl_df[f"prot{i}"] = fp_xls[f"Atom{i}"].str.split( r'-{1,2}' ).str[2]
			fp_xl_df[f"res{i}"] = fp_xls[f"Atom{i}"].str.split( r'-{1,2}' ).str[1]

		fp_xl_df = fp_xl_df.reset_index( drop = True )
		return fp_xl_df







	# def get_tp_xls( self, xl_df: pd.DataFrame ) -> pd.DataFrame:
	# 	"""
	# 	Obtain true positive (TP) XLs.
	# 		XLs with distance <= max bound.
	# 	"""
	# 	tp_xl_df = copy.copy( xl_df )
	# 	# Using SASD provides more accurate XLs.
	# 	tp_xl_df = tp_xl_df.loc[tp_xl_df["SASD"] <= self.xl_max_bound]
	# 	return tp_xl_df


	# def get_fp_xls( self, xl_df: pd.DataFrame ) -> pd.DataFrame:
	# 	"""
	# 	JWalk provided XLs are all TPs. We consider XLs above a certain
	# 		SASD noisy XLs or FP.
	# 	Obtain false positive (FP) XLs.
	# 		We consider XLs with distance > the (max bound + 10A) as FP.
	# 		We further sort them in descending order based on SASD.
	# 	"""
	# 	fp_xl_df = copy.copy( xl_df )
	# 	# Using SASD provides more accurate XLs.
	# 	fp_xl_df = fp_xl_df.loc[fp_xl_df["SASD"] > self.xl_max_bound+10.0]
	# 	fp_xl_df.sort_values( by = "SASD", ascending = False, inplace = True )
	# 	return fp_xl_df


	# def get_interprotein_xls( self, df: pd.DataFrame ) -> pd.DataFrame:
	# 	"""
	# 	JWalk provides the following info:
	# 		Index, Model (input file name)
	# 		Atom1 --> AA-RES-CHAIN-CA
	# 		Atom2 --> AA-RES-CHAIN-CA
	# 			Atom field may have 1 or 2 hyphens.
	# 		SASD --> Solvent accessible surface distance.
	# 		Eculidean distance --> distance between CA atoms.
	# 	Fetch all inter-protein XLs for which the SASD is less than xl_max_bound.
	# 	Note: In some cases. JWalk adds "--" instead of "-" (e.g. 8sjj.
	# 	"""
	# 	inter = df[df["Atom1"].str.split( r'-{1,2}' ).str[2] != df["Atom2"].str.split( r'-{1,2}' ).str[2]]

	# 	# Extract chain ID and res no.
	# 	interprotein_xls = pd.DataFrame()
	# 	for i in [1, 2]:
	# 		interprotein_xls[f"prot{i}"] = inter[f"Atom{i}"].str.split( r'-{1,2}' ).str[2]
	# 		interprotein_xls[f"res{i}"] = inter[f"Atom{i}"].str.split( r'-{1,2}' ).str[1]

	# 	interprotein_xls = interprotein_xls.reset_index( drop = True )

	# 	return interprotein_xls


	# def get_xls_for_entry_id( self, entry_id: str ):
	# 	"""
	# 	Given an entry_ids, obtain inter-protein XLs.
	# 	JWalk needs the PDB file in the current dir to run.
	# 	Instead of moving to the PDB dir, I am copying the
	# 		PDB to the current dir and running JWalk.
	# 	This allows to parallelize JWalk.
	# 	"""
	# 	logs = {}
	# 	struct_file = os.path.join( self.pdb_struct_dir,
	# 								f"{entry_id}.{self.struct_format}" )

	# 	# Copy PDB file to current dir.
	# 	run_subprocess( ["cp", f"{struct_file}", "./"] )

	# 	self.run_jwalk( entry_id )
	# 	print( f"JWalk run complete for {entry_id}." )

	# 	xl_df = self.parse_jwalk_output( entry_id )

	# 	if xl_df is None:
	# 		tp_inter_xls, fp_inter_xls = None, None
	# 		logs["failed_to_run_jwalk"] = entry_id
	# 	else:
	# 		tp_xls = self.get_tp_xls( xl_df )
	# 		tp_inter_xls = self.get_interprotein_xls( tp_xls )
	# 		fp_xls = self.get_fp_xls( xl_df )
	# 		fp_inter_xls = self.get_interprotein_xls( fp_xls )

	# 		if tp_inter_xls.shape[0] == 0:
	# 			logs["no_inter_xls"]  = [entry_id]
	# 			tp_inter_xls, fp_inter_xls = None, None

	# 		elif tp_inter_xls.shape[0] < self.min_inter_xls:
	# 			logs["too_few_xls"]  = [entry_id]
	# 			tp_inter_xls, fp_inter_xls = None, None

	# 	# remove PDB file from current dir.
	# 	run_subprocess( ["rm", f"./{entry_id}.{self.struct_format}"] )
	# 	print( f"Completed for {entry_id}." )
	# 	return entry_id, tp_inter_xls, fp_inter_xls, logs


	# def get_xls_serially( self ):
	# 	"""
	# 	Obtain XLs serially for the given entry_id's.
	# 	"""
	# 	for idx, entry_id in enumerate( self.pdb_ids_list ):
	# 		entry_id, tp_inter_xls, fp_inter_xls, logs = self.get_xls_for_entry_id( entry_id )
	# 		if tp_inter_xls is None:
	# 			for k in logs:
	# 				self.jwalk_logs[k][0].extend( logs[k] )
	# 				self.jwalk_logs[k][1] += len( logs[k] )
	# 		else:
	# 			print( f"{idx}/{len( self.pdb_ids_list )} --> {entry_id}" )
	# 			print( f"TP inter-protein XLs = {tp_inter_xls.shape[0]}" +
	# 				f"\tFP inter-protein XLs = {fp_inter_xls.shape[0]}" )
	# 			self.xls_dict[entry_id] = {"tp_xls": tp_inter_xls,
	# 										"fp_xls": fp_inter_xls,
	# 										"count_tp_xls": tp_inter_xls.shape[0]}
	# 	run_subprocess( ["rm", "-r", f"./Jwalk_results/"] )

