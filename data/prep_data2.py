"""
Contains module to obtain metadat for benchmark dataset creation.
1. PDB IDs from any benchmark (AF_Unmasked, PINDER, SAbDab).
2. We need the following details about the complex:
	PDB structure (both .pdb and .cif)
		We want the SEQRES sequence.
		PDB residue numbers.

3. Obtain data: need support for both simulated and real.
	Simulated
		XLs -> JWalk
"""
from typing import List, Tuple, Dict
import os, glob, copy, time, warnings
import numpy as np
import pandas as pd

from utils.utils import ( run_subprocess,
							open_file_handler,
							read_json, write_json )
from api_data_modules import( DownloadPdbStructure,
								SeqResDict )

from simulate_data import ( SimulateCrosslinks ) 


class Metadata():
	"""
	Obtain all required metadata for the benchmark dataset.
	"""
	def __init__( self ):
		self.benchmark_name = "xlsim"  # "xlsim", "abag"

		self.dataset_configs = {
			"global": {
				"benchmark_name": self.benchmark_name,
				"struct_format": "both",
				"max_sys_length": 1400,
				"frac_coverage": 0.99,
				"cores": 50,
				"max_trials": 5,
				"wait_time": 10
			},
			"jwalk": {
				"enabled": True,
				"xl_max_bound": 30,
				"min_inter_xls": 5,
				"max_allowed": 80,
				"add_fp": False,
				"frac_fp": 0.1
			}
		}
		self.seqres_dict = {}
		# Dict containing mapping  between pdb_seq_num and seq_id.
		self.pdb_num_seq_id_map = {}
		# Dict to store inter-protein XLs.
		self.xls_dict = {}

		self.logs = {}


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.create_required_dir()

		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )

		self.get_pdb_ids_from_input()

		self.run_dataset_creation_pipeline()
		self.save_dataset_configs()
		self.write_logs_to_csv()
		print( "\n May the Force be with you..." )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def create_required_paths( self ):
		"""
		Create paths for all required directories and files.
		"""
		## --------------------------
		# Global paths
		## --------------------------
		# PDB benchmark from AF Unmasked paper.
		self.afu_pdb_benchmark = os.path.join( "../raw/af_unmasked_pdb_benchmark.txt" )
		# SAbDab dataset.
		self.sabdab_input_file = os.path.join( "../raw/sabdab_seqid70_res4.tsv" )
		# FoldBench antigen-antibody dataset.
		self.foldbench_ab_ag_input_file = os.path.join( "../raw/interface_antibody_antigen.csv" )
		# FoldBench protein-protein dataset.
		self.foldbench_prot_prot_input_file = os.path.join( "../raw/interface_protein_protein.csv" )
		# FoldBench protein-protein dataset.
		self.foldbench_prot_pep_input_file = os.path.join( "../raw/interface_protein_peptide.csv" )

		# Base directory for all benchmarks.
		self.base_dir = os.path.join( os.path.abspath( "../benchmark/" ) )
		# Benchmark specific dir.
		self.benchmark_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_benchmark" )
		# Dir to store benchmark metadata including PDB API files,
		# 	structure file, SIFTS mapping, benchamrk csv and the required intermediate files.
		self.meta_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_metadata" )

		# PDB IDs remaining after metadat collection.
		self.benchmark_pdbs_file =  os.path.join( self.meta_dir,
										f"{self.benchmark_name}_benchmark_pdb_ids.txt" )
		# Logs file path.
		self.logs_file = os.path.join( self.meta_dir, f"Logs_{self.benchmark_name}.json" )
		self.logs_csv_file = os.path.join( self.meta_dir, f"Logs_{self.benchmark_name}.csv" )
		self.dataset_configs_file = os.path.join( self.meta_dir,
								f"Dataset_configs_{self.benchmark_name}.json" )

		## --------------------------
		# For DownloadPdbStructure module
		## --------------------------
		# Dir contaiing PDB structure.
		self.pdb_struct_dir = os.path.join( self.meta_dir, "struct" )
		self.pdb_struct_dwnld_file = os.path.join( self.meta_dir,
												"pdb_struct_dwnld.txt" )

		## --------------------------
		# For SeqResDict module
		## --------------------------
		self.seqres_dict_file = os.path.join( self.meta_dir, "seqres_dict.npy" )
		self.pdb_num_seq_id_map_file = os.path.join( self.meta_dir, "pdb_num_seq_id_map.npy" )

		## --------------------------
		# For Simulated data
		## --------------------------
		self.xls_dict_file = os.path.join( self.meta_dir, "jwalk_xls.npy" )



	def create_required_dir( self ):
		"""
		Create the required directories if not already existing.
		"""
		os.makedirs( self.base_dir, exist_ok = True )
		os.makedirs( self.meta_dir, exist_ok = True )
		os.makedirs( self.pdb_struct_dir, exist_ok = True )
		os.makedirs( self.benchmark_dir, exist_ok = True )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def get_pdb_ids_from_input( self ):
		"""
		Obtain the PDB IDs from the input files of the required benchmark.
		"""
		# if self.benchmark_name == "afu":
		# 	print( "Using PDB benchmark from AFUnmasked..." )
		# 	self.benchmark_pdb_ids_list = self.parse_pdb_afu_benchmark()
		# elif self.benchmark_name == "sabdab":
		# 	print( "Using Antigen-Antibody complexes from SAbDab..." )
		# 	self.benchmark_pdb_ids_list = self.parse_sabdab_benchmark()
		# elif self.benchmark_name in ["fbabag", "fbpp"]:
		# 	print( "Using Foldbench benchmark..." )
		# 	self.benchmark_pdb_ids_list = self.parse_foldbench_benchmark()
		if self.benchmark_name == "xlsim":
			self.benchmark_pdb_ids_list = self.get_pdb_ids_for_xl_benchmark()
		elif self.benchmark_name == "abag":
			self.benchmark_pdb_ids_list = self.get_pdb_ids_for_abag_benchmark()
		else:
			raise ValueError( "Unsupported benchmark specified..." )


	def get_pdb_ids_for_xl_benchmark( self ) -> List:
		"""
		For the simulated XL benchmark, obtain PDB IDs from:
			PDB benchmark from AFUnmasked
			Protein-protein and protein-peptide benchmark from FoldBench
		"""
		afu = self.parse_pdb_afu_benchmark()
		print( f"PDB IDs from AF Unmasked PDB benchmark: {len( afu )}" )
		# fb_prot_prot = self.parse_foldbench_benchmark( "prot_prot" )
		# print( f"PDB IDs from FoldBench protein-protein benchmark: {len( fb_prot_prot )}" )
		# fb_prot_pep = self.parse_foldbench_benchmark( "prot_pep" )
		# print( f"PDB IDs from FoldBench protein-peptide benchmark: {len( fb_prot_pep )}" )

		# Remove duplicate PDB IDs.
		# pdb_ids = sorted( list( set( afu + fb_prot_prot + fb_prot_pep ) ) )
		pdb_ids = sorted( list( set( afu ) ) )

		return pdb_ids


	def get_pdb_ids_for_abag_benchmark( self ) -> List:
		"""
		For the simulated antigen-antibody XL benchmark, obtain PDB IDs from:
			SAbDab database
			Antigen-Antibody benchmark from FoldBench
		"""
		sabdab = self.parse_sabdab_benchmark()
		print( f"PDB IDs from SAbDab benchmark: {len( sabdab )}" )
		fb_ab_ag = self.parse_foldbench_benchmark( "ab_ag" )
		print( f"PDB IDs from FoldBench Ab-Ag benchmark: {len( fb_ab_ag )}" )

		# Remove duplicate PDB IDs.
		pdb_ids = sorted( list( set( sabdab + fb_ab_ag ) ) )
		# pdb_ids = sorted( list( set( sabdab ) ) )

		return pdb_ids


	def parse_pdb_afu_benchmark( self ) -> List:
		"""
		Using the PDB benchmark provided in the AF Unmasked paper.
		"""
		fh = open_file_handler( self.afu_pdb_benchmark, "r" )
		pdb_ids = fh.readlines()[0].strip().split( "," )
		fh.close()
		pdb_ids = [id_.lower() for id_ in pdb_ids]

		return pdb_ids


	def parse_sabdab_benchmark( self ) -> List:
		"""
		A .tsv file was download from SAbDab database
			(https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab)
			using the following filters:
				Non-redundant at 60% sequence identity.
				In complex: bound-only
				resolution cutoff: 3.0
		Select entries for which the antigen type is protein/peptide.
		"""
		df = pd.read_csv( self.sabdab_input_file, sep = "\t" )
		groups = df.groupby( ["pdb"] )

		pdb_ids = []
		for group in groups:
			pdb = group[0]
			# Must have atleast 1 protein/peptide antigen.
			ag_type = set( group[1]["antigen_type"]
				).intersection( set( ["protein", "peptide"] ) )
			if len( ag_type ) != 0:
				pdb_ids.append( pdb[0].lower() )
		return pdb_ids


	def parse_foldbench_benchmark( self, target: str ) -> List:
		"""
		.csv file was obtained from https://github.com/BEAM-Labs/FoldBench.git
		Will parse the following as specified:
			Antigen-antibody benchmark.
			Protein-protein benchmark.
			Protein-peptide benchmark.
		"""
		if target == "prot_prot":
			df = pd.read_csv( self.foldbench_prot_prot_input_file )
		elif target == "prot_pep":
			df = pd.read_csv( self.foldbench_prot_pep_input_file )
		elif target == "ab_ag":
			df = pd.read_csv( self.foldbench_ab_ag_input_file )
		else:
			raise ValueError( "Incorrect FoldBench target name specified. " +
							"Supported: prot_prot, prot_pep, ab_ag" )
		pdb_ids = df["pdb_id"].str.split( "-" ).str[0].tolist()

		return pdb_ids


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_dataset_creation_pipeline( self ):
		"""
		Pipeline all modules for benchamrk dataset creation.
		Get .pdb and .cif structures for all complexes.
		Get data - simulated or real.
		"""
		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- Download PDB Structures ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_download_pdb_structure_module()

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- MMCIF Dict ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_seqres_dict_module()
		print( f"PDB IDs remaining: {len( self.benchmark_pdb_ids_list )}" )

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- Simulated Data ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.simulate_experimental_data()

		w = open_file_handler( self.benchmark_pdbs_file, "w" )
		w.writelines( ",".join( list( self.xls_dict.keys() ) ) )
		w.close()


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_download_pdb_structure_module( self ):
		"""
		Download .pdb and .cif structures for all complexes.
		"""
		if os.path.exists( self.pdb_struct_dwnld_file ):
			print( "Structures for benchmark already downloaded..." )
			f = open_file_handler( self.pdb_struct_dwnld_file, "r" )
			self.benchmark_pdb_ids_list = f.readlines()[0].split( "," )
			f.close()

		else:
			obj = DownloadPdbStructure(
				pdb_ids_list = self.benchmark_pdb_ids_list,
				pdb_struct_dir = self.pdb_struct_dir,
				struct_format = self.dataset_configs["global"]["struct_format"],
				cores = self.dataset_configs["global"]["cores"],
				max_trials = self.dataset_configs["global"]["max_trials"],
				wait_time = self.dataset_configs["global"]["wait_time"]
				)
			obj.forward()

			self.benchmark_pdb_ids_list = copy.copy( obj.downloaded_struct )
			self.logs["DownloadPdbStructure"] = copy.deepcopy( obj.struct_logs )

			del obj
			
			w = open_file_handler( self.pdb_struct_dwnld_file, "w" )
			w.writelines( ",".join( self.benchmark_pdb_ids_list ) )
			w.close()
			write_json( self.logs, self.logs_file )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_seqres_dict_module( self ):
		"""
		Obtain the cif dict containing entity_id, chain_id, seq,
			and res no. for all entries.
		"""
		if os.path.exists( self.seqres_dict_file ):
			print( "CIF dict already present..." )
			self.seqres_dict = np.load( self.seqres_dict_file,
										allow_pickle = True ).item()
			self.pdb_num_seq_id_map = np.load( self.pdb_num_seq_id_map_file,
										allow_pickle = True ).item()

		else:
			obj = SeqResDict(
				pdb_ids_list = self.benchmark_pdb_ids_list,
				pdb_struct_dir = self.pdb_struct_dir,
				cores = self.dataset_configs["global"]["cores"],
				max_sys_length = self.dataset_configs["global"]["max_sys_length"],
				frac_coverage = self.dataset_configs["global"]["frac_coverage"]
				)
			obj.forward()

			self.seqres_dict = copy.deepcopy( obj.seqres_dict )
			self.logs["SeqResDict"] = copy.deepcopy( obj.cif_logs )

			del obj

			self.create_pdb_num_to_seq_id_mapping()
			
			np.save( self.seqres_dict_file,
					self.seqres_dict,
					allow_pickle = True )
			np.save( self.pdb_num_seq_id_map_file,
					self.pdb_num_seq_id_map,
					allow_pickle = True )

			write_json( self.logs, self.logs_file )

		self.benchmark_pdb_ids_list = list( self.seqres_dict.keys() )


	def create_pdb_num_to_seq_id_mapping( self ):
		"""
		Craete a mapping between the seq_id and pdb_seq_num obtained
			from the .cif file.
		This is because pdb_seq_num may be discontinous in some cases
			(8g0q_B, 8g0q_D) however, seq_id is always continous.
		pdb_id: {
			"chain_id": dict( zip( pdb_seq_num, seq_id ) )
		}
		"""
		for pdb_id in self.seqres_dict:
			self.pdb_num_seq_id_map[pdb_id] = {}
			for entity_id in self.seqres_dict[pdb_id]:
				for chain_id in self.seqres_dict[pdb_id][entity_id]:
					chain = self.seqres_dict[pdb_id][entity_id][chain_id]
					start_seq_id = chain["start_seq_id"]
					end_seq_id = chain["end_seq_id"]

					seq_id = chain["seq_id"]
					pdb_seq_num = chain["res_num"]
					seq = chain["seq"]

					if ( end_seq_id-start_seq_id+1 ) != len( seq ):
						raise ValueError( f"Mismatch in seq_id and sequence for PDB: {pdb_id}..." )
					if len( seq_id ) != len( pdb_seq_num ):
						raise ValueError( f"Mismatch in seq_id and pdb_seq_num for PDB: {pdb_id}..." )
					self.pdb_num_seq_id_map[pdb_id][chain_id] = dict( zip( pdb_seq_num, seq_id ) )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def simulate_experimental_data( self ):
		"""
		Simulate experimental dat for the required protein complexes.
		Currently supporting XL data (JWalk).
		"""
		if self.dataset_configs["jwalk"]["enabled"]:
			self.simulate_xls()


	def simulate_xls( self ):
		"""
		Simulate XL data using JWalk.
		Obtain inter-protein XLs.
		Map residue numbers to UniProt numbering based on SIFTS mapping.
		"""
		if os.path.exists( self.xls_dict_file ):
			print( "JWalk simulated XLs already exist..." )
			self.xls_dict = np.load( self.xls_dict_file,
							allow_pickle = True ).item()
		else:
			obj = SimulateCrosslinks(
				pdb_ids_list = self.benchmark_pdb_ids_list,
				pdb_struct_dir = self.pdb_struct_dir,
				struct_format = "pdb",
				xl_max_bound = self.dataset_configs["jwalk"]["xl_max_bound"],
				min_inter_xls = self.dataset_configs["jwalk"]["min_inter_xls"],
				cores = self.dataset_configs["global"]["cores"],
				)
			obj.forward()

			self.xls_dict = copy.deepcopy( obj.xls_dict )
			self.logs["Jwalk"] = copy.deepcopy( obj.jwalk_logs )

			del obj

			np.save( self.xls_dict_file,
					self.xls_dict,
					allow_pickle = True )

		self.map_xls_to_seq_id()
		np.save( self.xls_dict_file,
				self.xls_dict,
				allow_pickle = True )
		write_json( self.logs, self.logs_file )


	def map_xls_to_seq_id( self ):
		"""
		Given a dataframe for inter-protein XLs, map the pdb_seq_num
			to the corresponding seq_id.
		Columns: Protein1, Residue1, Protein1, Residue1
		"""
		for pdb_id in self.xls_dict:
			for xl_type in ["tp_xls", "fp_xls"]:
				df = self.xls_dict[pdb_id][xl_type]
				drop_index = []
				for i in df.index:
					chain1 = df.iloc[i, 0]
					res1 = int( df.iloc[i, 1] )
					chain2 = df.iloc[i, 2]
					res2 = int( df.iloc[i, 3] )

					chain1_map = self.pdb_num_seq_id_map[pdb_id][chain1]
					chain2_map = self.pdb_num_seq_id_map[pdb_id][chain2]

					# Ignore XLs for residues not in pdb_seq_num
					# 	(not selected for modeling).
					if res1 not in chain1_map:
						drop_index.append( i )
						continue
					if res2 not in chain2_map:
						drop_index.append( 2 )
						continue

					df.iloc[i, 1] = chain1_map[res1]
					df.iloc[i, 3] = chain2_map[res2]
				df = df.drop( drop_index )
				df = df.reset_index( drop = True )


	def save_dataset_configs( self ):
		"""
		Save the datset configs to a JSON file on disk.
		"""
		write_json( self.dataset_configs, self.dataset_configs_file )


	def write_logs_to_csv( self ):
		"""
		Write the logs dict to a csv file.
		"""
		flat_dict = {k:[] for k in ["Module",
									"Description",
									"Count"]}

		for module in self.logs:
			for desc in self.logs[module]:
				desc_val = self.logs[module][desc]
				flat_dict["Module"].append( module )
				flat_dict["Description"].append( desc )
				if isinstance( desc_val, List ):

					val = desc_val[1]
				else:
					val = desc_val
				flat_dict["Count"].append( val )
			[flat_dict[k].append( "" ) for k in flat_dict]
		df = pd.DataFrame( flat_dict )
		df.to_csv( self.logs_csv_file, index = False )


if __name__ == "__main__":
	Metadata().forward()
