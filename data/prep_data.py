"""
Script to obtain and create the required files
	for the benchmark dataset.
Create the input files required for modeling, given the PDB ID.

1. PDB IDs from any benchmark (AF_Unmasked, PINDER).
2. We need the following details about the complex:
	Entry/Entity data from PDB API
	SIFTS mapping
	PDB structure
	UniProt sequences
Remove PDB IDs if:
	Could not retrieve data from PDB.
	Complex contains non-protein entities
	Total system length >1400 residues.
	Monomer sequence.
	No SIFTS mapping.
	Missing residues present in mapping.
		It's OK if missing residues only at termini?
3. Obtain data: need support for both simulated and real.
	Simulated
		XLs -> JWalk
Remove complexes if:
	Couldn't run the tool.
	No data (e.g. inter-protein XLs) obtained.
4. Create system config dict for all systems.

"""
from typing import List, Tuple, Dict
import os, glob, copy, time
from collections import defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import tqdm

from utils.utils import ( run_subprocess,
							open_file_handler,
							read_json, write_json )
from utils.pdb_utils import ( get_distance_map, Parser )
from utils.api_utils import ( get_uniprot_seq,
								download_pdb )
from api_data_modules import( PdbData, SiftsMapping,
							DownloadUniprotSequences,
							DownloadPdbStructure )

from simulate_data import ( SimulateCrosslinks ) 

class Metadata():
	"""
	Obtain all required metadata for the benchmark dataset.
	"""
	def __init__( self ):
		self.benchmark_name = "afu"

		self.dataset_configs = {
			"global": {
				"benchmark_name": self.benchmark_name,
				"struct_format": "pdb",
				"max_sys_length": 1400,
				"frac_coverage": 0.99,
				"cores": 50,
				"max_trials": 5,
				"wait_time": 10
			},
			"jwalk": {
				"enabled": True,
				"xl_max_bound": 35,
				"min_inter_xls": 10
			}
		}

		# Dict containing info. from PDB REST API.
		self.pdb_data_dict = {}
		# Dict containing residue-level PDB to UniProt mapping.
		self.sifts_data_dict = {}
		# Dict to store all UniProt sequences.
		self.uni_seq_dict = {}
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

		self.benchmark_pdb_ids_list = self.parse_pdb_afu_benchmark()

		self.run_dataset_creation_pipeline()


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

		# Base directory for all benchmarks.
		self.base_dir = os.path.join( os.path.abspath( "../benchmark/" ) )
		# Benchmark specific dir.
		self.benchmark_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_benchmark" )
		# Dir to store benchmark metadata including PDB API files,
		# 	structure file, SIFTS mapping, benchamrk csv and the required intermediate files.
		self.meta_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_metadata" )

		# .csv file to store relevant details for the benchmark complexes.
		self.output_benchmark_csv = os.path.join( self.meta_dir, f"benchmark.csv" )
		# Logs file path.
		self.logs_file = os.path.join( self.meta_dir, f"Logs_{self.benchmark_name}.json" )

		## --------------------------
		# For PdbData module
		## --------------------------
		# Dir containing PDB entry and entity dicts.
		self.pdb_api_dir = os.path.join( self.meta_dir, "pdb_api" )
		self.pdb_data_dict_file = os.path.join( self.meta_dir,
												"pdb_api_dict.npy" )

		## --------------------------
		# For SiftsMapping module
		## --------------------------
		# Dict to store PDB API info.
		self.pdb_data_dict = {}
		# Dir containing SIFTS XML file.
		self.sifts_xml_dir = os.path.join( self.meta_dir, "sifts_xml" )
		# Dir containing processed SIFTS mapping.
		self.sifts_dict_dir = os.path.join( self.meta_dir, "sifts_dict" )
		self.sifts_dict_file = os.path.join( self.meta_dir,
											"sifts_mapping_dict.npy" )

		## --------------------------
		# For DownloadUniprotSequences module
		## --------------------------
		# UniProt seq dict file path.
		self.uni_seq_dict_file = os.path.join(
											self.meta_dir,
											f"uni_seq.json" )

		## --------------------------
		# For DownloadPdbStructure module
		## --------------------------
		# Dir contaiing PDB structure.
		self.pdb_struct_dir = os.path.join( self.meta_dir, "struct" )
		self.pdb_struct_dwnld_file = os.path.join( self.meta_dir,
												"pdb_struct_dwnld.txt" )

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
		os.makedirs( self.pdb_api_dir, exist_ok = True )
		os.makedirs( self.pdb_struct_dir, exist_ok = True )
		os.makedirs( self.benchmark_dir, exist_ok = True )
		os.makedirs( self.sifts_xml_dir, exist_ok = True )
		os.makedirs( self.sifts_dict_dir, exist_ok = True )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def parse_pdb_afu_benchmark( self ) -> List:
		"""
		Using the PDB benchmark provided in the AF Unmasked paper.
		"""
		fh = open_file_handler( self.afu_pdb_benchmark, "r" )
		pdb_ids = fh.readlines()[0].strip().split( "," )
		fh.close()

		print( f"PDB IDs from AF Unmasked PDB benchmark: {len( pdb_ids )}" )
		return pdb_ids



	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_dataset_creation_pipeline( self ):
		"""
		Pipeline all modules for benchamrk dataset creation.
		Get PDB API info.
		Get SIFTS mapping.
		Get Uniprot sequences.
		Get structures for all complexes.
		Get data - simulated or real.
		"""
		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- PDB API Data ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_pdb_api_module()
		self.transition_pdb_to_sifts_module()

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- SIFTS Mapping ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_sifts_mapping_module()
		uni_ids_list = self.transition_sifts_to_uni_seq_module()

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- Download UniProt Sequences ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_download_uniprot_sequence_module( uni_ids_list )
		self.transition_uni_seq_to_struct_module()

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- Download PDB Structures ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_download_pdb_structure_module()

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- Simulated Data ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.simulate_experimental_data()



	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_pdb_api_module( self ):
		"""
		Get required info for all PDB IDs (entry_id).
		Save the API data dict and the logs on disk.
		If data dict already present do not run again.
		"""
		if os.path.exists( self.pdb_data_dict_file ):
			print( "PDB API data dict already exist..." )
			self.pdb_data_dict = np.load( self.pdb_data_dict_file,
										allow_pickle = True ).item()

		else:
			obj = PdbData(
				pdb_ids_list = self.benchmark_pdb_ids_list,
				pdb_api_dir = self.pdb_api_dir,
				pdb_data_dict_file = self.pdb_data_dict_file,
				cores = self.dataset_configs["global"]["cores"],
				max_trials = self.dataset_configs["global"]["max_trials"],
				wait_time = self.dataset_configs["global"]["wait_time"]
				)
			obj.forward()

			self.pdb_data_dict = copy.deepcopy( obj.pdb_data_dict )
			self.logs["PdbData"] = copy.deepcopy( obj.pdb_logs )

			del obj

			np.save( self.pdb_data_dict_file, self.pdb_data_dict, allow_pickle = True )
			write_json( self.logs, self.logs_file )



	def transition_pdb_to_sifts_module( self ) -> List:
		"""
		Remove the PDB IDs from the initial set of IDs that
			were excluded in PdbData module.
		"""
		self.benchmark_pdb_ids_list = sorted( list( 
			set( self.benchmark_pdb_ids_list ).intersection( set( self.pdb_data_dict.keys() ) )
		) )



	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_sifts_mapping_module( self ):
		"""
		Obtain PDB-Uniprot mapping using SIFTS.
		"""
		if os.path.exists( self.sifts_dict_file ):
			print( "SIFTS mapping dict already exist..." )
			self.sifts_data_dict = np.load( self.sifts_dict_file,
										allow_pickle = True ).item()

		else:
			obj = SiftsMapping(
				pdb_ids_list = self.benchmark_pdb_ids_list,
				sifts_xml_dir = self.sifts_xml_dir,
				sifts_dict_dir = self.sifts_dict_dir,
				sifts_dict_file = self.sifts_dict_file,
				max_sys_length = self.dataset_configs["global"]["max_sys_length"],
				frac_coverage = self.dataset_configs["global"]["frac_coverage"],
				cores = self.dataset_configs["global"]["cores"],
				max_trials = self.dataset_configs["global"]["max_trials"],
				wait_time = self.dataset_configs["global"]["wait_time"]
				)
			obj.forward()

			self.sifts_data_dict = copy.deepcopy( obj.sifts_data_dict )
			self.logs["SiftsMapping"] = copy.deepcopy( obj.sifts_logs )

			del obj

			np.save( self.sifts_dict_file,
					self.sifts_data_dict,
					allow_pickle = True )
			write_json( self.logs, self.logs_file )



	def transition_sifts_to_uni_seq_module( self ) -> List:
		"""
		Remove the PDB IDs that were excuded in the SiftsMapping module.
		Remove PDB IDs for which the mapped Uniprot Id does not
			match the Uniprot ID obtained from PDB.
		Obtain all Uniprot IDs for all selected PDB IDs.
		"""
		pdb_ids_list = []
		uni_ids_list = []
		for pdb_id in self.sifts_data_dict:
			pdb_uni_id = self.pdb_data_dict[pdb_id]["uniprot_ids"].split( "," )
			auth_asym_id = self.pdb_data_dict[pdb_id]["auth_asym_ids"].split( "," )

			tmp_uni_ids = []
			passed = True
			for i, auth_id in enumerate( auth_asym_id ):
				for aa_id in auth_id.split( "-" ):
					if aa_id not in self.sifts_data_dict[pdb_id]["mapping"]:
						passed = False
						break
					mapped_uni_id = self.sifts_data_dict[pdb_id]["mapping"][aa_id]["uni_id"]

					# Ignore a PDB ID if any single chain is mapped to incorrect Uni ID.
					if mapped_uni_id != pdb_uni_id[i]:
						passed = False
						break
					else:
						tmp_uni_ids.append( pdb_uni_id[i] )

			if passed:
					uni_ids_list.extend( tmp_uni_ids )
					pdb_ids_list.append( pdb_id )

		self.benchmark_pdb_ids_list = pdb_ids_list

		return uni_ids_list


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_download_uniprot_sequence_module( self, uni_ids_list: List ):
		"""
		Obtain PDB-Uniprot mapping using SIFTS.
		"""
		if os.path.exists( self.uni_seq_dict_file ):
			print( "Uniprot seq dict already exist..." )
			self.uni_seq_dict = read_json( self.uni_seq_dict_file )

		else:
			obj = DownloadUniprotSequences(
				uni_ids_list = uni_ids_list,
				uni_seq_dict_file = self.uni_seq_dict_file,
				cores = self.dataset_configs["global"]["cores"],
				max_trials = self.dataset_configs["global"]["max_trials"],
				wait_time = self.dataset_configs["global"]["wait_time"]
				)
			obj.forward()

			self.uni_seq_dict = copy.deepcopy( obj.uni_seq_dict )
			self.logs["DownloadUniprotSequence"] = copy.deepcopy( obj.uni_logs )

			del obj

			write_json( self.uni_seq_dict, self.uni_seq_dict_file )
			write_json( self.logs, self.logs_file )



	def transition_uni_seq_to_struct_module( self ) -> List:
		"""
		Remove the PDB IDs containing Uniprot IDs that were excluded.
		For efficiency in filtering PDB IDs, we use the no. of excluded
			Uniprot IDs, assuming the latter will be lower than all selected
			Uniprot IDs.
		"""
		excluded_uni_ids = self.logs["DownloadUniprotSequence"]["uni_seq_not_found"]

		excluded_pdb_ids = []
		for pdb_id in self.benchmark_pdb_ids_list:
			uni_ids = self.pdb_data_dict[pdb_id]["uniprot_ids"].split( "," )

			# If any Uniprot ID for a PDB is not downloaded.
			if any( [uid in excluded_uni_ids for uid in uni_ids] ):
				excluded_pdb_ids.append( pdb_id )


		self.benchmark_pdb_ids_list = sorted( list( 
			set( self.benchmark_pdb_ids_list ) - set( excluded_pdb_ids )
		) )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_download_pdb_structure_module( self ):
		"""
		Obtain PDB-Uniprot mapping using SIFTS.
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
				struct_format = self.dataset_configs["global"]["struct_format"],
				xl_max_bound = self.dataset_configs["jwalk"]["xl_max_bound"],
				min_inter_xls = self.dataset_configs["jwalk"]["min_inter_xls"],
				cores = self.dataset_configs["global"]["cores"],
				)
			obj.forward()

			self.xls_dict = copy.deepcopy( obj.xls_dict )
			self.logs["jwalk"] = copy.deepcopy( obj.jwalk_logs )

			del obj

			self.map_xls_to_uni_pos()

			np.save( self.xls_dict_file,
					self.xls_dict,
					allow_pickle = True )
			write_json( self.logs, self.logs_file )


	def map_xls_to_uni_pos( self ):
		"""
		Given a dataframe for inter-protein XLs, map the PDB positions
			to the corresponding UniProt positions.
		Columns: Protein1, Residue1, Protein1, Residue1
		"""
		for pdb_id in self.xls_dict:
			df = self.xls_dict[pdb_id]["xls"]
			for i in range( df.shape[0] ):
				chain1 = df.loc[i, "prot1"]
				res1 = int( df.loc[i, "res1"] )
				chain2 = df.loc[i, "prot2"]
				res2 = int( df.loc[i, "res2"] )

				df.iloc[i, 1] = self.sifts_pdb_to_uni[pdb_id]["mapping"][chain1][res1]
				df.iloc[i, 3] = self.sifts_pdb_to_uni[pdb_id]["mapping"][chain2][res2]


	def write_logs_to_csv( self ):
		"""
		Write the logs dict to a csv file.
		"""
		flat_dict = [k:[] for k in ["Module",
									"Description",
									"Count"]]

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



	# def get_entry_entity_files( self, entry_id: str ) -> Tuple[str, str]:
	# 	"""
	# 	Return the file paths for the entry and entity dict.
	# 	"""
	# 	entry_file = os.path.join( self.pdb_api_dir, f"entry_dict_{entry_id}.json" )
	# 	entity_file = os.path.join( self.pdb_api_dir, f"entity_dict_{entry_id}.json" )
	# 	return entry_file, entity_file


	# def save_entry_entity_dict( self, rest_api: PdbRestApi,
	# 							entry_file: str,
	# 							entity_file: str ):
	# 	"""
	# 	Save the entry and entity dict on disk.
	# 	Assuming the entity dict contains info. for all entities to be saved.
	# 	"""
	# 	if not os.path.exists( entry_file ):
	# 		write_json( rest_api.entry_data, entry_file )
	# 	if not os.path.exists( entity_file ):
	# 		write_json( rest_api.entity_data, entity_file )


	# def instantiate_pdb_rest_api( self, entry_id: str,
	# 								entry_file: str,
	# 								entity_file: str ) -> PdbRestApi:

	# 	"""
	# 	Instantiate the PDB REST API object.
	# 	"""
	# 	rest_api = PdbRestApi( entry_id = entry_id,
	# 							entry_file = entry_file,
	# 							entity_file = entity_file )
	# 	return rest_api


	# def entry_from_pdb_rest_api( self, entry_id: str ):
	# 	"""
	# 	Given a PDB ID, fetch the following info. from the PDB REST API:
	# 		All entities.
	# 		All polymer entities.
	# 		UniProt IDs.
	# 		Auth asym IDs.
	# 		Stoichiometry.
	# 		Aligned UniProt start-end position.
	# 		Aligned PDB start-end position.
	# 	entry_id corresponds to the PDB ID.
	# 	"""
	# 	entry_file, entity_file = self.get_entry_entity_files( entry_id )

	# 	# Instantiate the PDB REST API object.
	# 	rest_api = self.instantiate_pdb_rest_api( entry_id,
	# 												entry_file,
	# 												entity_file )
	# 	# Get all entity IDs in the.
	# 	all_entities = rest_api.get_all_entities()
	# 	# Get all polymer entity IDs in the.
	# 	polymer_entities = rest_api.get_polymer_entities()

	# 	pdb_dict = {k:[] for k in [
	# 							"entity_ids", "polymer_entity_ids", "uniprot_ids",
	# 							"stoichiometry", "uni_pos", "pdb_pos"]}
	# 	stoichiometry = []
	# 	uniprot_ids = []
	# 	auth_asym_ids = []
	# 	uni_pos, pdb_pos = [], []
	# 	total_length = 0
	# 	for entity_id in polymer_entities:
	# 		rest_api.retrieve_polymer_entity_data( entity_id )
	# 		uniprot_ids.extend( rest_api.get_uniprot_ids_for_entity( entity_id ) )
	# 		asym_ids = rest_api.get_asym_ids_for_entity( entity_id )
	# 		auth_asym_ids.append( "-".join( rest_api.get_auth_asym_ids_for_entity( entity_id ) ) )
	# 		pos, length = rest_api.get_poly_entity_align( entity_id )
	# 		uni_pos.append( pos[0] )
	# 		pdb_pos.append( pos[0] )
	# 		# Account for all copies of an entity.
	# 		total_length += length*len( asym_ids )

	# 		stoichiometry.append( f"{len( asym_ids )}" )

	# 	pdb_dict["entity_ids"] = ",".join( all_entities )
	# 	pdb_dict["polymer_entity_ids"] = ",".join( polymer_entities )
	# 	pdb_dict["uniprot_ids"] = ",".join( uniprot_ids )
	# 	pdb_dict["auth_asym_ids"] = ",".join( auth_asym_ids )
	# 	pdb_dict["stoichiometry"] = "-".join( stoichiometry )
	# 	pdb_dict["uni_pos"] = ",".join( uni_pos )
	# 	pdb_dict["pdb_pos"] = ",".join( pdb_pos )
	# 	pdb_dict["total_length"] = total_length

	# 	self.save_entry_entity_dict( rest_api,
	# 								entry_file,
	# 								entity_file )
	# 	return entry_id, pdb_dict


	# def where_the_magic_happens( self, benchamrk_pdb_ids: List ):
	# 	"""
	# 	For all the benchmark PDB IDs, get the required info.
	# 	(See self.entry_from_pdb_rest_api doc-str)
	# 	"""
	# 	for idx, pdb_id in enumerate( benchamrk_pdb_ids ):
	# 		# Skip obsolete PDB IDs - 8h4x.
	# 		# Skip PDBs with insertion code - 8a82, 7yls
	# 		if pdb_id in ["8h4x", "8a82", "7yls"]:
	# 			continue
	# 		print( f"{idx} --> {pdb_id}" )

	# 		pdb_id, pdb_dict = self.entry_from_pdb_rest_api( pdb_id )

	# 		self.pdb_benchmark_dict[pdb_id] = {}
	# 		for k in pdb_dict:
	# 			self.pdb_benchmark_dict[pdb_id][k] = pdb_dict[k]
	# 	print( "Info. obtained for PDB entries: ", len( self.pdb_benchmark_dict ) )


	# def filter_benchmark( self ):
	# 	"""
	# 	Remove a PDB entry if:
	# 		1. It contains a non-polymeric entity.
	# 			entity_ids > polymer_entity_ids.
	# 		2. No. of UniProt IDs does not match the no. of polymer_entity_ids.
	# 		3. No aligned PDB-UniProt positions available.
	# 		4. Total system length > 1450.
	# 	"""
	# 	selected_pdb_ids = []
	# 	c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0
	# 	for pdb_id in self.pdb_benchmark_dict:
	# 		# entity_ids = self.pdb_benchmark_dict[pdb_id]["entity_ids"].split( "," )
	# 		polymer_entity_ids = self.pdb_benchmark_dict[pdb_id]["polymer_entity_ids"].split( "," )
	# 		uniprot_ids = self.pdb_benchmark_dict[pdb_id]["uniprot_ids"].split( "," )
	# 		uni_pos = self.pdb_benchmark_dict[pdb_id]["uni_pos"].split( "," )
	# 		pdb_pos = self.pdb_benchmark_dict[pdb_id]["pdb_pos"].split( "," )

	# 		# Removing non-polymeric entities.
	# 		# if len( entity_ids ) > len( polymer_entity_ids ):
	# 		# 	c1 += 1
	# 		# 	continue

	# 		if len( polymer_entity_ids ) < len( uniprot_ids ):
	# 			c2 += 1
	# 			continue

	# 		if len( polymer_entity_ids ) > len( uniprot_ids ):
	# 			c3 += 1
	# 			continue

	# 		# Aligned PDB-UniProt positiosn not present.
	# 		# if any( [len( u ) == 0 for u in uni_pos] ) or any( [len( p ) == 0 for p in pdb_pos] ):
	# 		if not all( uni_pos ) and not all( pdb_pos ):
	# 			c4 += 1
	# 			continue

	# 		if self.pdb_benchmark_dict[pdb_id]["total_length"] > self.max_length:
	# 			c5 += 1
	# 			continue

	# 		selected_pdb_ids.append( pdb_id )
	# 	print( c1, "  ", c2, "  ", c3, "  ", c4, "  ", c5 )
	# 	return selected_pdb_ids


	# ##------------------------------------------------------------##
	# ##------------------------------------------------------------##
	# def segregate_benchmark( self, selected_pdb_ids ):
	# 	"""
	# 	Segregate all remaining complexes into:
	# 		Heteromers: >1 polymer_entity_ids all with 1 chain.
	# 		Homomers: only >=1 polymer_entity_ids with >1 chain.
	# 	"""
	# 	segregated_pdb_ids = {k: [] for k in ["hetero", "homo"]}
	# 	for pdb_id in selected_pdb_ids:

	# 		stoichiometry = self.pdb_benchmark_dict[pdb_id]["stoichiometry"]
	# 		stoichiometry = stoichiometry.split( "-" )

	# 		hetero = all( [s == "1" for s in stoichiometry] )

	# 		if hetero:
	# 			segregated_pdb_ids["hetero"].append( pdb_id )
	# 		else:
	# 			segregated_pdb_ids["homo"].append( pdb_id )
	# 	return segregated_pdb_ids


	# def filter_and_segregate( self ):
	# 	"""
	# 	Remove a PDB entry if:
	# 		1. It contains a non-polymeric entity.
	# 			entity_ids > polymer_entity_ids.
	# 		2. No. of UniProt IDs does not match the no. of polymer_entity_ids.

	# 	Segregate all remaining complexes into:
	# 		Heteromers: >1 polymer_entity_ids all with 1 chain.
	# 		Homoromers: only >=1 polymer_entity_ids with >1 chain.
	# 	"""
	# 	selected_pdb_ids = self.filter_benchmark()
	# 	print( f"Selected PDB IDs: {len( selected_pdb_ids )}" )
	# 	selected_pdb_ids = self.segregate_benchmark( selected_pdb_ids )

	# 	return selected_pdb_ids


	# def clean_benchmark_dict( self, selected_pdb_ids: List ):
	# 	"""
	# 	Remove PDB entries which were not selected.
	# 	Remove PDB entries for which UniProt IDs could not be downloaded.
	# 	"""
	# 	tmp = copy.deepcopy( self.pdb_benchmark_dict )

	# 	self.pdb_benchmark_dict = {}
	# 	for pdb_id in tmp:
	# 		if pdb_id in selected_pdb_ids["hetero"]:
	# 			self.pdb_benchmark_dict[pdb_id] = tmp[pdb_id]
	# 		elif pdb_id in selected_pdb_ids["homo"]:
	# 			self.pdb_benchmark_dict[pdb_id] = tmp[pdb_id]

	# 	print( "Remaiing entries: ", len( self.pdb_benchmark_dict ) )


	# ##------------------------------------------------------------##
	# ##------------------------------------------------------------##
	# def get_unique_uni_seq( self, selected_pdb_ids: Dict[str, List] ):
	# 	"""
	# 	Get unique UniProt IDs.
	# 	"""
	# 	unique_uni_ids = []
	# 	for category in selected_pdb_ids:
	# 		for pdb_id in selected_pdb_ids[category]:

	# 			uniprot_ids = self.pdb_benchmark_dict[pdb_id]["uniprot_ids"].split( "," )

	# 			for uni_id in uniprot_ids:
	# 				if uni_id not in unique_uni_ids:
	# 					unique_uni_ids.append( uni_id )

	# 	return unique_uni_ids


	# def dwnld_uni_seq( self, selected_pdb_ids: Dict[str, List] ):
	# 	"""
	# 	Download unique UniProt sequences for all UniProt accessions in the
	# 		selected PDB IDs (hetero/homo-mers).
	# 	"""
	# 	unique_uni_ids = self.get_unique_uni_seq( selected_pdb_ids )
	# 	total = len( unique_uni_ids )
	# 	for idx, uni_id in enumerate( unique_uni_ids ):
	# 	# with Pool( 5 ) as p:
	# 	# 	for result in tqdm.tqdm(
	# 	# 							p.imap_unordered( partial( get_uniprot_seq, max_trials = 10,
	# 	# 																wait_time = 5,
	# 	# 																return_id = True ),
	# 	# 												unique_uni_ids ),
	# 	# 							total = total ):
	# 		print( f"Downloading seq for Uni ID: {uni_id} -- {idx}/{total}... " )
	# 		uni_seq = get_uniprot_seq( uni_id,
	# 								max_trials = 10,
	# 								wait_time = 5,
	# 								return_id = False )

	# 		# uni_id, uni_seq = result
	# 		if len( uni_seq ) != 0:
	# 			self.uni_seq_dict[uni_id] = uni_seq
	# 		else:
	# 			print( f"{uni_id} --> {uni_seq}" )

	# 	write_json( self.uni_seq_dict, self.uni_seq_file )


	# def dwnld_pdb_struct( self, selected_pdb_ids: Dict[str, List] ):
	# 	"""
	# 	Download the structure as a .pdb file.
	# 	"""
	# 	for category in selected_pdb_ids:
	# 		total = len( selected_pdb_ids[category] )
	# 		for idx, pdb_id in enumerate( selected_pdb_ids[category] ):
	# 			print( f"Downloading PDB struct for: {pdb_id} -- {idx}/{total}... " )
	# 			pdb_file = os.path.join( self.pdb_struct_dir, f"{pdb_id}.pdb" )
	# 			sys_pdb_path = os.path.join( self.benchmark_dir, f"{pdb_id}/{pdb_id}.pdb" )

	# 			# If system already selected and created, do not download again.
	# 			if os.path.exists( sys_pdb_path ):
	# 				continue
	# 			# If the .pdb file in metadata dir does not exist.
	# 			if not os.path.exists( pdb_file ):
	# 				result = download_pdb( pdb_id, "pdb", pdb_file )
	# 				# If .pdb file doesn't exist, try .cif file.
	# 				if not result:
	# 					result = download_pdb( pdb_id, "cif", pdb_file )
	# 					# Throw an error if can't download a .cif also.
	# 					if not result:
	# 						print( f"Warning: Unable to download PDB: {pdb_id}" )


	# ##------------------------------------------------------------##
	# ##------------------------------------------------------------##
	# def map_pdb_to_uniprot( self, selected_pdb_ids: Dict[str, List] ):
	# 	"""
	# 	Get PDB to UniProt mapping using SIFTS.
	# 	"""
	# 	for category in selected_pdb_ids:
	# 		total = len( selected_pdb_ids[category] )
	# 		for idx, pdb_id in enumerate( selected_pdb_ids[category] ):
	# 			print( f"Obtaining SIFTS mapping for: {pdb_id} -- {idx}/{total}... " )

	# 			xml_file_path = os.path.join( self.sifts_xml_dir, f"{pdb_id}.xml" )
	# 			if not os.path.exists( xml_file_path ):
	# 				download_sifts_mapping( pdb_id,
	# 										xml_file_path,
	# 										max_trials = 5,
	# 										wait_time = 5 )
	# 			sifts_dict = parse_sifts_xml( xml_file_path )
	# 			write_json( sifts_dict,
	# 						os.path.join( self.sifts_dict_dir, f"{pdb_id}.json" ) )

	# 			pdb_uni_res = self.get_pdb_to_uni_res_map( sifts_dict )
	# 			self.sifts_pdb_to_uni[pdb_id] = pdb_uni_res


	# def get_pdb_to_uni_res_map( self, sifts_dict: Dict ) -> Dict[str, Dict[str, str]]:
	# 	"""
	# 	Create a dict with PDB positions as keys and UniProt positions 
	# 		as values for all chains.
	# 	"""
	# 	pdb_uni_res = {}
	# 	for chain in sifts_dict:
	# 		# if chain not in pdb_uni_res:
	# 		# 	pdb_uni_res[chain] = {}
	# 		pdb_pos = sifts_dict[chain]["resolved"]["PDB position"]
	# 		uni_pos = sifts_dict[chain]["resolved"]["Uniprot position"]
	# 		if len( pdb_pos ) != len( uni_pos ):
	# 			raise ValueError( "Error in PDB and UniProt mapping for" +
	# 								f" PDB {pdb_id}, chain {chain}. \n"
	# 								f"PDB residues = {len( pdb_pos )}" +
	# 								f" and UniProt residues = {len( uni_pos )}" )
	# 		pdb_uni_res[chain] = dict( zip( pdb_pos, uni_pos ) )

	# 	return pdb_uni_res


	# ##------------------------------------------------------------##
	# ##------------------------------------------------------------##
	# def create_system_dir( self, pdb_id: str ):
	# 	"""
	# 	Create directories for all benchmark systems.
	# 	"""
	# 	sys_dir = os.path.join( self.benchmark_dir, f"{pdb_id}" )
	# 	os.makedirs( sys_dir, exist_ok = True )


	# def get_entities( self, stoichiometry: List,
	# 					uniprot_ids: Dict,
	# 					uni_pos: List ) -> List[Dict]:
	# 	"""
	# 	Create all entities part of the system. Have the following:
	# 		entity_id (starts from 1).
	# 		UniProt ID.
	# 		copy_num --> no. of copies for an entity.
	# 		start, end UniProt residue positions.
	# 		Complete UniProt sequence.
	# 	"""
	# 	entities = []
	# 	for i in range( len( stoichiometry ) ):
	# 		uni_id = uniprot_ids[i]
	# 		copy_num = int( stoichiometry[i] )
	# 		# Residue positions are stored as "{start}-{end}"
	# 		start, end = list( map( int, uni_pos[i].split( "-" ) ) )

	# 		if uni_id not in self.uni_seq_dict:
	# 			raise KeyError( f"{uni_id} not present in the self.uni_seq_dict..." )
	# 		# The downstream script will select the required sequence.
	# 		seq = self.uni_seq_dict[uni_id]

	# 		entities.append(
	# 			{
	# 				"entity_id": i+1,
	# 				"uni_id": uni_id,
	# 				"copy_num": copy_num,
	# 				"start": start,
	# 				"end": end,
	# 				"sequence": seq
	# 			}
	# 		 )

	# 	return entities


	# def create_sys_dict_entry( self, sys_num: int,
	# 							sys_name: str,
	# 							entities: Dict ) -> Dict:
	# 	"""
	# 	Create a config dict containing:
	# 		"system_{index}": {
	# 				"name",
	# 				"entity": {
	# 					{},
	# 					{}
	# 				},
	# 				"data_gathering": {
	# 					"xl_restraint": {}
	# 				}
	# 		}
	# 	"""
	# 	sys_dict = {
	# 				f"System_{sys_num}": {
	# 						"name": sys_name,
	# 						"entity": entities,
	# 						"data_gathering": {
	# 							"xl_restraint": {
	# 								"xl_max_bound": self.xl_length,
	# 								"file_name": f"{sys_name}_interprotein_xls.csv",
	# 							}
	# 						}
	# 					}
	# 				}
	# 	return sys_dict


	# def create_sys_config_dict( self ):
	# 	"""
	# 	For all the benchmark entries create a config dict.
	# 	We use the PDB ID as the system name.
	# 	"""
	# 	for sys_num, sys_name in enumerate( self.pdb_benchmark_dict ):
	# 		sys_dict = {}
	# 		# Stoichiometry is stored as '-' separated values.
	# 		stoichiometry = self.pdb_benchmark_dict[sys_name]["stoichiometry"].split( "-" )
	# 		uniprot_ids = self.pdb_benchmark_dict[sys_name]["uniprot_ids"].split( "," )
	# 		uni_pos = self.pdb_benchmark_dict[sys_name]["uni_pos"].split( "," )

	# 		sys_config_file = os.path.join( self.benchmark_dir,
	# 										f"{sys_name}/sys_conf_{sys_name}.json" )

	# 		self.create_system_dir( sys_name )
	# 		entities = self.get_entities( stoichiometry, uniprot_ids, uni_pos )
	# 		sys_dict = self.create_sys_dict_entry( sys_num, sys_name, entities )

	# 		src_path = os.path.join( self.pdb_struct_dir, f"{sys_name}.pdb" )
	# 		dest_path = os.path.join( self.benchmark_dir, f"{sys_name}/" )
	# 		if os.path.exists( src_path ):
	# 			cmd = ["cp", f"{src_path}", f"{dest_path}"]
	# 			run_subprocess( cmd )

	# 		write_json( sys_dict, sys_config_file )


	# ##------------------------------------------------------------##
	# ##------------------------------------------------------------##
	# def run_jwalk( self, pdb_file: str ):
	# 	"""
	# 	Run Jwalk to obtain XLs given a .pdb file.
	
	# 	Input:
	# 	----------
	# 	pdb_file --> Path to the .pdb file.

	# 	returns:
	# 	----------
	# 	None
	# 	"""
	# 	cmd = [
	# 	f"{self.jwalk_exec}",
	# 	"-i", f"{pdb_file}",
	# 	]

	# 	run_subprocess( cmd )


	# def parse_jwalk_output( self, name: str ) -> pd.DataFrame:
	# 	"""
	# 	Jwalk writes a .txt file containing all the XLs.
	# 	It provides the following info:
	# 		Index, Model (input file name)
	# 		Atom1 --> AA-RES-CHAIN-CA
	# 		Atom2 --> AA-RES-CHAIN-CA
	# 		SASD --> Solvent accessible surface distance.
	# 		Eculidean distance --> distance between CA atoms.
	# 	Fetch all intra and inter-protein XLs for which the Euclidean distance is less than xl_length bound.

	# 	Input:
	# 	----------
	# 	name --> System name.

	# 	Returns:
	# 	----------
	# 	None
	# 	"""
	# 	file_path = glob.glob( "./Jwalk_results/*.txt" )
	# 	if len( file_path ) == 0:
	# 		raise Exception( f"Jwalk results .txt file does not exist for {name}..." )
	# 	file_path = file_path[0]
	# 	df = pd.read_csv( file_path, sep = "\s+" ) # delim_whitespace = True

	# 	# Remove XLs with SASD distances higher than the XL_length.
	# 	# 	Using SASD provides more accurate XLs.
	# 	df = df.loc[df["SASD"] <= self.xl_length]

	# 	inter = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( "-" ).str[2]]

	# 	# Extract chain ID and res no.
	# 	interprotein_xls = pd.DataFrame()
	# 	for i in [1, 2]:
	# 		interprotein_xls[f"prot{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[2]
	# 		interprotein_xls[f"res{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[1]

	# 	interprotein_xls = interprotein_xls.reset_index( drop = True )

	# 	return interprotein_xls


	# def get_uni_pos_for_xls( self, pdb_id: str, df: pd.DataFrame
	# 						) -> pd.DataFrame:
	# 	"""
	# 	Given a dataframe for inter-protein XLs, map the PDB positions
	# 		to the corresponding UniProt positions.
	# 		Columns: Protein1, Residue1, Protein1, Residue1
	# 	"""
	# 	for i in range( df.shape[0] ):
	# 		chain1 = df.loc[i, "prot1"]
	# 		res1 = int( df.loc[i, "res1"] )
	# 		chain2 = df.loc[i, "prot2"]
	# 		res2 = int( df.loc[i, "res2"] )

	# 		df.iloc[i, 1] = self.sifts_pdb_to_uni[pdb_id][chain1][res1]
	# 		df.iloc[i, 3] = self.sifts_pdb_to_uni[pdb_id][chain2][res2]
	# 	return df


	# def get_chain_entity_map( self, pdb_id: str ) -> Dict[str, int]:
	# 	"""
	# 	For a given PDB ID, create dict mapping the auth_asym_ids to
	# 		their respective entity_id.
	# 	"""
	# 	chain_entity_map = {}
	# 	entity_ids = self.pdb_benchmark_dict[pdb_id]["polymer_entity_ids"].split( "," )
	# 	auth_asym_ids = self.pdb_benchmark_dict[pdb_id]["auth_asym_ids"].split( "," )

	# 	for i in range( len( auth_asym_ids ) ):
	# 		for aa_id in auth_asym_ids[i].split( "-" ):
	# 			chain_entity_map[aa_id] = entity_ids[i]

	# 	return chain_entity_map


	# def map_chain_to_protein( self, pdb_id: str, df: pd.DataFrame
	# 							) -> pd.DataFrame:
	# 	"""
	# 	Given the inter-protein XL pairs, map the chains to the respective proteins.
	# 	Each protein is identified by an entity_id, so we replace chains with -
	# 		f"prot_{entity_id}".
	# 	"""
	# 	chain_entity_map = self.get_chain_entity_map( pdb_id )

	# 	for i in range( df.shape[0] ):
	# 		chain1 = df.iloc[i, 0]
	# 		chain2 = df.iloc[i, 2]

	# 		prot = f"prot_{chain_entity_map[chain1]}"
	# 		df.iloc[i, 0] = prot
	# 		prot = f"prot_{chain_entity_map[chain2]}"
	# 		df.iloc[i, 2] = prot
	# 	return df


	# def simulate_xl_data( self ):
	# 	"""
	# 	Simulate XLs for the system uisng Jwalk.
	# 	Save the intraprotein and interprotein XLs as csv files.
	# 	"""
	# 	print( "Simulating XL data..." )
	# 	num_xls = {}
	# 	# Remove PDB IDs for which JWalk couldn't be run or for which all residues were not mapped.
	# 	no_xls, exclude = [], []
	# 	for idx, pdb_id in enumerate( self.pdb_benchmark_dict ):
	# 		print( f"\n--> {idx} --> {pdb_id}" )
	# 		# Not all PDB positions are mapped to UniProt.
	# 		# if pdb_id in ["8g0k", "8ck8"]:
	# 		# 	continue
	# 		sys_dir = os.path.join( self.benchmark_dir, f"{pdb_id}/" )

	# 		# Move to system dir.
	# 		os.chdir( sys_dir )
	# 		if os.path.exists( f"./{pdb_id}.pdb" ):
	# 			struct_file = os.path.join( f"{pdb_id}.pdb" )
	# 		elif os.path.exists( f"./{pdb_id}.cif" ):
	# 			struct_file = os.path.join( f"{pdb_id}.cif" )
	# 		else:
	# 			raise FileNotFoundError( f"PDB/CIF file not found for entry {pdb_id}..." )

	# 		try:
	# 			# Run Jwalk.
	# 			if not os.path.exists( "./Jwalk_results/" ):
	# 				self.run_jwalk( struct_file )
	# 			else:
	# 				print( f"Jwalk_results already present in {pdb_id} dir..." )
	# 		except:
	# 			print( f"Couldn't run JWalk for {pdb_id}..." )
	# 			exclude.append( pdb_id )

	# 		# Obtain intraprotein and interprotein XLs from Jwalk output.
	# 		interprotein_xls = self.parse_jwalk_output( pdb_id )
	# 		try:
	# 			# Map PDB positions to UniProt positions for all XLs.
	# 			interprotein_xls = self.get_uni_pos_for_xls( pdb_id, interprotein_xls )
	# 			# Convert the chain IDs to proteins.
	# 			interprotein_xls = self.map_chain_to_protein( pdb_id, interprotein_xls )
	# 			print( f"Inter-XLs = {len( interprotein_xls )}" )

	# 			if len( interprotein_xls ) > 0:
	# 				# just logging the no. of XLs.
	# 				num_xls[pdb_id] = len( interprotein_xls )
	# 				# Save on disk.
	# 				interprotein_xls.to_csv( f"./{pdb_id}_interprotein_xls.csv", index = False )
	# 			else:
	# 				no_xls.append( pdb_id )
	# 		except:
	# 			print( "Not all PDB residues mapped to UniProt..." )
	# 			exclude.append( pdb_id )

	# 		os.chdir( "../../../" )

	# 	for pdb_id in exclude+no_xls:
	# 		self.pdb_benchmark_dict.pop( pdb_id )

	# 	return num_xls


	# ##------------------------------------------------------------##
	# ##------------------------------------------------------------##
	# def get_coords_for_pdb( self, pdb_file: str ) -> Dict[str, np.array]:
	# 	"""
	# 	Given a pdb file path, return a coordinates dict for all chains in the model.
	# 	"""
	# 	p = Parser( pdb_file )

	# 	# Just considering the 1st model.
	# 	model = list( p.get_models() )[0]
	# 	coords_dict = p.get_coordinates( model )
	# 	return coords_dict


	# def get_contact_map( self, coords1: np.array, coords2: np.array ):
	# 	"""
	# 	Given the coordinates, create a conntact map.
	# 	"""
	# 	distance_map = get_distance_map( coords1, coords2 )
	# 	contact_map = np.where( distance_map <= self.contact_threshold, 1, 0 )
	# 	return contact_map


	# def split_domain( self, D: str ):
	# 	"""
	# 	Split the domain D and map to int.
	# 	"""
	# 	s, e = list( map( int, D.split( "-" ) ) )
	# 	return s, e


	# def present_in_domain( self, d: str, R: int ):
	# 	"""
	# 	Check if a residue R is present in the domain D.
	# 	"""
	# 	s, e = self.split_domain( d )
	# 	if R>=s and R <e:
	# 		present = True
	# 	else:
	# 		present = False
	# 	return present


	# def find_existing_domain( self, D: List, R: int ):
	# 	"""
	# 	Find the domain which contains the residue R.
	# 	"""
	# 	domain = ""
	# 	for d in D:
	# 		if self.present_in_domain( d, R ):
	# 			domain = d

	# 	if len( domain ) == 0:
	# 		raise ValueError( f"No domain exists for {R}..." )
	# 	return domain


	# def check_domain_overlap( self, D1: str, D2: str ):
	# 	"""
	# 	Given two domain_id's, check if they oerlap.
	# 	"""
	# 	s, e = self.split_domain( D1 )
	# 	domain1 = np.arange( s, e+1, 1 )

	# 	s, e = self.split_domain( D2 )
	# 	domain2 = np.arange( s, e+1, 1 )

	# 	overlap = any( set( D1 ).intersection( set( D2 ) ) )
	# 	return overlap


	# def create_domain( self, D: List, R: int, length: int ):
	# 	"""
	# 	We represent a domain as a '-' separated string (domain_id),
	# 		comprising the start and end residue positions of the domain.
	# 	Given a residue R, we consider a domain as the set of 10-residues
	# 		with R at the center.
	# 	Assumption:
	# 		A domain shares no overlap with any other existing domain.
	# 		R represents the index of the residue not 
	# 			the residue position itself.
	# 	"""
	# 	# Moin index can be 0.
	# 	s = max( 0, R-5 )
	# 	e = min( length, R+5 )
	# 	new_D = f"{s}-{e}"
	# 	for d in D:
	# 		# Do not create a new domain if it overalps with any of the existing domains.
	# 		if self.check_domain_overlap( d, new_D ):
	# 			new_D = ""
	# 			break
	# 	return new_D


	# def contact_map_to_domains( self, contact_map: np.array ):
	# 	"""
	# 	Given a contact map, get domains spanning 10-residues.
	# 	Identify a contact and cosnider flanking region of 5 residues
	# 		on either side as a domain.
	# 	"""
	# 	p1_len, p2_len = contact_map.shape
	# 	p1_idx, p2_idx = np.where( contact_map == 1 )
	# 	# A list to keep track of all domains created for prot1/2.
	# 	D1, D2 = [], []
	# 	# A dict to store all interacting domains of prot1 with prot2.
	# 	domain_dict = {}

	# 	# Here we are dealing with indices. Will convert to PDB positions later.
	# 	for r1, r2 in zip( p1_idx, p2_idx ):
	# 		# Create a new domain.
	# 		d1 = self.create_domain( D1, r1, p1_len )
	# 		d2 = self.create_domain( D2, r2, p2_len )

	# 		# Find existing domains if couldn't create a new domain.
	# 		if len( d1 ) == 0:
	# 			D1.append( d1 )
	# 		else:
	# 			d1 = self.find_existing_domain( D1, r1 )

	# 		if len( D2 ) == 0:
	# 			D2.append( d2 )
	# 		else:
	# 			d2 = self.find_existing_domain( D2, r2 )

	# 		# Ignore if a residue is not part of any new/existing domain.
	# 		# 	Can happen for overlapping domains.
	# 		if len( d1 ) == 0 or len( d2 ) == 0:
	# 			continue
	# 		else:
	# 			if d1 in domain_dict:
	# 				if not d2 in domain_dict[d1]:
	# 					domain_dict[d1].append( d2 )
	# 			else:
	# 				domain_dict[d1] = [d2]
	# 	return domain_dict


	# def simulate_domain_interaction_data( self ):
	# 	"""
	# 	Simulate domain-level inter-protein interactions for the benchmark.
	# 	this is akin to data obtained from Y2H, co-IP.

	# 	For each PDB in the benchmark:
	# 		Get inter-chain contact maps.
	# 			Select the 1st model only.
	# 			Select chains.
	# 		Coarse-grain the contact maps.
	# 		Select interacting coarse-grained regions (domains).
	# 		Save as a .csv file.
	# 	"""
	# 	print( "Simulating domain-level data..." )
	# 	num_domains = {}
	# 	for idx, pdb_id in enumerate( self.pdb_benchmark_dict ):
	# 		print( f"\n--> {idx} --> {pdb_id}" )




	# def write_benchmark_to_csv( self, num_xls ):
	# 	"""
	# 	Create a .csv file for all the PDB IDs in the benchmark.
	# 	Ignore the ones with 0 interprotein-XLs.
	# 	"""
	# 	dum = list( self.pdb_benchmark_dict.keys() )[0]
	# 	keys = ["pdb_id"] + list( self.pdb_benchmark_dict[dum].keys() )
	# 	flat_dict = {k:[] for k in keys}
	# 	flat_dict["Interprotein-XLs"] = []

	# 	for pdb_id in self.pdb_benchmark_dict:
	# 		flat_dict["pdb_id"].append( pdb_id )
	# 		for k, v in self.pdb_benchmark_dict[pdb_id].items():
	# 			flat_dict[k].append( v )
	# 		flat_dict["Interprotein-XLs"].append( num_xls[pdb_id] )

	# 	df = pd.DataFrame( flat_dict )
	# 	df.to_csv( self.output_benchmark_csv, index = False )


if __name__ == "__main__":
	Metadata().forward()
