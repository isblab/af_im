"""
This module contains classes for obtaining:
	Entry details from PDB REST API
	PDB-Uniprot mapping using SIFTS
	UniProt sequences
	PDB structures
"""
from typing import List, Tuple, Dict
import os, time, re, copy
from collections import defaultdict
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import tqdm

from utils.utils import ( open_file_handler,
							read_json, write_json,
							ranges )
from utils.api_utils import ( PdbRestApi,
							download_sifts_mapping,
							parse_sifts_xml,
							get_uniprot_seq,
							download_pdb )
from utils.pdb_utils import ( MmcifDictParser )



class PdbData():
	"""
	Given a list of PDB IDs, obtain relevant data for each.
	"""
	def __init__( self, pdb_ids_list: List[str],
					pdb_api_dir: str,
					pdb_data_dict_file: str,
					cores: int,
					max_trials: int,
					wait_time: int ):
		self.pdb_ids_list = pdb_ids_list
		self.pdb_api_dir = pdb_api_dir
		self.pdb_data_dict_file = pdb_data_dict_file

		self.cores = cores
		self.max_trials = max_trials
		self.wait_time = wait_time
		# If True, allows entities with no Uniprot IDs.
		self.ignore_no_uni_id = False

		self.pdb_data_dict = {}
		self.pdb_logs = {}



	def forward( self ):
		"""
		"""
		t_start = time.time()
		self.initialize_logs_dict()
		self.pdb_logs["total_pdb_ids"] = len( self.pdb_ids_list )

		self.get_entry_info_in_parallel()
		# write_json( self.pdb_data_dict, self.pdb_data_dict_file )

		t_end = time.time()
		time_taken = t_end - t_start
		self.pdb_logs["remaining_pdb_ids"] = len( self.pdb_data_dict )
		self.pdb_logs["time_taken"] = time_taken


	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.pdb_logs = {
		k:[[], 0] for k in ["no_pdb_data", "monomer",
						"has_non_polymer_entity",
						"too_many_uni_ids"]
		}


	# def get_empty_pdb_dict( self ):
	# 	"""
	# 	Create an empty dict with the required fields.
	# 	"""
	# 	empty_dict = {k: [] for k in [
	# 					"entity_ids", "polymer_entity_ids",
	# 					"uniprot_ids", "stoichiometry",
	# 					"uni_pos", "pdb_pos"]}
	# 	return empty_dict



	def get_entry_entity_files( self, entry_id: str
								) -> Tuple[str, str]:
		"""
		Return the file paths for the entry and entity dict.
		"""
		entry_file = os.path.join( self.pdb_api_dir, f"entry_dict_{entry_id}.json" )
		entity_file = os.path.join( self.pdb_api_dir, f"entity_dict_{entry_id}.json" )
		return entry_file, entity_file



	def save_entry_entity_dict( self, rest_api: PdbRestApi,
								entry_file: str,
								entity_file: str ):
		"""
		Save the entry and entity dict on disk.
		Assuming the entity dict contains info. for all entities to be saved.
		"""
		if not os.path.exists( entry_file ):
			write_json( rest_api.entry_data, entry_file )
		if not os.path.exists( entity_file ):
			write_json( rest_api.entity_data, entity_file )



	def instantiate_pdb_rest_api( self, entry_id: str,
									entry_file: str,
									entity_file: str ) -> PdbRestApi:

		"""
		Instantiate the PDB REST API object.
		"""
		rest_api = PdbRestApi( entry_id = entry_id,
								entry_file = entry_file,
								entity_file = entity_file,
								max_trials = self.max_trials,
								wait_time = self.wait_time )
		return rest_api



	def get_entry_info( self, all_entities: List[int],
						polymer_entity_ids: List[int],
						rest_api: PdbRestApi
						) -> Dict[str, str]:
		"""
		Given a list of polymer entity IDs, obtain the following info:
			UniProt IDs
			Asym ID and Auth Asym ID
			UniProt and PDB residue positions
			Total length
		"""
		stoichiometry = []
		uniprot_ids = []
		auth_asym_ids = []
		uni_pos, pdb_pos = [], []
		total_length = 0

		ignore = False
		for entity_id in polymer_entity_ids:
			rest_api.retrieve_polymer_entity_data( entity_id )
			uni_id = rest_api.get_uniprot_ids_for_entity( entity_id )
			if len( uni_id ) <= 1 and self.ignore_no_uni_id:
				uniprot_ids.extend( uni_id )
				asym_ids = rest_api.get_asym_ids_for_entity( entity_id )
				auth_asym_ids.append( "-".join(
					rest_api.get_auth_asym_ids_for_entity( entity_id ) )
				)
				pos, length = rest_api.get_poly_entity_align( entity_id )
				uni_pos.append( pos[0] )
				pdb_pos.append( pos[1] )
				# Account for all copies of an entity.
				total_length += length*len( asym_ids )

				stoichiometry.append( f"{len( asym_ids )}" )
			else:
				ignore = True
				break

		if ignore:
			entry_dict = None
		else:
			entry_dict = {
			"entity_ids": ",".join( all_entities ),
			"polymer_entity_ids": ",".join( polymer_entity_ids ),
			"uniprot_ids": ",".join( uniprot_ids ),
			"auth_asym_ids": ",".join( auth_asym_ids ),
			"stoichiometry": stoichiometry,
			"uni_pos": ",".join( uni_pos ),
			"pdb_pos": ",".join( pdb_pos ),
			"total_length": total_length,
			}
		return entry_dict



	def entry_from_pdb_rest_api( self, entry_id: str
								) -> Tuple[str, Dict[str, str], Dict[str, str]]:
		"""
		Given a PDB ID, fetch the required info. from the PDB REST API
			(See get_entry_info()).
		"""
		logs = {}
		entry_file, entity_file = self.get_entry_entity_files( entry_id )

		# Instantiate the PDB REST API object.
		rest_api = self.instantiate_pdb_rest_api( entry_id,
													entry_file,
													entity_file )

		# Could not retrieve PDB entry data.
		if rest_api.entry_data is None:
			logs["no_pdb_data"] = [entry_id]
			entry_dict = None

		else:
			# Get all entity IDs in the.
			all_entities = rest_api.get_all_entities()
			# Get all polymer entity IDs in the.
			polymer_entity_ids = rest_api.get_polymer_entities()

			# Monomer entry.
			if len( all_entities ) == 1:
				logs["monomer"] = [entry_id]
			# # PDB contains non-polymer entities.
			# elif len( all_entities ) > len( polymer_entity_ids ):
			# 	logs["has_non_polymer_entity"] = [entry_id]
			# 	entry_dict = None
			# else:
			else:
				entry_dict = self.get_entry_info( all_entities,
												polymer_entity_ids,
												rest_api )

				if entry_dict is None:
					logs["too_many_uni_ids"] = [entry_id]
				else:
					self.save_entry_entity_dict( rest_api,
												entry_file,
												entity_file )
		return entry_id, entry_dict, logs



	def get_entry_info_in_parallel( self ):
		"""
		For all the benchmark PDB IDs, get the required info.
		(See self.entry_from_pdb_rest_api doc-str)
		"""
		with Pool( self.cores ) as p:
			for result in tqdm.tqdm(
					p.imap_unordered( self.entry_from_pdb_rest_api,
										self.pdb_ids_list ),
					total = len( self.pdb_ids_list )
					):
				entry_id, entry_dict, logs = result

				if entry_dict is None:
					for k in logs:
						self.pdb_logs[k][0].extend( logs[k] )
						self.pdb_logs[k][1] += len( logs[k] )
				else:
					self.pdb_data_dict[entry_id] = entry_dict


###########################################################################
###########################################################################
class SiftsMapping():
	"""
	Given a list of PDB IDs, obtain the PDB-UniProt mapping using SIFTS.
	"""
	def __init__( self, pdb_ids_list: List[str],
					sifts_xml_dir: str,
					sifts_dict_dir: str,
					sifts_dict_file: str,
					max_sys_length: int,
					frac_coverage: float,
					cores: int,
					max_trials: int,
					wait_time: int ):
		self.pdb_ids_list = pdb_ids_list
		self.sifts_xml_dir = sifts_xml_dir
		self.sifts_dict_dir = sifts_dict_dir
		self.sifts_dict_file = sifts_dict_file

		self.max_sys_length = max_sys_length
		self.frac_coverage = frac_coverage
		self.cores = cores
		self.max_trials = max_trials
		self.wait_time = wait_time

		self.sifts_data_dict = {}
		self.sifts_logs = {}


	def forward( self ):
		"""
		"""
		t_start = time.time()
		self.initialize_logs_dict()
		self.sifts_logs["total_pdb_ids"] = len( self.pdb_ids_list )

		self.get_sifts_mapping_in_parallel()
		# write_json( self.sifts_data_dict, self.sifts_dict_file )

		t_end = time.time()
		time_taken = t_end - t_start

		self.sifts_logs["remaining_pdb_ids"] = len( self.sifts_data_dict )
		self.sifts_logs["time_taken"] = time_taken


	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.sifts_logs = {
		k:[[], 0] for k in ["not_mapped", "icode_present",
						"mapped_length_mismatch",
						"ambiguous_uni_id",
						"low_coverage",
						"discontiguous_residues_in_chain",
						"exceed_max_length"]
		}



	def extract_digits( self, res_num: str ) -> str:
		"""
		Remove alphabets from an alphanum str.
		e.g. "222A" -> "222"
		Negative no. are also accounted.
		"""
		match = re.match( r"(-?\d+)", res_num )
		res_num = match.group( 1 ) if match else None
		return res_num



	def remove_insertion_code( self,
								mapping_dict: Dict[str, Dict]
								) -> bool:
		"""
		Remove insertion code from the PDB residue number.
		Make changes inplace.
		"""
		for chain_id in mapping_dict:
			pdb_res_ = mapping_dict[chain_id]["resolved"]["PDB position"]

			pdb_res = [self.extract_digits( res ) for res in pdb_res_]

			mapping_dict[chain_id]["resolved"]["PDB position"] = pdb_res

			if pdb_res != pdb_res_:
				icode_present = True
			else:
				icode_present = False

		return icode_present



	def convert_to_int( self, mapping_dict: Dict[str, Dict] ):
		"""
		Convert the PDB and UniProt residue positions to int,
			for the resolved residues.
		"""
		for k in ["PDB position", "Uniprot position"]:
			for chain_id in mapping_dict:
				# try:
				mapping_dict[chain_id]["resolved"][k] = list( 
					map( int, mapping_dict[chain_id]["resolved"][k] )
				 )
				# except:
				# 	print( entry_id, "  ", chain_id )
				# 	print( mapping_dict[chain_id]["resolved"][k] )




	def pdb_uni_length_mismatch( self, mapping_dict: Dict[str, Dict] ) -> bool:
		"""
		Check if for any chain in PDB the no. of resolved PDB and
			Uniprot residues is different.
		"""
		passed = []
		for chain_id in mapping_dict:
			pdb_res = mapping_dict[chain_id]["resolved"]["PDB position"]
			uni_res = mapping_dict[chain_id]["resolved"]["Uniprot position"]

			passed.append( len( pdb_res ) == len( uni_res ) )
		return all( passed )



	def mapped_to_multiple_uni_ids( self, mapping_dict: Dict[str, Dict] ) -> bool:
		"""
		Check if a chain is mapped to multiple Uni IDs.
		"""
		passed = []
		for chain_id in mapping_dict:
			uni_ids = mapping_dict[chain_id]["resolved"]["Uniprot ID"]

			passed.append( len( uni_ids ) == 1 )
		return all( passed )


	def compute_mapped_seq_coverage( self, mapping_dict: Dict[str, Dict] ) -> float:
		"""
		Calculate the seq coverage as the fraction of residues present in the mapping
			vs the expected no. of residues in sequence.
		"""
		expected_length = 0
		res_present = 0
		for chain_id in mapping_dict:
			uni_pos = mapping_dict[chain_id]["resolved"]["Uniprot position"]
			start, end = uni_pos[0], uni_pos[-1]
			expected_length += ( end - start + 1 )

			res_present += len( uni_pos )
		coverage = res_present/expected_length

		return coverage


	def contiguous_residues_in_pdb( self, mapping_dict: Dict[str, Dict] ) -> bool:
		"""
		Check if for any chain in PDB a contiguous set
			of residues is not present.
		"""
		passed = []
		for chain_id in mapping_dict:
			uni_res = mapping_dict[chain_id]["resolved"]["Uniprot position"]
			residue_ranges = ranges( uni_res )

			passed.append( len( residue_ranges ) == 1 )
		return all( passed )


	def get_total_sys_length( self, mapping_dict: Dict[str, Dict] ) -> int:
		"""
		Get the total length for the system across all chains.
		"""
		total_length = 0
		for chain_id in mapping_dict:
			chain_len = len( mapping_dict[chain_id]["resolved"]["Uniprot position"] )
			total_length += chain_len
		return total_length



	def get_pdb_to_uni_res_map( self, mapping_dict: Dict[str, Dict]
								) -> Dict[str, Dict[int, int]]:
		"""
		Create a dict with PDB positions as keys and UniProt positions 
			as values for all chains.
		"""
		pdb_uni_map = {}
		for chain in mapping_dict:
			# if chain not in pdb_uni_res:
			# 	pdb_uni_res[chain] = {}
			uni_id = mapping_dict[chain]["resolved"]["Uniprot ID"][0]
			pdb_pos = mapping_dict[chain]["resolved"]["PDB position"]
			uni_pos = mapping_dict[chain]["resolved"]["Uniprot position"]
			pdb_uni_map[chain] = {
			"uni_id": uni_id,
			"pdb_to_uni": dict( zip( pdb_pos, uni_pos ) ),
			"uni_to_pdb": dict( zip( uni_pos, pdb_pos ) )
			}

		return pdb_uni_map


	def get_mapping_dict( self, entry_id ) -> Tuple[str, Dict]:
		"""
		Get the SIFTS mapping_dict for the given entry_id.		
		"""
		xml_file_path = os.path.join( self.sifts_xml_dir,
										f"{entry_id}.xml" )
		# Don't download, if XML file already present.
		if os.path.exists( xml_file_path ):
			success = True
		else:
			success = download_sifts_mapping( entry_id,
											xml_file_path )
		if success:
			sifts_dict_path = os.path.join( self.sifts_dict_dir,
											f"{entry_id}.json" )
			if os.path.exists( sifts_dict_path ):
				mapping_dict = read_json( sifts_dict_path )
			else:
				mapping_dict = parse_sifts_xml( xml_file_path )
				write_json( mapping_dict,
							sifts_dict_path )
		else:
			mapping_dict = {}
		return mapping_dict



	def get_sifts_mapping_for_entry( self, entry_id: str
								 ) -> Tuple[str, Dict]:
		"""
		Obtain the SIFTS mapping SIFTS maping for the given entry_id.
		Filter out cases where,
			No SIFTS mapping available.
		"""
		logs = {}
		total_length = 0

		mapping_dict = self.get_mapping_dict( entry_id )
		if mapping_dict == {}:
			logs["not_mapped"] = [entry_id]
			sifts_dict = None
		else:
			# Make changes inplace.
			icode_present = self.remove_insertion_code( mapping_dict )
			self.convert_to_int( mapping_dict )

			if icode_present:
				logs["icode_present"] = [entry_id]

			if not self.pdb_uni_length_mismatch( mapping_dict ):
				sifts_dict = None
				logs["mapped_length_mismatch"] = [entry_id]
			elif self.mapped_to_multiple_uni_ids( mapping_dict ):
				sifts_dict = None
				logs["ambiguous_uni_id"] = [entry_id]
			# elif not self.contiguous_residues_in_pdb( mapping_dict ):
			# 	sifts_dict = None
			# 	logs["discontiguous_residues_in_chain"] = [entry_id]
			elif self.compute_mapped_seq_coverage( mapping_dict ) <= self.frac_coverage:
				sifts_dict = None
				logs["low_coverage"] = [entry_id]
			else:
				total_length = self.get_total_sys_length( mapping_dict )
				if total_length > self.max_sys_length:
					sifts_dict = None
					logs["exceed_max_length"] = [entry_id]
				else:
					sifts_dict = self.get_pdb_to_uni_res_map( mapping_dict )

		return entry_id, total_length, sifts_dict, logs



	def get_sifts_mapping_in_parallel( self ):
		"""
		Parallelize downloading SIFTS mapping for all PDB IDs.
		"""
		with Pool( self.cores ) as p:
			for result in tqdm.tqdm( 
				p.imap_unordered( self.get_sifts_mapping_for_entry, self.pdb_ids_list ),
				total = len( self.pdb_ids_list ) ):
				entry_id, total_length, sifts_dict, logs = result

				if sifts_dict is None:
					for k in logs:
						self.sifts_logs[k][0].extend( logs[k] )
						self.sifts_logs[k][1] += len( logs[k] )
				else:
					self.sifts_data_dict[entry_id] = {"mapping": sifts_dict,
														"total_length": total_length}


###########################################################################
###########################################################################
class DownloadUniprotSequences():
	"""
	Given a list of UniProt IDs, obtain the sequences for all.
	"""
	def __init__ (self, uni_ids_list: List[str],
					uni_seq_dict_file: str,
					cores: int,
					max_trials: int,
					wait_time: int ):
		self.uni_ids_list = uni_ids_list
		self.uni_seq_dict_file = uni_seq_dict_file

		self.cores = cores
		self.max_trials = max_trials
		self.wait_time = wait_time

		self.uni_seq_dict = {}
		self.uni_logs = {}


	def forward( self ):
		"""
		"""
		t_start = time.time()
		self.initialize_logs_dict()

		self.uni_logs["total_uni_ids"] = len( self.uni_ids_list )

		self.deduplicate_uni_ids()
		self.uni_logs["unique_uni_ids"] = len( self.uni_ids_list )

		self.get_uni_seq_in_parallel()
		# write_json( self.uni_seq_dict, self.uni_seq_dict_file )

		t_end = time.time()
		time_taken = t_end - t_start

		self.uni_logs["remaining_uni_ids"] = len( self.uni_seq_dict )
		self.uni_logs["time_taken"] = time_taken


	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.uni_logs = {
		k:[[], 0] for k in ["uni_seq_not_found"]
		}


	def deduplicate_uni_ids( self ):
		"""
		Deduplicate the list of UniProt Ids provided.
		"""
		unique = set( self.uni_ids_list )
		self.uni_ids_list = sorted( list( unique ) )



	def get_uni_seq( self, uni_id: str
					) -> Tuple[Dict[str, str], Dict[str, str]]:
		"""
		Obtain the sequence for the given uni ID.
		"""
		logs = {}
		uni_seq = get_uniprot_seq( uni_id,
								max_trials = self.max_trials,
								wait_time = self.wait_time,
								return_id = False )

		# uni_id, uni_seq = result
		if len( uni_seq ) == 0:
			logs["uni_seq_not_found"] = [uni_id]
			seq_dict = None
		else:
			seq_dict = {uni_id: uni_seq}
		return seq_dict, logs



	def get_uni_seq_in_parallel( self ):
		"""
		Parallelize downlaoding Uniprot sequences.
		"""
		with Pool( self.cores ) as p:
			for result in tqdm.tqdm( 
				p.imap_unordered( self.get_uni_seq, self.uni_ids_list ),
				total = len( self.uni_ids_list ) ):
				seq_dict, logs = result

				if seq_dict is None:
					for k in logs:
						self.uni_logs[k][0].extend( logs[k] )
						self.uni_logs[k][1] += len( logs[k] )
				else:
					self.uni_seq_dict.update( seq_dict )


###########################################################################
###########################################################################
class DownloadPdbStructure():
	"""
	Given a list of PDB IDs, download structures
		from PDB in the specified format.
	"""
	def __init__ (self, pdb_ids_list: List[str],
					pdb_struct_dir: str,
					struct_format: str,
					cores: int,
					max_trials: int,
					wait_time: int ):
		self.pdb_ids_list = pdb_ids_list
		self.pdb_struct_dir = pdb_struct_dir
		self.struct_format = struct_format

		self.cores = cores
		self.max_trials = max_trials
		self.wait_time = wait_time

		self.downloaded_struct = []
		self.struct_logs = {}


	def forward( self ):
		"""
		"""
		t_start = time.time()
		if self.struct_format not in ["pdb", "cif", "both"]:
			raise ValueError( "Incorrect structure file format: " +
								f"{self.struct_format} specified..." )
		self.initialize_logs_dict()

		self.struct_logs["total_pdb_ids"] = len( self.pdb_ids_list )
		self.dwnld_struct_in_parallel()
		# w = open_file_handler( self.downloaded_struct_file, "w" )
		# w.writelines( ",".join( self.downloaded_struct ) )
		# w.close()

		t_end = time.time()
		time_taken = t_end - t_start
		self.struct_logs["remaining_pdb_ids"] = len( self.downloaded_struct )
		self.struct_logs["time_taken"] = time_taken



	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.struct_logs = {
		k:[[], 0] for k in ["not_downloaded"]
		}


	def dwmld_struct_for_entry_id( self, entry_id: str
									) -> [Dict[str, str]]:
		"""
		Download the structure for the given entry_id (PDB ID)
			in the specified file format.
		"""
		extension = ["pdb", "cif"] if self.struct_format == "both" else [self.struct_format]
		success = []
		for ext in extension:
			struct_file_path = os.path.join( self.pdb_struct_dir,
											f"{entry_id}.{ext}" )

			if os.path.exists( struct_file_path ):
				success.append( True )
			else:
				result = download_pdb( entry_id,
										ext,
										struct_file_path )
				if not result:
					success.append( False )
				else:
					success.append( True )

		return entry_id, all( success )



	def dwnld_struct_in_parallel( self ):
		"""
		Parallelize downlaoding structures for PDB IDs.
		"""
		with Pool( self.cores ) as p:
			for result in tqdm.tqdm( 
				p.imap_unordered( self.dwmld_struct_for_entry_id,
									self.pdb_ids_list ),
				total = len( self.pdb_ids_list ) ):
				entry_id, success = result

				if success:
					self.downloaded_struct.append( entry_id )
				else:
					self.struct_logs["not_downloaded"][0].append( entry_id )
					self.struct_logs["not_downloaded"][1] += 1



###########################################################################
###########################################################################
class SeqResDict():
	"""
	Parse a CIF file to extract the following:
		Sequence as per the SEQRES
		Entity ID
		Asym, Auth asym ID
		SEQRES and PDB residue numbering
	"""
	def __init__ (self, pdb_ids_list: List[str],
					pdb_struct_dir: str,
					cores: int,
					max_sys_length: int,
					frac_coverage: float ):
		self.pdb_ids_list = pdb_ids_list
		self.pdb_struct_dir = pdb_struct_dir

		self.cores = cores
		self.max_sys_length = max_sys_length
		self.frac_coverage = frac_coverage

		self.seqres_dict = {}
		self.cif_logs = {}


	def forward( self ):
		"""
		"""
		t_start = time.time()
		self.initialize_logs_dict()

		self.cif_logs["total_pdb_ids"] = len( self.pdb_ids_list )
		self.get_cif_dict_in_parallel()

		t_end = time.time()
		time_taken = t_end - t_start
		self.cif_logs["remaining_pdb_ids"] = len( self.seqres_dict )
		self.cif_logs["time_taken"] = time_taken



	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.cif_logs = {
		k:[[], 0] for k in ["no_protein_entity",
							"monomer",
							"exceed_max_length",
							"low_coverage",
							"length_mismatch"]
		}


	def get_resolved_mask( self, auth_seq_num: np.array ) -> np.array:
		"""
		Create a binary mask for resolved/missing residues.
		"""
		mask = np.where( auth_seq_num == "?", 0, 1 )
		return mask


	def get_valid_res_mask( self, pdb_seq_num: List[str]
					) -> List[int]:
		"""
		Create a binary mask for residues not part of the protein.
		PDB struct can also include residues artificial/engineered
			residues, expression-tag for example.
		All residues part of a protein are numbered with
			non-negative integers.
		"""
		tmp = copy.copy( pdb_seq_num ).astype( int )
		mask = np.where( tmp < 1, 0, 1 )
		return mask


	def remove_terminal_missing( self, mask: np.array
		) -> Tuple[int, int]:
		"""
		Given the binary resolved_mask,
		Ignore missing residues at the termini.
		Get the start and end indices for the remaining residues.
		"""
		idx = np.where( mask == 1 )
		start = idx[0][0]
		end = idx[0][-1]

		return start, end, idx


	def build_hierarchy( self, entry_id, prot_entity_ids: List[str],
						seqres_dict: Dict[str, List[str]]
		) -> Tuple[str, Dict[int, Dict], float]:
		"""
		Arrange the the seqres info hierarchically:
			entity_id
				chain_id
					sequence
					residue no.
		Note: Some .cif files contain discontinous PDB
			numbering (8g0q_B, 8g0q_D) in both pdb_seq_num and auth_seq_num.
			Hence, also saving the seq_id which is always continous.
		"""
		coverage = []
		hier = {}
		length_mismatch = False
		total_length = 0
		
		for entity_id in prot_entity_ids:
			hier[entity_id] = {}

			entity_index = np.where( seqres_dict["entity_id"] == entity_id )
			# Chain IDs for an entity.
			chain_ids = np.unique( seqres_dict["pdb_strand_id"][entity_index] )
			for chain_id in chain_ids:
				hier[entity_id][chain_id] = {}
				chain_index = np.where( seqres_dict["pdb_strand_id"] == chain_id )

				seq_id = seqres_dict["seq_id"][chain_index]
				auth_seq_num = seqres_dict["auth_seq_num"][chain_index]
				pdb_seq_num = seqres_dict["pdb_seq_num"][chain_index]
				seq = seqres_dict["mon_id"][chain_index]

				resolved_mask = self.get_resolved_mask( auth_seq_num )
				valid_res_mask = self.get_valid_res_mask( pdb_seq_num )
				mask = resolved_mask*valid_res_mask

				start_idx, end_idx, resolved_idx = self.remove_terminal_missing( mask )
				start_pdb_pos = int( pdb_seq_num[start_idx] )
				end_pdb_pos = int( pdb_seq_num[end_idx] )

				resolved_len = np.count_nonzero( mask[start_idx:end_idx+1] )
				chain_len = len( mask[start_idx:end_idx+1] )
				total_length += chain_len

				coverage.append( resolved_len/chain_len )

				seq = "".join( seq[start_idx:end_idx+1] )

				# Select all resolved+missing residues in sequece.
				# 	Missing residue in seq will create a mutant seq.
				# Keep only the resolved residue indices.
				if len( seq ) != ( end_idx - start_idx + 1 ):
					raise ValueError( f"Sequence length ({len( seq )}) " +
									f"does not match the residue numbering ({start_idx}-{end_idx}..." )
				start_seq_id  = int( seq_id[start_idx] )
				end_seq_id  = int( seq_id[end_idx] )
				if ( end_seq_id-start_seq_id+1 ) != len( seq ):
					length_mismatch = True
					hier[entity_id][chain_id] = None
				else:
					hier[entity_id][chain_id] = {
						"seq": seq,
						"start_pos": start_pdb_pos,
						"end_pos": end_pdb_pos,
						"res_num": pdb_seq_num[resolved_idx].astype( int ),
						"start_seq_id": start_seq_id,
						"end_seq_id": end_seq_id,
						"seq_id": seq_id[resolved_idx].astype( int )
					}

		return hier, coverage, total_length, length_mismatch


	def get_cif_dict_for_entry( self, entry_id: str ):
		"""
		Obtain the seq and residue no. for all
			entity and chain IDs in a given entry_id.
		"""
		logs = {}
		file = os.path.join( self.pdb_struct_dir, f"{entry_id}.cif" )
		if not os.path.exists( file ):
			raise FileNotFoundError( f"{file} not found..." )

		obj = MmcifDictParser( file )
		prot_entity_ids, _ = obj.get_protein_entity_ids()
		seqres_dict = copy.deepcopy(
			obj.get_protein_entity_details()
			)

		del obj

		if len( prot_entity_ids ) == 0:
			logs["no_protein_entity"] = entry_id
			cif_dict = None
			coverage = []
		elif len( prot_entity_ids ) == 1:
			logs["monomer"] = entry_id
			cif_dict = None
			coverage = []
		else:
			( cif_dict,
				coverage,
				total_length,
				length_mismatch ) = self.build_hierarchy( entry_id,
												prot_entity_ids, seqres_dict )
		if total_length > self.max_sys_length:
			logs["exceed_max_length"] = entry_id
			cif_dict = None
			coverage = []
		if length_mismatch:
			logs["length_mismatch"] = entry_id
			cif_dict = None
			coverage = []
		return entry_id, cif_dict, coverage, logs



	def get_cif_dict_in_parallel( self ):
		"""
		Paralelize extracting the seq and res num from a .cif file.
		"""
		with Pool( self.cores ) as p:
			for result in tqdm.tqdm( 
				p.imap_unordered( self.get_cif_dict_for_entry,
									self.pdb_ids_list ),
				total = len( self.pdb_ids_list ) ):
				entry_id, cif_dict, coverage, logs = result

				if cif_dict is None:
					for k in logs:
						self.cif_logs[k][0].append( logs[k] )
						self.cif_logs[k][1] += 1
				else:
					if any( [c < self.frac_coverage for c in coverage]):
						self.cif_logs["low_coverage"][0].append( entry_id )
						self.cif_logs["low_coverage"][1] += 1
					else:
						self.seqres_dict[entry_id] = cif_dict
