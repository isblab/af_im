"""
This script contains functions for handling API requests, downloading files.
"""
import os
import time
import xml.etree.ElementTree as ET
from io import StringIO
import warnings
import requests
from typing import List, Tuple, Dict, Optional
from Bio import SeqIO
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from utils.utils import ( read_json, write_to_file, run_subprocess )

warnings.filterwarnings("ignore")

####################################################################################
####----------------------------------------------------------------------------####
def aa_3_to_1( aa ):
	"""
	Converts 3-letter amino acid names to symbols

	Input:
	----------
	aa --> 3-letter code for amino acid.

	Returns:
	----------
	1-letter code for amino acid.
	"""
	if aa in ["ALA", "Ala", "ala"]:
		symbol = "A"
	elif aa in ["ARG", "Arg", "arg"]:
		symbol = "R"
	elif aa in ["ASN", "Asn", "asn"]:
		symbol = "N"
	elif aa in ["ASP", "Asp", "asp"]:
		symbol = "D"
	elif aa in ["CYS", "Cys", "cys"]:
		symbol = "C"
	elif aa in ["GLN", "Gln", "gln"]:
		symbol = "Q"
	elif aa in ["GLU", "Glu", "glu"]:
		symbol = "E"
	elif aa in ["GLY", "Gly", "gly"]:
		symbol = "G"
	elif aa in ["HIS", "His", "his"]:
		symbol = "H"
	elif aa in ["ILE", "Ile", "ile"]:
		symbol = "I"
	elif aa in ["LEU", "Leu", "leu"]:
		symbol = "L"
	elif aa in ["LYS", "Lys", "lys"]:
		symbol = "K"
	elif aa in ["MET", "Met", "met"]:
		symbol = "M"
	elif aa in ["PHE", "Phe", "phe"]:
		symbol = "F"
	elif aa in ["PRO", "Pro", "pro"]:
		symbol = "P"
	elif aa in ["SER", "Ser", "ser"]:
		symbol = "S"
	elif aa in ["THR", "Thr", "thr"]:
		symbol = "T"
	elif aa in ["TRP", "Trp", "trp"]:
		symbol = "W"
	elif aa in ["TYR", "Tyr", "tyr"]:
		symbol = "Y"
	elif aa in ["VAL", "Val", "val"]:
		symbol = "V"
	else:
		symbol = "X"

	return symbol


####################################################################################
####----------------------------------------------------------------------------####
def send_request( url, _format = "json", max_trials = 10, wait_time = 5 ):
	"""
	Send a request to the server to fetch the data.
		For Httpresponse 404 returns "not_found".
		For Httpresponse 400 returns "bad_request".
	
	Input:
	----------
	url --> URL of the server from where to fetch the data.
	_format --> output format for the server reponse (json or text).
				If None, the response is returned.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
		This is a variant of the Exponential backoff algorithm.


	Returns:
	----------
	Currently, will return either of:
		_format = None: response object.
		_format = json: return a JSON dict.
		_format = text: return the response in text format.
	"""
	for trial in range( 0, max_trials ):
		try:
			response = requests.get( url )

			# Resource not found.
			if response.status_code == 404:
				# Try till half the no. of max_trials.
				# 	Don't want it stuck for too long if the URL doesn't actually exist.
				if trial > ( max_trials/2 ):
					send_response = "not_found"
				time.sleep( wait_time )

			# Bad request.
			elif response.status_code == 400:
				if trial == ( max_trials - 1 ):
					send_response = "bad_request"
				time.sleep( wait_time )

			elif response.status_code == 200:

				if _format is None:
					send_response = response
				elif _format == "json":
					send_response = response.json()
				else:
					send_response = response.text

				break

			else:
				raise requests.HTTPError( f"--> Encountered status code: {response.status_code}\n" )

		except Exception as e:
			send_response = "not_found"
			print( f"Exception encountered: {e}" )
	return send_response


##################################### UniProt ######################################
####----------------------------------------------------------------------------####
def get_uniprot_seq( uni_id, max_trials = 5, wait_time = 5, return_id = False ):
	"""
	Obtain Uniprot seq for the specified Uniprot ID.

	Input:
	----------
	uni_id --> Uniprot accession.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
	
	Returns:
	----------
	Sequence for the given Uni ID.
		Uni ID is returned if specified.
	An empty list will be returned if couldn't find the sequence for the given Uni ID.
	"""
	url = f"http://www.uniprot.org/uniprot/{uni_id}.fasta"

	data = send_request( url, _format = "text", max_trials = max_trials, wait_time = wait_time )
	if data in ["not_found", "bad_request"]:
		response = [uni_id, []] if return_id else []

	else:
		seq_record = [str( record.seq ) for record in SeqIO.parse( StringIO( data ), 'fasta' )]

		if seq_record == []:
			response = [uni_id, []] if return_id else []
		else:
			response = [uni_id, seq_record[0]] if return_id else seq_record[0]

	return response


######################################## PDB #######################################
####----------------------------------------------------------------------------####
def pdb_valid( file_name: str, ext: str ):
	"""
	Check if a valid PDB file has been downloaded.
    PDB file is valid if:
        Can be read with Biopython Parser.
        Contains at least 1 model.
    Use PDB or MMCIF Parser as needed.

    Input:
    ----------
    file_name --> Path for the downloaded PDB file.
    ext --> pdb or cif.

    Returns:
    ----------
    True if PDB file is valid else False.
    """
	success = False
	try:
		if ext == "cif":
			structure = MMCIFParser().get_structure( "cif", file_name )

		elif ext == "pdb":
			structure = PDBParser().get_structure( "pdb", file_name )

		if len( list( structure.get_models() ) ) == 0:
			success = False
		else:
			success = True

		return success

	except PDBConstructionException:
		return success


def download_pdb( pdb_id: str, ext: str, file_name: str, 
					max_trials: int = 5, wait_time: int = 5,
					return_id: bool = True ):
	"""
	Download the PDB entry in the specified format.

	Input:
	----------
	pdb_id --> PDB ID to be downloaded.
	ext --> pdb or cif.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
	return_id --> return thr pdb_id if True.

	Returns:
	----------
	pdb_id --> same as input.
	"""
	pdb_id = pdb_id.lower()

	if ext =="cif":
		url = f"https://files.rcsb.org/download/{pdb_id}.cif"

	elif ext == "pdb":
		url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

	else:
		raise ValueError( "Incorrect file format (Choose .pdb/,cif)..." )

	response = send_request( url, _format = None, max_trials = max_trials, wait_time = wait_time )

	if response not in ["not_found", "bad_request"]:
		success = True

		# file_name = f"./{pdb_id}.{ext}"
		write_to_file( response, file_name, "w" )

	else:
		print( f"HTTP response - {response}" )
		success = False

	if success:
		if pdb_valid( file_name, ext ):
			result = pdb_id if return_id else None
		else:
			result = pdb_id if return_id else None
	else:
		result = False

	return result


#################################### PDB REST API ##################################
####----------------------------------------------------------------------------####
class PdbRestApi():
	def __init__( self, entry_id: str,
					entry_file: Optional[str] = None,
					entity_file: Optional[str] = None,
					max_trials: Optional[int] = 5,
					wait_time: Optional[int] = 10 ):
		# Entry ID aka PDB ID.
		self.entry_id = entry_id
		self.entry_file = entry_file
		self.entity_file = entity_file
		# Dict to store Entry level data.
		self.entry_data = {}
		# Dict to store Entity level data.
		self.entity_data = {}

		# Accessory attributes.
		self.max_trials = max_trials
		self.wait_time = wait_time

		# if already downloaded, use existing entry and entity dicts.
		entry_dict_exists = self.load_predownloaded_data()
		if not entry_dict_exists:
			# Retrieve entry_data.
			self.retrieve_entry_data()

	##------------------------------------------------------------------------------
	# URL methods.
	##------------------------------------------------------------------------------
	def get_entry_url( self ) -> str:
		"""
		Return the PDB REST Entry URL for the given entry_id (PDB ID).
		"""
		entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{self.entry_id}"
		return entry_url


	def get_polymer_entity_url( self, entity_id: str ) -> str:
		"""
		Return the PDB REST Entity URL for the given entry_id (PDB ID) and entity_id.
		"""
		entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{self.entry_id}/{entity_id}"
		return entity_url


	def load_predownloaded_data( self ) -> bool:
		"""
		Load the entry and entity dict from pre-downloaded JSON files.
		"""
		if self.entry_file is not None:
			if os.path.exists( self.entry_file ):
				self.entry_data = read_json( self.entry_file )
				entry_dict_exists = True
			else:
				entry_dict_exists = False

		if self.entity_file is not None:
			if os.path.exists( self.entity_file ):
				self.entity_data = read_json( self.entity_file )

		return entry_dict_exists

	##------------------------------------------------------------------------------
	# Data retrieval (from PDB) methods.
	##------------------------------------------------------------------------------
	def retrieve_entry_data( self ):
		"""
		Retrieve entry level information from PDB for the given entry_id.
		"""
		entry_url = self.get_entry_url()
		data = send_request( entry_url, _format = "json",
							max_trials = self.max_trials,
							wait_time = self.wait_time )

		if data in ["not_found", "bad_request"]:
			warnings.warn( "Could not retrieve info. for the entry_id" +
							f" = {self.entry_id}..." )
			# raise requests.HTTPError( "Could not retrieve info. for the entry_id" +
			# 							f" = {self.entry_id}..." )
			self.entry_data = None
		else:
			self.entry_data = data


	def retrieve_polymer_entity_data( self, entity_id ):
		"""
		Retrieve entity level information from PDB for the given entity_id.
		Use the existing info. if already exists.
		"""
		if entity_id not in self.entity_data.keys():
			entity_url = self.get_polymer_entity_url( entity_id )
			data = send_request( entity_url, _format = "json",
								max_trials = self.max_trials,
								wait_time = self.wait_time )

			if data in ["not_found", "bad_request"]:
				raise requests.HTTPError( f"Could not retrieve info. for the entry_id" +
								f" = {self.entry_id} and entity_id = {entity_id}..." )
			else:
				self.entity_data[entity_id] = data


	def retrieve_all_polymer_entity_data( self ):
		"""
		Retrieve entity level information from PDB for all entities associated with an entry_id.
		"""
		polymer_entity_ids = self.get_polymer_entities()

		for entity_id in polymer_entity_ids:
			self.retrieve_polymer_entity_data( entity_id )


	##------------------------------------------------------------------------------
	# Methods to get required information.
	##------------------------------------------------------------------------------
	def get_polymer_entry_container_identifiers( self ) -> Dict:
		"""
		Return entry_container_identifiers for a given entry_id.
		"""
		if "rcsb_entry_container_identifiers" in self.entry_data.keys():
			pol_entry_cont_id = self.entry_data["rcsb_entry_container_identifiers"]
		else:
			pol_entry_cont_id = {}

		return pol_entry_cont_id


	def get_polymer_entity_container_identifiers( self, entity_id: str ) -> Dict:
		"""
		Return polymer_entity_container_identifiers for a given entity_id.
		"""
		if "rcsb_polymer_entity_container_identifiers" in self.entity_data[
															entity_id].keys():
			pol_entity_cont_id = self.entity_data[entity_id][
										"rcsb_polymer_entity_container_identifiers"
										]
		else:
			pol_entity_cont_id = {}

		return pol_entity_cont_id


	def get_all_entities( self ) -> List:
		"""
		Get the entity_ids for all the polymer entities in a given entry.
		Returns a list of all polymer entity_ids.
		"""
		pol_entry_cont_id = self.get_polymer_entry_container_identifiers()
		if "entity_ids" in pol_entry_cont_id.keys():
			entity_ids = pol_entry_cont_id["entity_ids"]
		else:
			entity_ids = []

		return entity_ids


	def get_polymer_entities( self ) -> List:
		"""
		Get the entity_ids for all the polymer entities in a given entry.
		Returns a list of all polymer entity_ids.
		"""
		pol_entry_cont_id = self.get_polymer_entry_container_identifiers()
		if "polymer_entity_ids" in pol_entry_cont_id.keys():
			polymer_entity_ids = pol_entry_cont_id["polymer_entity_ids"]
		else:
			polymer_entity_ids = []

		return polymer_entity_ids


	def get_uniprot_ids_for_entity( self, entity_id: str ) -> List:
		"""
		uniprot_id can be obtained at the entity level.
		Get all uniprot_id for a given polymer entity_id.
		"""
		pol_entity_cont_id = self.get_polymer_entity_container_identifiers(
																	entity_id )
		if "uniprot_ids" in pol_entity_cont_id.keys():
			uniprot_ids = pol_entity_cont_id["uniprot_ids"]
		else:
			uniprot_ids = []
		return uniprot_ids


	def get_asym_ids_for_entity( self, entity_id: str ) -> List:
		"""
		asym_id can be obtained at the entity level.
		Get all asym_id for a given polymer entity_id.
		"""
		pol_entity_cont_id = self.get_polymer_entity_container_identifiers( entity_id )
		if "asym_ids" in pol_entity_cont_id.keys():
			asym_ids = pol_entity_cont_id["asym_ids"]
		else:
			asym_ids = []
		return asym_ids


	def get_auth_asym_ids_for_entity( self, entity_id: str ) -> List:
		"""
		auth_asym_id can be obtained at the entity level.
		Get all auth_asym_id for a given polymer entity_id.
		"""
		pol_entity_cont_id = self.get_polymer_entity_container_identifiers( entity_id )
		if "auth_asym_ids" in pol_entity_cont_id.keys():
			auth_asym_ids = pol_entity_cont_id["auth_asym_ids"]
		else:
			auth_asym_ids = []
		return auth_asym_ids


	def get_poly_entity_align( self, entity_id: str ) -> List:
		"""
		Get the UniProt and PDB start residue position for the entity and the length.
		End residue position = start + length
		For entities with multiple aligned regions
		"""
		if "rcsb_polymer_entity_align" in self.entity_data[entity_id].keys():
			poly_ent_align = self.entity_data[entity_id]["rcsb_polymer_entity_align"][0]
			# We are only selecting the 1st aligned region.
			aligned_region = poly_ent_align["aligned_regions"]

			if len( aligned_region ) == 1:
				aligned_region = aligned_region[0]

				length = int( aligned_region["length"] )
				uni_start = int( aligned_region["ref_beg_seq_id"] )
				uni_end = uni_start + length - 1
				
				pdb_start = int( aligned_region["entity_beg_seq_id"] )
				pdb_end = pdb_start + length - 1

				uni_pos = f"{uni_start}-{uni_end}"
				pdb_pos = f"{pdb_start}-{pdb_end}"

			else:
				uni_pos, pdb_pos = "", ""
				length = 0
		else:
			uni_pos, pdb_pos = "", ""
			length = 0

		return [uni_pos, pdb_pos], length


####################################### SIFTS ######################################
####----------------------------------------------------------------------------####
def download_sifts_mapping( pdb_id: str, file_path: str ):
	"""
	Fetch the PDB to UniProt mapping from SIFTS.
	Save as an XML file.
	file_path has a .xml extension.
	"""
	url = f"https://www.ebi.ac.uk/pdbe/files/sifts/{pdb_id}.xml.gz"
	cmd = ["curl", f"{url}", "--output", f"{file_path}.gz", "--silent"]
	run_subprocess( cmd )
	cmd = ["gunzip", "-f", f"{file_path}.gz"]
	run_subprocess( cmd )
	if os.path.exists( file_path ):
		success = True
	else:
		success = False
	return success



def parse_sifts_xml( file: str ):
	"""
	Extract the PDB-UniProt mapping from the SIFTS XML file.
	The data is stored as a nested dict for every chain in the PDB.
	Residues present in the struct ("resolved") and those missing ("missing") are segregated.
	"""
	root = ET.parse( file ).getroot()

	sifts_dict = {}

	for parent in root:
		if "entity" in parent.tag:
			for child in parent:
				if "segment" in child.tag:
					for subchild in child:
						if "listResidue" in subchild.tag:
							for leaf in subchild:
								is_null = False
								allow = [False, False]
								# Just to check if both PDB and UniProt identifiers are
								# 	present for a residue.
								for res_detail in leaf:
									if res_detail.attrib["dbSource"] == "PDB":
										allow[0] = True
									elif res_detail.attrib["dbSource"] == "UniProt":
										allow[1] = True
								if not all( allow ):
									continue

								for res_detail in leaf:
									data = res_detail.attrib

									if data["dbSource"] == "PDB":
										chain_id = data["dbChainId"]

										if chain_id not in sifts_dict:
											sifts_dict[chain_id] = {k: {} for k in ["resolved", "missing", "sequence"]}
											for key in sifts_dict[chain_id]:
												if key == "sequence":
													sifts_dict[chain_id][key] = {"PDB": [], "Uniprot": []}
												else:
													sifts_dict[chain_id][key] = {k:[] for k in
																	["PDB residue", "PDB position",
																		"Uniprot ID", "Uniprot residue", "Uniprot position"]
																	}

										sifts_dict[chain_id]["sequence"]["PDB"].append( aa_3_to_1( data["dbResName"] ) )

										if data["dbResNum"] == "null":
											is_null = True
											sifts_dict[chain_id]["missing"]["PDB residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["missing"]["PDB position"].append( data["dbResNum"] )
										else:
											sifts_dict[chain_id]["resolved"]["PDB residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["resolved"]["PDB position"].append( str( data["dbResNum"] ) )

									if data["dbSource"] == "UniProt":
										sifts_dict[chain_id]["sequence"]["Uniprot"].append( data["dbResName"] )
										if is_null:
											sifts_dict[chain_id]["missing"]["Uniprot ID"].append( data["dbAccessionId"] )
											sifts_dict[chain_id]["missing"]["Uniprot residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["missing"]["Uniprot position"].append( data["dbResNum"] )
										else:
											sifts_dict[chain_id]["resolved"]["Uniprot ID"].append( data["dbAccessionId"] )
											sifts_dict[chain_id]["resolved"]["Uniprot residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["resolved"]["Uniprot position"].append( str( data["dbResNum"] ) )
	return sifts_dict


####################################### CASP #######################################
####----------------------------------------------------------------------------####
def get_casp_entry( casp_id: str, max_trials: int = 5, wait_time: int = 5 ):
	"""
	Get the FASTA sequence and the structure for the required CASP entry.
	Return the entry sequences and structure file as string.
	"""
	fasta_url = f"https://predictioncenter.org/casp15/target.cgi?target={casp_id}&view=sequence"
	struct_url = f"https://predictioncenter.org/casp15/target.cgi?target={casp_id}&view=template"

	fasta_response = send_request( fasta_url, _format = None, max_trials = max_trials,
									wait_time = wait_time )
	if fasta_response in ["not_found", "bad_request"]:
		raise requests.HTTPError( f"FASTA file for {casp_id} could not be obtained..." )

	# struct_response = send_request( struct_url, _format = None, max_trials = max_trials,
	# 								wait_time = wait_time )
	# if fasta_response in ["not_found", "bad_request"]:
	# 	raise requests.HTTPError( f"Structure file for {casp_id} could not be obtained..." )

	return fasta_response #, struct_response
