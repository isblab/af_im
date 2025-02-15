"""
This script contains functions for handling API requests, downloading files.
"""

import time
import xml.etree.ElementTree as ET
from io import StringIO
import warnings
import requests
from Bio import SeqIO
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from utils import write_to_file

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
			response = requests.get( url, timeout = 300 )

			# Resource not found.
			if response.status_code == 404:
				# Try till half the no. of max_trials.
				# 	Don't want it stuck for too long if the URL doesn't actually exist.
				if trial > ( max_trials/2 ):
					return "not_found"
				time.sleep( wait_time )

			# Bad request.
			elif response.status_code == 400:
				if trial == ( max_trials - 1 ):
					return "bad_request"
				time.sleep( wait_time )

			elif response.status_code == 200:

				if _format is None:
					send_response = response
				elif _format == "json":
					send_response = response.json()
				else:
					send_response = response.text

				return send_response
				break

			else:
				raise requests.HTTPError( f"--> Encountered status code: {response.status_code}\n" )

		except requests.HTTPError:
			return "not_found"



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
			models = MMCIFParser().get_structure( "cif", file_name )
			if len( models ) == 0:
				success = False
			else:
				success = True

		elif ext == "pdb":
			models = PDBParser().get_structure( "pdb", file_name )
			if len( models ) == 0:
				success = False
			else:
				success = True

		return success

	except PDBConstructionException:
		return success



def download_pdb( pdb_id: str, ext: str, max_trials: int = 5, wait_time: int = 5,
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

		file_name = f"./{pdb_id}.{ext}"
		write_to_file( response.content, file_name, "wb" )

	else:
		success = False

	if success:
		if pdb_valid( file_name, ext ):
			result = pdb_id if return_id else None
		else:
			result = pdb_id if return_id else None
	else:
		result = False

	return result



####################################### SIFTS ######################################
####----------------------------------------------------------------------------####
def download_sifts_mapping( pdb_id: str, max_trials: int = 5, wait_time: int = 5 ):
	"""
	Fetch the PDB to UniProt mapping from SIFTS.
	Save as an XML file.
	"""
	url = f"https://www.ebi.ac.uk/pdbe/files/sifts/{pdb_id}.xml"
	response = send_request( url, _format = None, max_trials = max_trials, wait_time = wait_time )

	if response not in ["not_found", "bad_request"]:
		success = True
		write_to_file( response.content, f"{pdb_id}.xml", "wb" )
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
																	["PDB Residue", "PDB position",
																		"Uniprot ID", "Uniprot Residue", "Uniprot position"]
																	}

										sifts_dict[chain_id]["sequence"]["PDB"].append( aa_3_to_1( data["dbResName"] ) )
										if data["dbResNum"] == "null":
											is_null = True
											sifts_dict[chain_id]["missing"]["PDB Residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["missing"]["PDB position"].append( data["dbResNum"] )
										else:
											sifts_dict[chain_id]["resolved"]["PDB Residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["resolved"]["PDB position"].append( data["dbResNum"] )

									if data["dbSource"] == "UniProt":
										sifts_dict[chain_id]["sequence"]["Uniprot"].append( data["dbResName"] )
										if is_null:
											sifts_dict[chain_id]["missing"]["Uniprot ID"].append( data["dbAccessionId"] )
											sifts_dict[chain_id]["missing"]["Uniprot Residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["missing"]["Uniprot position"].append( data["dbResNum"] )
										else:
											sifts_dict[chain_id]["resolved"]["Uniprot ID"].append( data["dbAccessionId"] )
											sifts_dict[chain_id]["resolved"]["Uniprot Residue"].append( data["dbResName"] )
											sifts_dict[chain_id]["resolved"]["Uniprot position"].append( data["dbResNum"] )


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

	struct_response = send_request( struct_url, _format = None, max_trials = max_trials,
									wait_time = wait_time )
	if fasta_response in ["not_found", "bad_request"]:
		raise requests.HTTPError( f"Structure file for {casp_id} could not be obtained..." )

	return fasta_response, struct_response
