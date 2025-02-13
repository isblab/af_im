import requests
import xml.etree.ElementTree as ET
from io import StringIO
from Bio import SeqIO
import re

from typing import Dict


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
		return "A"
	elif aa in ["ARG", "Arg", "arg"]:
		return "R"
	elif aa in ["ASN", "Asn", "asn"]:
		return "N"
	elif aa in ["ASP", "Asp", "asp"]:
		return "D"
	elif aa in ["CYS", "Cys", "cys"]:
		return "C"
	elif aa in ["GLN", "Gln", "gln"]:
		return "Q"
	elif aa in ["GLU", "Glu", "glu"]:
		return "E"
	elif aa in ["GLY", "Gly", "gly"]:
		return "G"
	elif aa in ["HIS", "His", "his"]:
		return "H"
	elif aa in ["ILE", "Ile", "ile"]:
		return "I"
	elif aa in ["LEU", "Leu", "leu"]:
		return "L"
	elif aa in ["LYS", "Lys", "lys"]:
		return "K"
	elif aa in ["MET", "Met", "met"]:
		return "M"
	elif aa in ["PHE", "Phe", "phe"]:
		return "F"
	elif aa in ["PRO", "Pro", "pro"]:
		return "P"
	elif aa in ["SER", "Ser", "ser"]:
		return "S"
	elif aa in ["THR", "Thr", "thr"]:
		return "T"
	elif aa in ["TRP", "Trp", "trp"]:
		return "W"
	elif aa in ["TYR", "Tyr", "tyr"]:
		return "Y"
	elif aa in ["VAL", "Val", "val"]:
		return "V"
	else:
		return "X"


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
				if trial > ( max_trials/2 ):
					continue
				else:
					return "not_found"

			# Bad request.
			elif response.status_code == 400:
				if trial != ( max_trials - 1 ):
					continue
				else:
					return "bad_request"

			elif response.status_code == 200:
				if _format == None:
					return response
				elif _format == "json":
					return response.json()
				else:
					return response.text
				break
			
			else:
				raise Exception( f"--> Encountered status code: {response.status_code}\n" )
		except Exception as e:
			if trial != max_trials-1:
			# 	print( f"Trial {trial}: Exception {e} \t --> {url}" )
				continue
			else:
				return "not_found"


def write_content_to_file( response: requests.Response, file_name: str ) -> None:
	"""
	Given a Response object, write to a file.
	"""
	open( f"{file_name}", "wb" ).write( response.content )



def read_fasta_from_response( response: requests.Response ) -> Dict:
	"""
	Given a FASTA file as a str, obtain the sequences for all chains.
	"""
	fasta_content = StringIO( response.content.decode( "utf-8" ) )

	fasta_dict = {}
	idx = 0
	for record in SeqIO.parse( fasta_content, "fasta" ):
		fasta_dict[idx] = str( record.seq )
		idx += 1

	return fasta_dict



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
	if data == "not_found" or data == "bad_request":
		return [uni_id, []] if return_id else []
	
	else:
		seq_record = [str( record.seq ) for record in SeqIO.parse( StringIO( data ), 'fasta' )]

		if seq_record == []:
			return [uni_id, []] if return_id else []
		else:
			return [uni_id, seq_record[0]] if return_id else seq_record[0]



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

    try:
        if ext == "cif":
            models = MMCIFParser().get_structure( "cif", file_name )
            if len( models ) == 0:
                return False
            else:
                return True
            
        elif ext == "pdb":
            models = PDBParser().get_structure( "pdb", file_name )
            if len( models ) == 0:
                return False
            else:
                return True

    except:
        return False


def download_pdb( pdb_id: str, ext: str, max_trials: int = 5, wait_time: int = 5, return_id: bool = True ):
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
    import warnings
    warnings.filterwarnings("ignore")
    
    pdb_id = pdb_id.lower()

    if ext =="cif":
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"

    elif ext == "pdb":
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    else:
        raise Exception( "Incorrect file format (Choose .pdb/,cif)..." )

    response = send_request( url, _format = None, max_trials = max_trials, wait_time = wait_time )

    if response not in ["not_found", "bad_request"]:
    	success = True

    	file_name = f"./{pdb_id}.{ext}"
    	write_content_to_file( response, file_name )
    	# open( f"{file_name}", "wb" ).write( response.content )

    else:
    	success = False

    if success:
        if pdb_valid( file_name, ext ):
            return pdb_id if return_id else None
        else:
            return pdb_id if return_id else None



####################################### SIFTS ######################################
####----------------------------------------------------------------------------####
def download_sifts_mapping( pdb_id: str, max_trials: int = 5, wait_time: int = 5, return_id: bool = False ):
	"""
	Fetch the PDB to UniProt mapping from SIFTS.
	Save as an XML file.
	"""
	url = f"https://www.ebi.ac.uk/pdbe/files/sifts/{pdb_id}.xml"
	response = send_request( url, _format = None, max_trials = max_trials, wait_time = wait_time )

	if response not in ["not_found", "bad_request"]:
		success = True
		open( f"{pdb_id}.xml", "wb" ).write( response.content )
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

	for i, parent in enumerate( root ):
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
										
										if chain_id not in sifts_dict.keys():
											sifts_dict[chain_id] = {k: {} for k in ["resolved", "missing", "sequence"]}
											for key in sifts_dict[chain_id].keys():
												if key == "sequence":
													sifts_dict[chain_id][key] = {"PDB": [], "Uniprot": []}
												else:
													sifts_dict[chain_id][key] = {k:[] for k in 
																	["PDB Residue", "PDB position", "Uniprot ID", "Uniprot Residue", "Uniprot position"]
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

	fasta_response = send_request( fasta_url, _format = None, max_trials = max_trials, wait_time = wait_time )
	if fasta_response in ["not_found", "bad_request"]:
		raise Exception( f"FASTA file for {casp_id} could not be obtained..." )

	struct_response = send_request( struct_url, _format = None, max_trials = max_trials, wait_time = wait_time )
	if fasta_response in ["not_found", "bad_request"]:
		raise Exception( f"Structure file for {casp_id} could not be obtained..." )

	return fasta_response, struct_response



