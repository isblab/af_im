import ml_collections as mlc
import json
import subprocess
from Bio.PDB import PDBParser, MMCIFIO
import gemmi

from typing import List, Mapping, Sequence, Any, Dict

# from openfold.data.mmcif_parsing import (
# 		_get_first_model, _get_protein_chains,
# 		_get_atom_site_list, ParsingResult, MmcifObject,
# 		ResidueAtPosition, ResiduePosition,
# 		mmcif_loop_to_list, _is_set )

# ChainId = str
# PdbHeader = Mapping[str, Any]
# PdbStructure = PDB.Structure.Structure
# SeqRes = str
# MmCIFDict = Mapping[str, Sequence[str]]


def read_configdict_from_json( file_path: str ):
	"""
	Read from a mlc.ConfigDict saved JSON file and 
		return mlc.ConfigDict object.
	"""
	with open( file_path, 'r' ) as f:
		config_dict = json.loads( 
							json.load( f )
							 )
	return mlc.ConfigDict( config_dict )



def write_configdict_to_json( config_dict: mlc.ConfigDict, file_path: str ):
	"""
	Save an mlc.Configdict object to JSON file.
	"""
	with open( file_path, "w" ) as w:
		json.dump( config_dict.to_json(), w )


def read_json( file_path: str ):
	"""
	Parser for a JSON file, given the file path.
	"""
	with open( file_path, "r" ) as f:
		data = json.load( f )
	return data


def write_json( data: Dict,  file_path: str ):
	"""
	Write a dict to a JSON file, given the file path.
	"""
	with open( file_path, "w" ) as w:
		json.dump( data, w )



def pdb_to_cif_gemmi( pdb_file_path: str, cif_file_path: str ):
	"""
	Convert a .pdb file to a .cif file.

	Input:
	----------
	pdb_file_path --> Path to the .pdb file.
	cif_file_path --> Path to the .cif file.

	Returns:
	----------
	None
	"""
	struct = gemmi.read_structure( pdb_file_path )

	cif_doc = struct.make_mmcif_document()

	with open( cif_file_path, "w" ) as w:
		w.write( cif_doc.as_string() )



def pdb_to_cif_bio( pdb_file_path: str, cif_file_path: str ):
	"""
	Convert a .pdb file to a .cif file.
	Biopython does not preserve all the info while dumping to a .cif file.

	Input:
	----------
	pdb_file_path --> Path to the .pdb file.
	cif_file_path --> Path to the .cif file.

	Returns:
	----------
	None
	"""
	struct = PDBParser().get_structure( "pdb", pdb_file_path )

	cif = MMCIFIO()
	cif.set_structure( struct )
	cif.save( cif_file_path )


def run_subprocess( command: List ):
	"""
	Run shell command using subprocess.

	Input:
	----------
	command --> a list in subprocess acceptable format.

	Returns:
	----------
	None
	"""
	if len( command ) == 0:
		raise Exception( "Command cannot be empty..." )
	else:
		subprocess.call( command )


