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


def read_json( file_path: str ):
	"""
	Read a JSON file and return the dict.
	"""
	with open( file_path, 'r' ) as f:
		dict_ = json.load( f )
	return dict_


def write_json( dict_: Dict, file_path: str ):
	"""
	Save dict to a JSON file.
	"""
	with open( file_path, "w" ) as w:
		json.dump( dict_, w, indent = 4 )



def read_configdict_from_json( file_path: str ):
	"""
	Read from a mlc.ConfigDict saved JSON file and 
		return mlc.ConfigDict object.
	"""
	# with open( file_path, 'r' ) as f:
	# 	config_dict = json.load( f )
	config_dict = read_json( file_path )
	return mlc.ConfigDict( config_dict )



def write_configdict_to_json( config_dict: mlc.ConfigDict, file_path: str ):
	"""
	Save an mlc.Configdict object to JSON file.
	"""
	# with open( file_path, "w" ) as w:
	# 	json.dump( json.loads( config_dict.to_json() ), w, indent = 4 )
	write_json( json.loads( config_dict.to_json() ), file_path )


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


