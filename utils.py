"""
This script contains general purpose accessory functions.
"""

import json
import subprocess
from io import StringIO
from typing import List, Dict
from Bio import SeqIO
import requests
import ml_collections as mlc


def open_file_handler( file_path: str, mode: str ):
	"""
	Open a file handler in the desired mode.
	"""
	fh = open( file_path, mode, encoding = "utf-8" )
	return fh


def read_json( file_path: str ):
	"""
	Read a JSON file and return the dict.
	"""
	# with open( file_path, "r", encoding = "utf-8" ) as f:
	f = open_file_handler( file_path, "r" )
	dict_ = json.load( f )
	f.close()
	return dict_


def write_json( dict_: Dict, file_path: str ):
	"""
	Save dict to a JSON file.
	"""
	# with open( file_path, "w", encoding = "utf-8" ) as w:
	w = open_file_handler( file_path, "r" )
	json.dump( dict_, w, indent = 4 )
	w.close()


def read_configdict_from_json( file_path: str ):
	"""
	Read from a mlc.ConfigDict saved JSON file and 
		return mlc.ConfigDict object.
	"""
	config_dict = read_json( file_path )
	return mlc.ConfigDict( json.loads( config_dict ) )



def write_configdict_to_json( config_dict: mlc.ConfigDict, file_path: str ):
	"""
	Save an mlc.Configdict object to JSON file.
	"""
	write_json( json.loads( config_dict.to_json() ), file_path )


def write_to_file( content: str, file_name: str, mode: str ) -> None:
	"""
	Given a Response object, write to a file.
	"""
	# with open( f"{file_name}", mode, encoding = "utf-8" ) as w:
	w = open_file_handler( file_name, mode )
	w.write( content )
	w.close()


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
	if len( command ) != 0:
		subprocess.call( command )
	else:
		raise ValueError( "Command cannot be empty..." )

