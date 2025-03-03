"""
This script contains general purpose accessory functions.
"""

import json
import subprocess
from io import StringIO
from typing import List, Dict, TextIO
from Bio import SeqIO
import requests
import ml_collections as mlc


def open_file_handler( file_path: str, mode: str ) -> TextIO:
	"""
	Open a file handler in the desired mode.

	Input:
	----------
	file_path --> path for the file to be opened.
	mode --> mode in which to open the file (r, w, a, etc.).

	Returns:
	----------
	File handler for the specified file.
	"""
	if "b" in mode:
		fh = open( file_path, mode )
	else:
		fh = open( file_path, mode, encoding = "utf-8" )
	return fh


def read_json( file_path: str ) -> Dict:
	"""
	Read a JSON file and return the dict.

	Input:
	----------
	file_path --> path for the file to be opened.

	Returns:
	----------
	data --> dict from a JSON file.
	"""
	f = open_file_handler( file_path, "r" )
	data = json.load( f )
	f.close()
	return data


def write_json( dict_: Dict, file_path: str ) -> None:
	"""
	Save dict to a JSON file.

	Input:
	----------
	file_path --> path for the file to be opened.

	Returns:
	----------
	None
	"""
	w = open_file_handler( file_path, "w" )
	json.dump( dict_, w, indent = 4 )
	w.close()


def read_configdict_from_json( file_path: str ) -> mlc.ConfigDict:
	"""
	Read from a mlc.ConfigDict saved JSON file and 
		return mlc.ConfigDict object.

	Input:
	----------
	file_path --> path for the file to be opened.

	Returns:
	----------
	Dict as an mlc.ConfigDict.
	"""
	config_dict = read_json( file_path )
	return mlc.ConfigDict( json.loads( config_dict ) )



def write_configdict_to_json( config_dict: mlc.ConfigDict,
								file_path: str ) -> None:
	"""
	Save an mlc.Configdict object to JSON file.

	Input:
	----------
	config_dict --> an mlc.ConfidDict.
	file_path --> path for the file to be opened.

	Returns:
	----------
	None
	"""
	write_json( json.loads( config_dict.to_json() ), file_path )


def write_to_file( content: str, file_name: str, mode: str ) -> None:
	"""
	Given a Response object, write to a file.

	Input:
	----------
	file_path --> path for the file to be opened.

	Returns:
	----------
	None
	"""
	w = open_file_handler( file_name, mode )
	w.write( content.text )
	w.close()


def read_fasta_from_response( response: requests.Response ) -> Dict:
	"""
	Given a FASTA file as a str, obtain the sequences for all chains.

	Input:
	----------
	response --> requests.Response object for the requested URL.

	Returns:
	----------
	fasta_dict --> dict with integer keys and sequences as values.
	"""
	fasta_content = StringIO( response.content.decode( "utf-8" ) )

	fasta_dict = {}
	idx = 0
	for record in SeqIO.parse( fasta_content, "fasta" ):
		fasta_dict[idx] = str( record.seq )
		idx += 1

	return fasta_dict


def run_subprocess( command: List ) -> None:
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
