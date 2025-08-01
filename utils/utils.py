"""
This script contains general purpose accessory functions.
"""

import json, subprocess, traceback, time
from datetime import datetime
from io import StringIO
from typing import List, Tuple, Dict, TextIO
from Bio import SeqIO
import requests
import ml_collections as mlc



def ranges( positions: List ) -> List[Tuple[int, int]]:
	"""
	Get tuples of continous residue positions.
	e.g. [1, 2, 3, 4, 7, 8, 9, 10]
		[(1, 4), (7, 10)]
	"""
	positions = list( map( int, positions ) )
	positions = sorted( set( positions ) )
	# Get start, end positions for each continous confident patch.
	gaps = [[x, y] for x, y in zip( positions, positions[1:] ) if x+1 < y]
	edges = iter( positions[:1] + sum( gaps, [] ) + positions[-1:] )
	return list( zip( edges, edges ) )



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


def run_subprocess( command: List,
	stdout_file: str = None,
	stderr_file: str = "err_log" ) -> None:
	"""
	Run shell command using subprocess.

	Input:
	----------
	command --> a list in subprocess acceptable format.
	stdout_file --> file_path to save stdout.
	stderr_file --> file to log stderr is it occurs.

	Returns:
	----------
	None
	"""
	if len( command ) != 0:
		try:
			if stdout_file is None:
				retcode = subprocess.run( command,
										capture_output = True,
										text = True,
										check = True )
			else:
				w = open_file_handler( stdout_file, "w" )
				retcode = subprocess.run( command,
										stdout_obj = w,
										capture_output = True,
										text = True,
										check = True )
				w.close()

		except subprocess.CalledProcessError as e:
			print( "Writing error to log file..." )
			current_datetime = datetime.now()
			timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
			stderr_file = f"{stderr_file}_{timestamp}.txt"
			w = open_file_handler( stderr_file, "w" )
			w.write( e.stderr or "No error output captured" )
			w.write( "\n\nTraceback\n" )
			w.write( traceback.format_exc() )
			w.close()
			time.sleep( 5 )

	else:
		raise ValueError( "Command cannot be empty..." )
