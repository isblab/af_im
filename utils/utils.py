"""
This script contains general purpose accessory functions.
"""
from typing import List, Tuple, Dict, TextIO, Optional, Any
import json, os, shutil, subprocess, traceback, time, random, gzip
import pickle as pkl
from datetime import datetime
from io import StringIO
import numpy as np
from Bio import SeqIO
import requests
import ml_collections as mlc

import torch


def seed_worker( seed: int = 1 ):
	"""
	Set seed for PRNG.

	Input:
	----------
	seed: an integer seed for setting the state of the PRNGs.
	"""
	torch.manual_seed( seed+1 )
	# torch.cuda.manual_seed( worker_seed )
	torch.cuda.manual_seed_all( seed )
	np.random.seed( seed )
	random.seed( seed )


def get_gpu_mem_mb( gpu_id: int ):
	"""
	Obtain the current GPU memory usage (in MB) for a specific device by
		querying `nvidia-smi`.
	This function invokes `nvidia-smi` as a subprocess and parses the reported
		memory usage for the given GPU ID.

	Inputs:
	----------
	gpu_id: iIndex of the GPU as recognized by `nvidia-smi` (after any
		CUDA_VISIBLE_DEVICES remapping).

	Returns
	----------
	Memory currently in use on the GPU, in MB.
	"""
	out = subprocess.check_output(
		[
			"nvidia-smi",
			f"--id={gpu_id}",
			"--query-gpu=memory.used",
			"--format=csv,noheader,nounits"
		],
		encoding = "utf-8"
	)
	return int( out.strip() )


def parse_nested_dict( dict_: Dict, action: str, 
						device: Optional[str] = "cuda" ):
	for k in dict_:
		if isinstance( dict_[k], Dict ):
			dict_[k] = parse_nested_dict( dict_[k], action, device )
		else:
			if isinstance( dict_[k], np.ndarray ):
				if action == "to_tensor":
					dict_[k] = torch.from_numpy( dict_[k] )

			if isinstance( dict_[k], torch.Tensor ):
				if action == "add_dim":
					dict_[k] = dict_[k].unsqueeze( 0 )
				elif action == "add_to_device":
					dict_[k] = dict_[k].to( device )
				elif action == "detach":
					dict_[k] = dict_[k].detach().cpu()

	return dict_


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

################################################################################
################################################################################
def log_error( error_file: str ):
	"""
	Write the error traceback on disk at the specified file path.
	Add a time stamp to the error file.
	
	Input:
	----------
	error_file: file to write the error traceback.
	"""
	base, ext = os.path.splitext( error_file )
	current_datetime = datetime.now()
	timestamp = current_datetime.strftime( "%d_%m_%Y_%H_%M_%S" )
	error_file = base + f"_{timestamp}" + ext
	w = open_file_handler( error_file, "w" )
	w.write( traceback.format_exc() )
	w.close()

################################################################################
################################################################################
def open_file_handler( file_path: str, mode: str ) -> TextIO:
	"""
	Open a file handler in the desired mode.

	Input:
	----------
	file_path: path for the file to be opened.
	mode: mode in which to open the file (r, w, a, etc.).

	Returns:
	----------
	fh: File handler for the specified file.
	"""
	if "b" in mode:
		fh = open( file_path, mode )
	else:
		fh = open( file_path, mode, encoding = "utf-8" )
	return fh

################################################################################
################################################################################
def read_json( file_path: str ) -> Dict:
	"""
	Read a JSON file and return the dict.

	Input:
	----------
	file_path: path for the file to be opened.

	Returns:
	----------
	data: dict from a JSON file.
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
	file_path: path for the file to be opened.

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
	file_path: path for the file to be opened.

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
	config_dict: an mlc.ConfidDict.
	file_path: path for the file to be opened.

	Returns:
	----------
	None
	"""
	write_json( json.loads( config_dict.to_json() ), file_path )

################################################################################
################################################################################
def read_pkl_gz( file_path: str ) -> Any:
	"""
	Read the input .pkl.gz file and return the content.

	Input:
	----------
	file_path: .pkl.gz file file path.

	Returns:
	----------
	data: contents parsed from the input .pkl.gz file.
	"""
	with gzip.open( file_path, "rb" ) as f:
		data = pkl.load( f )
	return data


def write_pkl_gz( data: Any, file_path: str ):
	"""
	Write the given input object on disk at
		the dpecified file path.
	Only np.ndarray, torch.tensor, dict supported as of now.

	Input:
	----------
	data: input object (dict/ np.ndarray/pd.DataFrame, etc.)
	file_path: .pkl.gz file file path.
	"""
	# if not all( [isinstance( data, obj ) for obj in [np.ndarray, dict, torch.Tensor]] ):
		# raise NotImplemented(
		# 	f"Object of type {type( data )} is currently" +
		# 	" not supported to be wriiten in .pkl.gz format." +
		# 	" Supported formats include: np.ndarray, torch.Tensor, dict." )
	with gzip.open( file_path, "wb" ) as w:
		pkl.dump( data, w, protocol = pkl.HIGHEST_PROTOCOL )

################################################################################
################################################################################
def write_to_file( content: str, file_name: str, mode: str ) -> None:
	"""
	Given a Response.context attribute, write to a file.

	Input:
	----------
	file_path: path for the file to be opened.

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
	response: requests.Response object for the requested URL.

	Returns:
	----------
	fasta_dict: dict with integer keys and sequences as values.
	"""
	fasta_content = StringIO( response.content.decode( "utf-8" ) )

	fasta_dict = {}
	idx = 0
	for record in SeqIO.parse( fasta_content, "fasta" ):
		fasta_dict[idx] = str( record.seq )
		idx += 1

	return fasta_dict

################################################################################
################################################################################
def create_dir( dir_path: str ):
	"""
	Create a directory.

	Input:
	----------
	dir_path: path to the directory to be created.

	Returns:
	----------
	None
	"""
	os.makedirs( dir_path, exist_ok = True )


def remove_dir( dir_path: str ):
	"""
	Remove a directory.

	Input:
	----------
	dir_path: path to the directory to be removed.

	Returns:
	----------
	None
	"""
	if os.path.exists( dir_path ):
		shutil.rmtree( dir_path )
	else:
		raise FileNotFoundError(
			f"The provided directory path - {dir_path} - does not exist..."
		)

################################################################################
################################################################################
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
			result = subprocess.run( command,
									capture_output = True,
									text = True,
									check = True )
			if stdout_file is not None:
				w = open_file_handler( stdout_file, "w" )
				w.write( result.stdout )
				w.close()
				
				# retcode = subprocess.run( command,
				# 						stdout_obj = w,
				# 						capture_output = True,
				# 						text = True,
				# 						check = True )

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
