import subprocess
from Bio.PDB import PDBParser, MMCIFIO
import gemmi

from typing import List, Mapping, Sequence, Any

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





def res_to_idx( res_pos: int ):
    """
    Convert an input residue position to index.
    Index = residue_position - 1

    Input:
    ----------
    res_pos --> residue position.

    Returns:
    ----------
    index --> index for the residue position.
    """
    return res_pos - 1



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


