import subprocess
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFIO
from Bio.Data import PDBData
import gemmi
import io

from typing import List, Mapping, Sequence, Any

from openfold.data.mmcif_parsing import (
		_get_first_model, _get_protein_chains,
		_get_atom_site_list, ParsingResult, MmcifObject,
		ResidueAtPosition, ResiduePosition,
		mmcif_loop_to_list, _is_set )

ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]

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



def parse(
    *, file_id: str, mmcif_string: str, catch_all_errors: bool = True
) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    Returns:
      A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure("", handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(
                None, {(file_id, ""): "No protein chains found in this file."}
            )
        seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, seq in valid_chains.items()
        }

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in valid_chains:
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                position = ResiduePosition(
                    chain_id=atom.author_chain_id,
                    residue_number=int(atom.author_seq_num),
                    insertion_code=insertion_code,
                )
                seq_idx = (
                    int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                )
                current = seq_to_structure_mappings.get(
                    atom.author_chain_id, {}
                )
                current[seq_idx] = ResidueAtPosition(
                    position=position,
                    name=atom.residue_name,
                    is_missing=False,
                    hetflag=hetflag,
                )
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(
                        position=None,
                        name=monomer.id,
                        is_missing=True,
                        hetflag=" ",
                    )

        author_chain_to_sequence = {}
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            seq = []
            for monomer in seq_info:
                code = PDBData.protein_letters_3to1.get(monomer.id, "X")
                seq.append(code if len(code) == 1 else "X")
            seq = "".join(seq)
            author_chain_to_sequence[author_chain] = seq

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            raw_string=parsed_info,
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)



def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list("_exptl.", parsed_info)
    header["structure_method"] = ",".join(
        [experiment["_exptl.method"].lower() for experiment in experiments]
    )

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    # if "_pdbx_audit_revision_history.revision_date" in parsed_info:
    #     header["release_date"] = get_release_date(parsed_info)
    # else:
    #     logging.warning(
    #         "Could not determine release_date: %s", parsed_info["_entry.id"]
    #     )

    header["resolution"] = 0.00
    # for res_key in (
    #     "_refine.ls_d_res_high",
    #     "_em_3d_reconstruction.resolution",
    #     "_reflns.d_resolution_high",
    # ):
    #     if res_key in parsed_info:
    #         try:
    #             raw_resolution = parsed_info[res_key][0]
    #             header["resolution"] = float(raw_resolution)
    #             break
    #         except ValueError:
    #             logging.debug(
    #                 "Invalid resolution format: %s", parsed_info[res_key]
    #             )

    return header
