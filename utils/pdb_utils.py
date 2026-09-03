"""
This script contains functions to read/write modify PDB files.
"""
import string
import os
from typing import List, Dict, Tuple, Iterator
import numpy as np
from scipy.spatial import distance_matrix

import Bio
from Bio.PDB import (
	PDBParser, MMCIFParser,
	MMCIFIO, PDBIO,
	Select,
	Structure, Model, Residue )
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from utils.utils import open_file_handler, remove_dir

# warnings.filterwarnings("ignore")

# Taken from openfold.np.protein.py
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert PDB_MAX_CHAINS == 62



####################################################################################
####----------------------------------------------------------------------------####
def get_chain_id( idx: int ) -> str:
	"""
	Map a 0-indexed chain ID to a alphabetical chain identifier.
	Chain IDs are assigned from the ordered set:
		"A-Z" followed by "0-9", allowing up to 36 unique chains.

	Inputs:
	----------
	idx: 0-indexed chain ID.

	Returns:
	----------
	Alphabetical chain ID.
	"""
	alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

	if idx < len( alphabet ):
		chain_id = alphabet[idx]

	else:
		raise ValueError( "Too many chains..." )
	return chain_id


####################################################################################
####----------------------------------------------------------------------------####
def amino_acid_radii() -> Dict[int, float]:
	"""
	Returns a dictionary mapping tokenized amino acids to their radii.
	Radii taken form -> Table-3: https://doi.org/10.1038/s41598-020-61205-w
	Tokenization taken from openfold/np/residue_constants.py
	"""
	radii = {
		0: 3.2,    # Ala
		1: 3.65,   # Cys, also U
		2: 4.04,   # Asp, also B
		3: 4.63,   # Glu, also Z
		4: 4.99,   # Phe
		5: 1.72,   # Gly
		6: 4.73,   # His
		7: 3.94,   # ile
		8: 5.02,   # Lys
		9: 4.24,   # Leu
		10: 4.47,  # Met
		11: 4.04,  # Asn
		12: 3.61,  # pro
		13: 4.64,  # Gln
		14: 5.6,   # Arg
		15: 3.39,  # Ser
		16: 3.56,  # Thr
		17: 3.55,  # Val
		18: 5.38,  # Trp
		19: 5.36,  # Tyr
		20: 4.0    # X, J, O
	}
	return radii

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


def get_distance_map( coords1: np.array, coords2: np.array ):
	"""
	Get the distance map for the given coordinates.
	"""
	distance_map = distance_matrix( coords1, coords2 )
	return distance_map

################################################################################
################################################################################
# def prep_output( out, batch, feature_dict, feature_processor, config_preset, subtract_plddt ):
#     plddt = out["plddt"]

#     plddt_b_factors = np.repeat(
#         plddt[..., None], residue_constants.atom_type_num, axis=-1
#     )

#     if subtract_plddt:
#         plddt_b_factors = 100 - plddt_b_factors

#     # Prep protein metadata
#     template_domain_names = []
#     template_chain_index = None
#     if feature_processor.config.common.use_templates and "template_domain_names" in feature_dict:
#         template_domain_names = [
#             t.decode("utf-8") for t in feature_dict["template_domain_names"]
#         ]

#         # This works because templates are not shuffled during inference
#         template_domain_names = template_domain_names[
#                                 :feature_processor.config.predict.max_templates
#                                 ]

#         if "template_chain_index" in feature_dict:
#             template_chain_index = feature_dict["template_chain_index"]
#             template_chain_index = template_chain_index[
#                                    :feature_processor.config.predict.max_templates
#                                    ]

#     no_recycling = feature_processor.config.common.max_recycling_iters
#     remark = ', '.join([
#         f"no_recycling={no_recycling}",
#         f"max_templates={feature_processor.config.predict.max_templates}",
#         f"config_preset={config_preset}",
#     ])

#     unrelaxed_protein = protein.from_prediction(
#         features=batch,
#         result=out,
#         b_factors=plddt_b_factors,
#         remove_leading_feature_dimension=False,
#         remark=remark,
#         parents=template_domain_names,
#         parents_chain_index=template_chain_index,
#     )

#     return unrelaxed_protein


# def prep_protein( outputs: Dict, feature_dict: Dict,
# 					feature_processor:feature_pipeline.FeaturePipeline ):
# 	"""
# 	Convert the predicted protein structure into a Protein object.
# 	Need to remove the batch dim.
# 	"""
# 	out = {}
# 	for k in outputs:
# 		if isinstance( outputs[k], dict ):
# 			if k not in out:
# 				out[k] = {}
# 			for m in outputs[k]:
# 				out[k][m] = outputs[k][m].squeeze( 0 ).detach().cpu().numpy()
# 		else:
# 			out[k] = outputs[k].squeeze( 0 ).detach().cpu().numpy()

# 	unrelaxed_protein = prep_output(
# 		out,                     # out,
# 		feature_dict,       # batch,
# 		feature_dict,       # feature_dict,
# 		feature_processor,  # feature_processor
# 		config_preset = None,
# 		#multimer_ri_gap = 1,
# 		subtract_plddt = True # Save b-factor instead of pLDDT (for Molprobity).
# 	)

# 	return unrelaxed_protein


################### Biopython MMCIFDict Parser ###################
##--------------------------------------------------------------##
class MmcifDictParser():
	"""
	Parse the .cif file as an MMCIFDict using Biopython.
	"""
	def __init__( self, cif_file: str ):
		self.cif_file = cif_file
		self.polymer_fields = ["asym_id", "entity_id",
								"seq_id",
								"pdb_seq_num",
								"auth_seq_num",
								"pdb_strand_id"]

		self.seqres_dict = {}

		self.mmcif_dict = self.parse_mmcif_dict()



	def is_cif( self ):
		"""
		Check if the input file is .cif or not.
		"""
		_, ext = os.path.splitest( self.cif_file )
		if "cif" not in ext:
			raise ValueError( "Incorrect sile type specified " +
						f"{self.cif_file}. Only .cif file supported..." )


	def parse_mmcif_dict( self ):
		"""
		Read the .cif file as an MMCIF Dict.
		"""
		mmcif_dict = MMCIF2Dict( self.cif_file )
		return mmcif_dict


	def get_resolution( self ):
		"""
		Extract the resolution of the structure.
			"_reflns.d_resolution_high" field.
		Some PDB entries can have >1 resolution values (8w6x).
			Just taking the first in such cases.
		"""
		# For X-ray structures.
		if "_reflns.d_resolution_high" in self.mmcif_dict:
			if "?" in self.mmcif_dict["_reflns.d_resolution_high"]:
				# e.g. 1osp
				resolution = self.mmcif_dict["_refine.ls_d_res_high"]
			else:
				resolution = self.mmcif_dict["_reflns.d_resolution_high"]
		# For EM sreuctures.
		elif "_em_3d_reconstruction.resolution" in self.mmcif_dict:
			resolution = self.mmcif_dict["_em_3d_reconstruction.resolution"]
			if "?" in resolution:
				print( self.cif_file, " <--" )
		elif "NMR" in self.mmcif_dict["_exptl.method"]:
			resolution = [0]
		else:
			resolution = [0]
		# Temporary
		# if len( resolution ) > 1:
		# 	raise Exception( f"Multiple values for resolution found in {self.cif_file}..." )
		return round( float( resolution[0] ), 2 )


	def get_polymer_entity_ids( self ):
		"""
		Return all the polymer entity_id's.
		Note: this will include protein/dna/rna.
		"""
		return self.mmcif_dict["_entity_poly.entity_id"]


	def get_polymer_entity_types( self ):
		"""
		Return the entity type for all polymer entities.
		"""
		return self.mmcif_dict["_entity_poly.type"]


	def get_protein_entity_ids( self ):
		"""
		Identify entity_id's for proteins.
			entity_type -> Polypeptide(L)
		"""
		entity_ids = self.get_polymer_entity_ids()
		entity_type = self.get_polymer_entity_types()

		prot_entity_ids = [
			entity_ids[i] for i in range( len( entity_ids ) ) if "peptide" in entity_type[i]
		]
		all_protein = len( entity_ids ) == len( prot_entity_ids )
		return prot_entity_ids, all_protein


	def get_protein_asym_ids( self ):
		"""
		Return sym_ids corresponding to protein entities.
		"""
		prot_entity_ids, _ = self.get_protein_entity_ids()
		prot_asym_ids = {
			asym_id
			for asym_id, entity_id in zip(
				self.mmcif_dict["_struct_asym.id"],
				self.mmcif_dict["_struct_asym.entity_id"],
			)
			if entity_id in prot_entity_ids
		}
		return prot_asym_ids


	def get_all_polymer_fields( self ):
		"""
		Extract the following fields from the MCIF Dict:
			asym_id -> PDB assigned chain ID.
			entity_id -> PDB assigned ID for an entity.
			(removed) seq_id -> residue no. as per the SEQRES.
			mon_id -> 3-letter amino acid symbol.
			pdb_seq_num -> PDB assigned residue no.
			auth_seq_num -> author assigned residue no
							(may or may not be the Uniprot residue no.).
			pdb_strand_id -> author assigned chain ID.
		Convert to np.array.
		Also add a binary field (1/0) indicating resolved/missing residue.
			This can be done using the suth_seq_num field.
		"""
		for field in self.polymer_fields:
			poly_key = f"_pdbx_poly_seq_scheme.{field}"
			self.seqres_dict[field] = np.array( self.mmcif_dict[poly_key] )


		mon_id_key = f"_pdbx_poly_seq_scheme.mon_id"
		self.seqres_dict["mon_id"] = np.array(
			[aa_3_to_1( aa ) for aa in self.mmcif_dict[mon_id_key]]
			)


	def get_protein_entity_details( self ):
		"""
		Return seqres_dict for only protein entities.
		"""
		self.get_all_polymer_fields()
		prot_entity_ids, all_protein = self.get_protein_entity_ids()

		if all_protein:
			return self.seqres_dict
		else:
			prot_seqres_dict = {}
			prot_indices = []

			prot_indices = np.isin( self.seqres_dict["entity_id"], prot_entity_ids )

			for field in self.seqres_dict:
				if len( prot_indices ) != len( self.seqres_dict[field] ):
					raise ValueError( f"Incorrect entity_id mask..." )
				prot_seqres_dict[field] = self.seqres_dict[field][prot_indices]
			return prot_seqres_dict


	def get_chain_mapping( self ):
		"""
		Create a mapping between the auth_asym_id and asym_id.
		We assume that each asym_id uniquely maps to a auth_asym_id.
		"""
		asym_id = self.mmcif_dict["_atom_site.label_asym_id"]
		auth_asym_id = self.mmcif_dict["_atom_site.auth_asym_id"]

		asym_to_auth = {}
		auth_to_asym = {}

		prot_asym_ids = self.get_protein_asym_ids()

		for asym, auth in zip( asym_id, auth_asym_id ):
			if asym not in prot_asym_ids:
				continue
			if asym in asym_to_auth and asym_to_auth[asym] != auth:
				raise ValueError(
					f"label_asym_id '{asym}' maps to multiple auth_asym_id values."
				)

			if auth in auth_to_asym and auth_to_asym[auth] != asym:
				raise ValueError(
					f"auth_asym_id '{auth}' maps to multiple label_asym_id values."
				)

			asym_to_auth[asym] = auth
			auth_to_asym[auth] = asym

		chain_mapping = {
			"asym": asym_to_auth,
			"auth_asym": auth_to_asym,
		}

		return chain_mapping

#################### Biopython PDB/CIF Parser ####################
##--------------------------------------------------------------##
class ModelSelect(Select):
	"""
    Selects a specific model from a structure when writing a PDB file.

    This class subclasses Bio.PDB.Select and overrides the "accept_model"
		method to allow only the specified model to be written to the output
		structure
	"""
	def __init__( self, model_id: int, chain_id: str ):
		self.model_id = model_id
		self.chain_id = chain_id

	def accept_model( self, model ):
		return model.id == self.model_id

	def accept_chain( self, chain ):
		return chain.get_id() == self.chain_id

	def accept_residue( self, residue ):
		hetfield, resseq, icode = residue.id

		# Remove HETATM
		# Only standard polymer residues with positive numbering
		return hetfield == " " and resseq > 0


class ChainSelect( Select ):
	"""
    Selects a specific chain from a structure when writing a PDB file.

    This class subclasses Bio.PDB.Select and overrides the "accept_chain"
		method to allow only the specified chain to be written to the output
		structure
	"""
	def __init__(
		self,
		chain_id: str
		# remove_tags: bool = False
	):
		self.chain_id = chain_id
	
	def accept_chain( self, chain ):
		return chain.get_id() == self.chain_id

	def accept_residue( self, residue ):
		hetfield, resseq, icode = residue.id

		# Remove HETATM
		# Only standard polymer residues with positive numbering
		return hetfield == " " and resseq > 0


class Parser():
	"""
	A parser class to read from the simulation output file.
	As we are using PDB format only, CIF compatibility is not required for now.
	"""
	def __init__( self, pdb_file: str ):
		self.pdb_file = pdb_file

		self._ensure_file_exists()
		# Biopython Structure object.
		self.structure = self.get_structure()


	def _ensure_file_exists( self ):
		"""
		Raises an error if the input pdb_file path does not exist.
		"""
		if not os.path.isfile( self.pdb_file ):
			raise FileNotFoundError( f"{self.pdb_file} does not exist or is not a file..." )


	def get_model_ids( self ) -> List:
		"""
		Get a list of all model IDs.
		"""
		return [model.id for model in self.structure]


	def get_parser( self ) -> Bio.PDB.PDBParser:
		"""
		Get the required parser (PDB/CIF) for the input file.
		"""
		ext = os.path.splitext( self.pdb_file )[1]

		if "pdb" in ext:
			parser = PDBParser()
		elif "cif" in ext:
			parser = MMCIFParser()
		else:
			raise ValueError( "Incorrect file format.. Only .pdb/.cif format supported for now..." )

		return parser


	def get_structure( self ) -> Structure.Structure:
		"""
		Return the Biopython Structure object.
		"""
		basename = os.path.basename( self.pdb_file )
		p = self.get_parser()
		structure = p.get_structure( basename, self.pdb_file )

		return structure


	def get_models( self ) -> Iterator[Model.Model]:
		"""
		Yield models in the structure.
		"""
		for model in self.structure:
			yield model


	def get_chains( self, model: Model.Model ):
		"""
		A generator that yields all Chain objects in a Model object.
		"""
		for chain in model:
			yield chain


	def get_residues( self, chain ) -> Iterator[Tuple[Residue.Residue, str]]:
		"""
		A generator that yields all Residue objects for a Chain object.
		"""
		# for chain in model:
		chain_id = chain.id[0]
		for residue in chain:
			yield residue, chain_id


	def get_residues_from_model( self, model: Model.Model
								)-> Iterator[Tuple[Residue.Residue, str]]:
		"""
		A generator object that yields the Residue object, Chain ID
			for all residues in a Model object.
		"""
		for chain in self.get_chains( model ):
			for residue, chain_id in self.get_residues( chain ):
				yield residue, chain_id


	def extract_perresidue_quantity( self, residue: Residue, quantity: str ):
		"""
		Given the Biopython residue object, return the specified quantity:
			1. residue position
			2. Ca-coordinate
		"""
		rep_atom = "CA"

		if quantity == "res_pos":
			quantity = residue.id[1]

		elif quantity == "res_name":
			resname = residue.get_resname()
			quantity = aa_3_to_1( resname )

		elif quantity == "coords":
			coords = residue[rep_atom].coord
			quantity = coords

		elif quantity == "plddt":
			coords = residue[rep_atom].bfactor
			quantity = coords

		else:
			raise ValueError( f"Specified quantity: {quantity} does not exist..." )

		return quantity


	def get_coordinates( self, model: Model.Model ) -> np.array:
		"""
		Extract coordinates for a model in the structure.
		"""
		# for model in self.get_models():
		coords_dict = {}

		for residue, chain_id in self.get_residues_from_model( model ):
			coords = self.extract_perresidue_quantity( residue, "coords" )

			if chain_id not in coords_dict:
				coords_dict[chain_id] = np.array( coords )
			else:
				coords_dict[chain_id] = np.append( coords_dict[chain_id], coords )

		coords_dict = {k: v.reshape( -1, 3 ) for k, v in coords_dict.items()}

		return coords_dict


def remap_chains_pdb(
	struct_file: str,
	map_dict: Dict[str, str],
	remapped_file: str = None ):
	"""
	Rename all chains in the given .pdb file.
	"""
	base, ext = os.path.splitext( struct_file )
	if remapped_file is None:
		remapped_file = base + "_remapped" + ext
	else:
		pass

	if "pdb" in ext:
		io = PDBIO()
	else:
		raise ValueError( f"Incorrect file format: {ext}. required .pdb..." )

	# Thie returns an iterable starting from A.
	if map_dict is None:
		new_ids = iter( string.ascii_uppercase )
	else:
		new_ids = iter( map_dict.values() )

	structure = Parser( struct_file ).structure
	for model in structure:
		for chain in model:
			chain.id = next( new_ids )

	io = PDBIO()
	io.set_structure( structure )
	io.save( remapped_file )


def remap_chains_cif(
	struct_file: str,
	map_dict: Dict[str, str],
	remapped_file: str = None,
	use_native_chains: bool = False
	):
	"""
	Map all chains in the given structure (.cif file) as specified in the map_dict.
	e.g. For mapping chains from [H, L, C] to [A, B, C],
	The map dict is assumed to contain the auth asym_ids.
	map_dict: {
		"H": "A", "L": "B", "C": "C"
	}
	Modify both the label_asym_id and auth_asym_id.

	Inputs:
	----------
	struct_file: path to the structure file for ehich the chain
		IDs are to be remapped.
	map_dict: dict containing native to system chain mapping.
	remapped_file: file path for the structure file with chain
		IDs remapped.
	use_native_chains: if True, uses the native chain ID mapping in the given map_dict.
		If False, redefines the map_dict using the auth_asym_ids in the input .cif file.
	"""
	base, ext = os.path.splitext(struct_file)
	if remapped_file is None:
		remapped_file = base + "_remapped" + ext

	if "cif" not in ext:
		raise ValueError( f"Incorrect file format: {ext}. Required .cif..." )

	mmcif_dict = MmcifDictParser( struct_file ).mmcif_dict

	if not use_native_chains:
		original_chains = set(mmcif_dict["_atom_site.auth_asym_id"])
		sys_chains = list( map_dict.values() )
		map_dict = dict( zip( original_chains, sys_chains ) )

	# map asym_id to auth_asym_id..
	# Raise an error if an asym_id maps to multiple auth_asym_ids.
	asym_to_auth = {}
	for label, auth in zip( mmcif_dict["_atom_site.label_asym_id"],
							mmcif_dict["_atom_site.auth_asym_id"] ):
		if label in asym_to_auth and asym_to_auth[label] != auth:
			raise ValueError(
				f"label_asym_id '{label}' maps to multiple auth_asym_ids"
				)
		asym_to_auth[label] = auth

	# Remap auth_asym_id directly.
	mmcif_dict["_atom_site.auth_asym_id"] = [
		map_dict.get(chain_id, chain_id)
		for chain_id in mmcif_dict["_atom_site.auth_asym_id"]
	]

	# Remap label_asym_id via its auth equivalent.
	mmcif_dict["_atom_site.label_asym_id"] = [
		map_dict.get( asym_to_auth.get( chain_id, chain_id ),
						asym_to_auth.get( chain_id, chain_id ) )
		for chain_id in mmcif_dict["_atom_site.label_asym_id"]
	]

	# Keep _struct_asym consistent.
	if "_struct_asym.id" in mmcif_dict:
		mmcif_dict["_struct_asym.id"] = [
			map_dict.get( chain_id, chain_id )
			for chain_id in mmcif_dict["_struct_asym.id"]
		]

	io = MMCIFIO()
	io.set_dict( mmcif_dict )
	io.save( remapped_file )


def extract_chains(
	struct_file: str,
	chain_id: str,
	ext: str,
	output_file: str
):
	"""
	Extract all the specified chains from the given structure file
		(pdb/cif) and save as a new file.

	Inputs:
	----------
	struct_file: path to the structure file for ehich the chain
		IDs are to be remapped.
	chain_id: chain identifier (auth asym ID) for the chain to be extracted.
	ext: must be either of cif or pdb.
	output_file: file path for the structure file with chain
		IDs remapped.
	"""
	if ext not in ["cif", "pdb"]:
		raise ValueError( f"Unsupported file format. Supported only pdb/cif... " )

	structure = Parser( struct_file ).structure

	chain_selector = ChainSelect( chain_id = chain_id )

	if ext == "pdb":
		io = PDBIO()
	else:
		io = MMCIFIO()
	io.set_structure( structure )
	io.save( output_file, chain_selector )

