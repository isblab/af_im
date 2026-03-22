"""
This script contains functions to read/write modify PDB files.
"""
import io
import string
import os
import warnings
from typing import List, Dict, Tuple, Iterator
import numpy as np
from scipy.spatial import distance_matrix

#import gemmi
import Bio
from Bio.PDB import (
	PDBParser, MMCIFParser,
	MMCIFIO, PDBIO,
	Select,
	Structure, Model, Residue )
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import freesasa
import modelcif
import modelcif.model
import modelcif.dumper
import modelcif.reference
import modelcif.protocol
import modelcif.alignment
import modelcif.qa_metric

# from openfold.utils.script_utils import prep_output
from openfold.np.protein import Protein, get_pdb_headers, _chain_end
from openfold.np import residue_constants
from openfold.data import feature_pipeline
from openfold.np import protein

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
	Get a chain ID based on an index.
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
def prep_output( out, batch, feature_dict, feature_processor, config_preset, subtract_plddt ):
    plddt = out["plddt"]

    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if subtract_plddt:
        plddt_b_factors = 100 - plddt_b_factors

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    if feature_processor.config.common.use_templates and "template_domain_names" in feature_dict:
        template_domain_names = [
            t.decode("utf-8") for t in feature_dict["template_domain_names"]
        ]

        # This works because templates are not shuffled during inference
        template_domain_names = template_domain_names[
                                :feature_processor.config.predict.max_templates
                                ]

        if "template_chain_index" in feature_dict:
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[
                                   :feature_processor.config.predict.max_templates
                                   ]

    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"max_templates={feature_processor.config.predict.max_templates}",
        f"config_preset={config_preset}",
    ])

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein


def prep_protein( outputs: Dict, feature_dict: Dict,
					feature_processor:feature_pipeline.FeaturePipeline ):
	"""
	Convert the predicted protein structure into a Protein object.
	Need to remove the batch dim.
	"""
	out = {}
	for k in outputs:
		if isinstance( outputs[k], dict ):
			if k not in out:
				out[k] = {}
			for m in outputs[k]:
				out[k][m] = outputs[k][m].squeeze( 0 ).detach().cpu().numpy()
		else:
			out[k] = outputs[k].squeeze( 0 ).detach().cpu().numpy()

	unrelaxed_protein = prep_output(
		out,                     # out,
		feature_dict,       # batch,
		feature_dict,       # feature_dict,
		feature_processor,  # feature_processor
		config_preset = None,
		#multimer_ri_gap = 1,
		subtract_plddt = True # Save b-factor instead of pLDDT (for Molprobity).
	)

	return unrelaxed_protein


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
			# for entity_id in prot_entity_ids:
			# 	prot_indices = np.append(
			# 		prot_indices, self.seqres_dict["entity_id"] == entity_id
			# 		)
			# print( len( prot_indices ) )
			prot_indices = np.isin( self.seqres_dict["entity_id"], prot_entity_ids )
			# print( len( prot_indices ) )

			for field in self.seqres_dict:
				if len( prot_indices ) != len( self.seqres_dict[field] ):
					raise ValueError( f"Incorrect entity_id mask..." )
				# try:
				prot_seqres_dict[field] = self.seqres_dict[field][prot_indices]
				# except:
				# 	print( prot_entity_ids )
				# 	print( field )
				# 	print( prot_indices )
				# 	print( self.cif_file )
			return prot_seqres_dict


#################### Biopython PDB/CIF Parser ####################
##--------------------------------------------------------------##
class ChainSelect( Select ):
	"""
    Selects a specific chain from a structure when writing a PDB file.

    This class subclasses Bio.PDB.Select and overrides the "accept_chain"
		method to allow only the specified chain to be written to the output
		structure
	"""
	def __init__( self, chain_id: str ):
		self.chain_id = chain_id
	
	def accept_chain( self, chain ):
		return chain.get_id() == self.chain_id


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
	new_ids = iter( string.ascii_uppercase )

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
	remapped_file: str = None ):
	"""
	Map all chains in the given structure (.cif file) as specified in the map_dict.
	e.g. For mapping chains from [H, L, C] to [A, B, C],
	The map dict is assumed to contain the auth asym_ids.
	map_dict: {
		"H": "A", "L": "B", "C": "C"
	}
	Modify both the label_asym_id and auth_asym_id.
	"""
	base, ext = os.path.splitext( struct_file )
	if remapped_file is None:
		remapped_file = base + "_remapped" + ext
	else:
		pass

	if "cif" in ext:
		io = MMCIFIO()
	else:
		raise ValueError( f"Incorrect file format: {ext}. required .cif..." )

	mmcif_dict = MmcifDictParser( struct_file ).mmcif_dict
	asym_to_auth = dict(
		zip( mmcif_dict["_atom_site.label_asym_id"], mmcif_dict["_atom_site.auth_asym_id"] )
	)

	mmcif_dict["_atom_site.auth_asym_id"] = [
		map_dict.get(chain_id, chain_id)
		for chain_id in mmcif_dict["_atom_site.auth_asym_id"]
	]
	mmcif_dict["_atom_site.label_asym_id"] = [
		map_dict.get(
			asym_to_auth[chain_id], asym_to_auth[chain_id] )
		for chain_id in mmcif_dict["_atom_site.label_asym_id"]
	]

	io = MMCIFIO()
	io.set_dict( mmcif_dict )
	io.save( remapped_file )


############################## SASA ##############################
##--------------------------------------------------------------##
class SolventAccessibleSurfaceArea():
	"""
	Obtain surface exposed residues for each chain in a complex.
	"""
	def __init__( self,
		pdb_file: str,
		chain_file_prefix: str,
		tmp_dir_path: str,
		calc_rsa: bool = True,
		calc_sasa: bool = False
		):
		self.pdb_file = pdb_file
		# Prefix for the file name of a chain from the complex.
		self.chain_file_prefix = chain_file_prefix
		# If True, compute RSA.
		self.calc_rsa = calc_rsa
		# If true SASA.
		self.calc_sasa = calc_sasa
		# Temp dir to store the struct of the chains.
		self.tmp_dir_path = tmp_dir_path

		self.structure = None


	def forward( self ) -> Dict[str, Dict[int, float]]:
		"""
		Create a temp dir.
		Get chain IDs from the input pdb file.
		Split the complex into monomers.
		Compute SASA for each chain.
		"""
		self.create_tmp_dir()

		self.init_structure()
		if self.calc_sasa and self.calc_rsa:
			raise ValueError( "Can only compute one of RSA or SASA..." )
		if self.calc_sasa:
			surface_area = self.compute_sasa()
		elif self.calc_rsa:
			surface_area = self.compute_rsa()
		else:
			raise ValueError( f"At least one of calc_rsa or calc_sasa must be True..." )

		self.remove_tmp_dir()
		return surface_area


	def create_tmp_dir( self ):
		"""
		Create a temporary dir to store structure for the
			chains in a complex.
		"""
		os.makedirs( self.tmp_dir_path, exist_ok = True )


	def remove_tmp_dir( self ):
		"""
		Remove the temporary dir.
		"""
		remove_dir( dir_path = self.tmp_dir_path )


	def init_structure( self ):
		"""
		Initialize the BioPython Structure object.
		"""
		parser = Parser( pdb_file = self.pdb_file )
		self.structure = parser.get_structure()


	def get_chain_ids( self ):
		"""
		Return the chain IDs from the structure.
		"""
		chain_ids = []
		# Only parse the 1st model in case of multi-model input.
		model = self.structure[0]
		for chain in model:
			chain_ids.append( chain.id )
		return chain_ids


	def split_complex_to_monomers( self ) -> Dict[str, str]:
		"""
		Split a multi-chain structure into individual chains and
			temporarily save on disk.
		Returns a dict that maps the chain to the corresponding file path.
		"""
		chain_to_file = {}
		chain_ids = self.get_chain_ids()

		for chain_id in chain_ids:
			file = os.path.join( self.tmp_dir_path, f"{self.chain_file_prefix}_{chain_id}.pdb" )
			chain_to_file[chain_id] = file
			io = PDBIO()
			io.set_structure( self.structure )
			io.save( file, ChainSelect( chain_id = chain_id ) )
		return chain_to_file


	def compute_sasa( self ) -> Dict[str, Dict[int, float]]:
		"""
		Use fresasa to compute the per-residue SASA for each chain in
			the complex.
		The complex must first be split into individual chains to obtain
			the SASA for each chain in isolation.
		freesasa computes per-atom surface area.
			For per-residue surface area we sum the area for all atoms in a residue.
				MDTraj does the same
					(https://www.mdtraj.org/1.9.5/examples/solvent-accessible-surface-area.html).
			"""
		chain_to_file = self.split_complex_to_monomers()
		sasa = {}

		for chain_id, file in chain_to_file.items():
			sasa[chain_id] = {}

			struct = freesasa.Structure( file )
			result = freesasa.calc( struct )
			chain_sasa = {}
			for i in range( struct.nAtoms() ):
				resi = struct.residueNumber( i )
				c_id = struct.chainLabel( i )
				area = result.atomArea( i )

				if c_id !=  chain_id:
					raise ValueError( f"Mismatched chain ID: {c_id} != {chain_id}. " +
						"freesasa derived chain ID does not match that from the input struct file..."
					)
				res = int( resi )
				chain_sasa[res] = chain_sasa.get( res, 0.0 ) + area

			sasa[chain_id] = chain_sasa
		return sasa


	def compute_rsa( self ) -> Dict[str, Dict[int, float]]:
		"""
		Use fresasa to compute the per-residue RSA for each chain in
			the complex.
		The complex must first be split into individual chains to obtain
			the SASA for each chain in isolation.
			"""
		chain_to_file = self.split_complex_to_monomers()
		rsa = {}

		for chain_id, file in chain_to_file.items():
			rsa[chain_id] = {}

			struct = freesasa.Structure( file )
			result = freesasa.calc( struct )

			residue_areas = result.residueAreas()
			for chain in residue_areas:
				# if c_id !=  chain_id:
				# 	raise ValueError( f"Mismatched chain ID: {c_id} != {chain_id}. " +
				# 		"freesasa derived chain ID does not match that from the input struct file..."
				# 	)

				chain_rsa = {}
				for resi in residue_areas[chain]:
					residue = residue_areas[chain][resi]
					rsa_total = residue.relativeTotal
					res = int( resi )
					chain_rsa[res] = rsa_total*100

			rsa[chain_id] = chain_rsa
		return rsa

################### AF2 module to save PDB/CIF ###################
##--------------------------------------------------------------##
class SaveModels():
	"""
	A class to save predicted structures as models to a PDB/CIF file.
	Only supporting PDB for now.
	"""
	def __init__( self,
			title: str,
			output_format: str,
			ensemble_dir: str,
			save_single_model: bool = True ):
		self.title = title
		self.entities_map = {}
		self.asym_unit_map = {}
		self.output_format = output_format
		self.ensemble_dir = ensemble_dir
		self.save_single_model = save_single_model
		# self.output_path = output_path

		if self.output_format not in ["pdb", "cif"]:
			raise ValueError( "Invalid output format specified. Use 'pdb' or 'cif'... " )

		# If save_single_models is True, ensemble_dir must exist.
		if self.save_single_model:
			if not os.path.exists( self.ensemble_dir ):
				raise ValueError( f"Ensemble dir = {self.ensemble_dir} " +
					"must exist if saving single models..." )


	def initialize_system( self ):
		"""
		Instantiate a modelcif.System object.
			Top-level class representing a complete modeled system
		"""
		if self.output_format == "pdb":
			self.system = []
		elif self.output_format == "cif":
			self.system = self.create_system()
			self.model_group = self.create_model_group()
			# Add model_group to system.
			self.system.model_groups.append( self.model_group )
		else:
			raise ValueError( f"Incorrect file format provided {self.output_format}..." )


	def create_system( self ):
		"""
		Instantiate a modelCIF.System object to which 
			all predicted structures will be added.
		"""
		system = modelcif.System( title = self.title )
		return system


	def create_model_group( self ):
		"""
		Instantiate an empty modelcif.model_group object.
		All models in the system will be appended to the same model group.
		"""
		model_group = modelcif.model.ModelGroup([], name = "Trajectory" )
		return model_group


	def create_attributes( self, prot: Protein ):
		"""
		Create the required attributes form the Protein object.
		Add entities and asym units to the system for all models.
		"""
		self.get_protein_attributes( prot )
		if self.output_format == "cif":
			self.create_entities()
			self.create_asym_units()



	def get_protein_attributes( self, prot: Protein ):
		"""
		Obtain the required attributes from the Protein object.
		"""
		self.restypes = residue_constants.restypes + ["X"]
		self.atom_types = residue_constants.atom_types

		self.atom_mask = prot.atom_mask
		self.aatype = prot.aatype
		self.atom_positions = prot.atom_positions
		self.residue_index = prot.residue_index.astype(np.int32)
		self.b_factors = prot.b_factors
		self.chain_index = prot.chain_index
		self.n = self.aatype.shape[0]

		if self.chain_index is None:
			self.chain_index = [0 for i in range( self.n )]



	def create_entities( self ):
		"""
		Select unique sequences and add them as entities across all models in the system.
		"""
		# system = modelcif.System( title = f"Epoch {epoch}" )

		# Finding chains and creating entities
		seqs = {}
		seq = []
		last_chain_idx = None

		for i in range( self.n ):
			if last_chain_idx is not None and last_chain_idx != self.chain_index[i]:
				seqs[last_chain_idx] = seq
				seq = []
			seq.append(self.restypes[self.aatype[i]])
			last_chain_idx = self.chain_index[i]
		# finally add the last chain
		seqs[last_chain_idx] = seq

		# Now reduce sequences to unique ones.
		# 	(note this won't work if different asyms have different unmodelled regions)
		unique_seqs = {}
		for chain_idx, seq_list in seqs.items():
			seq = "".join(seq_list)
			if seq in unique_seqs:
				unique_seqs[seq].append(chain_idx)
			else:
				unique_seqs[seq] = [chain_idx]

		# adding 1 entity per unique sequence
		# entities_map = {}
		for key, value in unique_seqs.items():
			# model_e = modelcif.Entity( key, description = f"Model subunit" )
			if key not in self.entities_map:
				model_e = modelcif.Entity( key, description = "Model subunit" )
				for chain_idx in value:
					self.entities_map[chain_idx] = model_e



	def create_asym_units( self ):
		"""
		Create a asym units for all entities.
		"""

		chain_tags = string.ascii_uppercase
		# asym_unit_map = {}
		for chain_idx in set( self.chain_index ):
			if chain_idx not in self.asym_unit_map:
				# Define the model assembly
				chain_id = chain_tags[chain_idx]
				asym = modelcif.AsymUnit(
						# self.entities_map[chain_idx], details = f"Model subunit {chain_id}", id = chain_id # - Kartik -
						self.entities_map[chain_idx], details='Model subunit %s' % chain_id, id=chain_id
						)
				self.asym_unit_map[chain_idx] = asym
		# modeled_assembly = modelcif.Assembly( self.asym_unit_map.values(), name = f"Modeled assembly {model_index}" ) # - Kartik -
		self.modeled_assembly = modelcif.Assembly(self.asym_unit_map.values(), name='Modeled assembly')


	def add_model( self, prot: Protein, model_id: int ):
		"""
		For the 1st model:
			Create all required attributes and add to model.
		For others, just add to model.
		Also, save each model on disk.
		Using the epoch no. as model_id.
		"""
		if  self.output_format == "pdb":
			if model_id == 0:
				headers = get_pdb_headers(prot)
				if len(headers) > 0:
					self.system.extend(headers)

		self.create_attributes( prot )

		if self.output_format == "pdb":
			system = self.add_to_pdb( prot = prot, model_id = model_id  )
			self.system.extend( system )
		elif self.output_format == "cif":
			model = self.add_to_modelcif( model_id = model_id )
			self.model_group.append( model )
			system = model

		if self.save_single_model:
			self.save( system,
						os.path.join( self.ensemble_dir, f"model_{model_id}" )
						 )



	def add_to_pdb( self, prot: Protein, model_id: int ):
		"""
		Taken from openfold.np.protein.py
		- Kartik - Modified to write multiple models in a PDB file format.

		Converts a `Protein` instance to a PDB string.

		Args:
		  prot: The protein to convert to PDB.

		Returns:
		  PDB string.
		"""
		system = []
		# - Kartik - Using the epoch as Model index.
		model_index = model_id
		# restypes = residue_constants.restypes + ["X"]
		res_1to3 = lambda r: residue_constants.restype_1to3.get(self.restypes[r], "UNK")
		# atom_types = residue_constants.atom_types

		# For uniformity, pdblines is replaced to self.system.
		# pdb_lines = []

		# atom_mask = prot.atom_mask
		# aatype = prot.aatype
		# atom_positions = prot.atom_positions
		# residue_index = prot.residue_index.astype(np.int32)
		# b_factors = prot.b_factors
		# chain_index = prot.chain_index.astype(np.int32)

		if np.any( self.aatype > residue_constants.restype_num ):
			raise ValueError("Invalid aatypes.")

		# Construct a mapping from chain integer indices to chain ID strings.
		chain_ids = {}
		for i in np.unique( self.chain_index ): # np.unique gives sorted output.
			if i >= PDB_MAX_CHAINS:
				raise ValueError(
					f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
				)
			chain_ids[i] = PDB_CHAIN_IDS[i]

		# headers = get_pdb_headers(prot)
		# if (len(headers) > 0):
		#     # pdb_lines.extend(headers)
		#     self.system.extend(headers)

		# pdb_lines.append("MODEL     1")
		system.append( f"MODEL     {model_index}" )
		# n = aatype.shape[0]
		atom_index = 1
		last_chain_index = self.chain_index[0]
		prev_chain_index = 0
		chain_tags = string.ascii_uppercase

		# Add all atom sites.
		for i in range( self.aatype.shape[0] ):
			# Close the previous chain if in a multichain PDB.
			if last_chain_index != self.chain_index[i]:
				# pdb_lines.append
				system.append(
					_chain_end(
						atom_index,
						res_1to3( self.aatype[i - 1] ),
						chain_ids[self.chain_index[i - 1]],
						self.residue_index[i - 1]
					)
				)
				last_chain_index = self.chain_index[i]
				atom_index += 1 # Atom index increases at the TER symbol.

			res_name_3 = res_1to3( self.aatype[i] )
			for atom_name, pos, mask, b_factor in zip(
				self.atom_types, self.atom_positions[i], self.atom_mask[i], self.b_factors[i]
			):
				if mask < 0.5:
					continue

				record_type = "ATOM"
				name = atom_name if len(atom_name) == 4 else f" {atom_name}"
				alt_loc = ""
				insertion_code = ""
				occupancy = 1.00
				element = atom_name[
					0
				]  # Protein supports only C, N, O, S, this works.
				charge = ""

				chain_tag = "A"
				if self.chain_index is not None:
					chain_tag = chain_tags[self.chain_index[i]]

				# PDB is a columnar format, every space matters here!
				atom_line = (
					f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
					#TODO: check this refactor, chose main branch version
					#f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
					f"{res_name_3:>3} {chain_tag:>1}" # main branch version
					f"{self.residue_index[i]:>4}{insertion_code:>1}   "
					f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
					f"{occupancy:>6.2f}{b_factor:>6.2f}          "
					f"{element:>2}{charge:>2}"
				)
				# pdb_lines.append(atom_line)
				system.append( atom_line )
				atom_index += 1

			should_terminate = i == self.n - 1
			if( self.chain_index is not None ):
				if i != self.n - 1 and self.chain_index[i + 1] != prev_chain_index:
					should_terminate = True
					prev_chain_index = self.chain_index[i + 1]

			if should_terminate:
				# Close the chain.
				chain_end = "TER"
				chain_termination_line = (
					f"{chain_end:<6}{atom_index:>5}      "
					f"{res_1to3( self.aatype[i]):>3} "
					f"{chain_tag:>1}{self.residue_index[i]:>4}"
				)
				# pdb_lines.append(chain_termination_line)
				system.append( chain_termination_line )
				# atom_index += 1 # I believe this line is a big in OpenFold implementation. - Kartik -
				# This will add an offset of 1 atom after every chain. - Kartik -

				# I don't need it after every chain. - Kartik -
				# if(i != self.n - 1):
					# "prev" is a misnomer here. This happens at the beginning of
					# each new chain.
					# pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))
					# self.system.extend( get_pdb_headers( prot, prev_chain_index ) )

		# pdb_lines.append("ENDMDL")
		# pdb_lines.append("END")
		system.append("ENDMDL")

		# Pad all lines to 80 characters
		# pdb_lines = [line.ljust(80) for line in pdb_lines]
		# return '\n'.join(pdb_lines) + '\n' # Add terminating newline.
		return system



	def add_to_modelcif( self, model_id: int ):
		"""
		Taken from openfold.np.protein.py
		- Kartik - modified this function to allow writing multiple models to the same CIF file.
		Instead of returning a ModelCIF string, this function will add a 
			model to a model group and the latter to the system.
		
		Converts a `Protein` instance to a ModelCIF string. Chains with identical modelled coordinates
		will be treated as the same polymer entity. But note that if chains differ in modelled regions,
		no attempt is made at identifying them as a single polymer entity.

		Args:
		  prot: The protein to convert to PDB. (deprecated)

		Returns:
		  ModelCIF object.
		"""
		# - Kartik - Using the epoch as Model index.
		model_index = model_id

		class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
			name = "pLDDT"
			software = None
			description = "Predicted lddt"

		class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
			name = "pLDDT"
			software = None
			description = "Global pLDDT, mean of per-residue pLDDTs"


		residue_index = self.residue_index
		chain_index = self.chain_index
		atom_types = self.atom_types
		atom_positions = self.atom_positions
		atom_mask = self.atom_mask
		b_factors = self.b_factors
		n = self.n
		asym_unit_map = self.asym_unit_map
		class _MyModel(modelcif.model.AbInitioModel):
			def get_atoms(self):
				# Add all atom sites.
				for i in range( n ):
					for atom_name, pos, mask, b_factor in zip(
							atom_types, atom_positions[i], atom_mask[i], b_factors[i]
					):
						if mask < 0.5:
							continue
						element = atom_name[0]  # Protein supports only C, N, O, S, this works.
						yield modelcif.model.Atom(
							asym_unit = asym_unit_map[chain_index[i]], type_symbol = element,
							seq_id = residue_index[i], atom_id=atom_name,
							x=pos[0], y = pos[1], z=pos[2],
							het = False, biso = b_factor, occupancy = 1.00)

			def add_scores(self):
				# local scores
				plddt_per_residue = {}
				for i in range( n ):
					for mask, b_factor in zip(atom_mask[i], b_factors[i]):
						if mask < 0.5:
							continue
						# add 1 per residue, not 1 per atom
						if chain_index[i] not in plddt_per_residue:
							# first time a chain index is seen: add the key and start the residue dict
							plddt_per_residue[chain_index[i]] = {residue_index[i]: b_factor}
						if residue_index[i] not in plddt_per_residue[chain_index[i]]:
							plddt_per_residue[chain_index[i]][residue_index[i]] = b_factor
				plddts = []
				for chain_idx in plddt_per_residue:
					for residue_idx in plddt_per_residue[chain_idx]:
						plddt = plddt_per_residue[chain_idx][residue_idx]
						plddts.append(plddt)
						self.qa_metrics.append(
							_LocalPLDDT(asym_unit_map[chain_idx].residue(residue_idx), plddt))
				# global score
				self.qa_metrics.append((_GlobalPLDDT(np.mean(plddts))))

		# Add the model and modeling protocol to the file and write them out:
		model = _MyModel( assembly = self.modeled_assembly, name = f"Model {model_index}" ) # - Kartik -
		# model = _MyModel(assembly=modeled_assembly, name='Best scoring model')
		model.add_scores()
		return model

		# self.model_group.append( model )
		# model_group = modelcif.model.ModelGroup([model], name = f"Model {model_index}" )
		# self.system.model_groups.append(model_group)


	# Will do this outside this function to allow writing multiple models to a single file.
	# fh = io.StringIO()
	# modelcif.dumper.write(fh, [system])
	# return fh.getvalue()



	def to_mmcif_string( self ):
		"""
		For writing predicted structures to modelCIF string.
		"""
		fh = io.StringIO()
		modelcif.dumper.write( fh, [self.system] )
		return fh


	def save( self, system: str, output_path: str ):
		"""
		Save the models on disk as per the format.
		"""
		if self.output_format == "pdb":
			# Pad all lines to 80 characters
			system.append("END")
			system = [line.ljust(80) for line in system]
			system = "\n".join( system ) + "\n"
			fp = open_file_handler( f"{output_path}.pdb", 'w' )
			fp.write( system )
		else:
			self.system.model_groups.append( self.model_group )
			fh = self.to_mmcif_string()

			fp = open_file_handler( f"{output_path}.cif", 'w' )
			fp.write( fh.getvalue() )
