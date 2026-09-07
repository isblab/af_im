"""
Creating a multi-state protein benchmark.
We use monomers known to exist in holo and apo conformations taken from
	https://gitlab.com/sbgunq/publications/af2confdiv-oct2021.git
"""
from typing import List, Tuple, Dict, Any
import os, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import PDBIO
from Bio import pairwise2
from multiprocessing import Pool
import tqdm

from data.simulate_data import SimulateCrosslinks
from config import get_config_dict
from utils.utils import write_json
from utils.api_utils import download_pdb, PdbRestApi
from utils.pdb_utils import (
	Parser,
	MmcifDictParser,
	aa_3_to_1,
	ModelSelect,
	ChainSelect
)
from utils.paths import (
	get_meta_dir_path,
	get_benchmark_dir_path,
	get_sys_data_dir_path,
	get_benchmark_csv_file,
	get_sys_config_path
	)
from utils.tools import usalign, get_alignment_score

warnings.filterwarnings( "ignore" )


class ApoHoloStates():
	"""
	Given the structres for the apo and holo states
		of a protein, extract the common (representative)
		sequence from.
	"""
	def __init__(
		self,
		apo_cif_file: str,
		holo_cif_file: str,
		apo_asym_id: str,
		apo_model_id: int,
		holo_asym_id: str,
		holo_model_id: int,
		apo_out_file: str,
		holo_out_file: str
	):
		self.input_dict = {
			# CIF file path for apo/holo states.
			"apo_cif_file": apo_cif_file,
			"holo_cif_file": holo_cif_file,
			# asym_id for apo/holo states.
			"apo_asym_id": apo_asym_id,
			"holo_asym_id": holo_asym_id,
			# Model IDs for the apo/holo states.
			"apo_model_id": apo_model_id,
			"holo_model_id": holo_model_id,
			# File to save the processed apo/holo states.
			"apo_out_file": apo_out_file,
			"holo_out_file": holo_out_file
		}
		self.usalign_script = "USalign"


	def forward( self ):
		"""
		To model the multi-state proteins, we need to find the
			representative sequence.
			The longest stretch of aa common in the structures
				available for both the apo and holo states.
		
		Parse the MMCIF file for apo and holo states and extract,
			SEQRES sequence
			PDB residue numbering
		Obtain the common seq (representative) between the apo and
			holo states.
		Save the selected common sequence as a new PDB file.
			Modify the residue numbering to be 1-indexed for
			both states.
		"""
		state_dict = self.get_state_dict_from_input()

		rep_dict = self.get_representative_seq(
			state_dict = state_dict
		)
		if rep_dict == None:
			pass
		else:
			self.save_rep_seq( rep_dict = rep_dict )
			rmsd, tm = self.compute_tm_score()
			rep_dict["tm"] = tm
			rep_dict["rmsd"] = rmsd
		return rep_dict

	################################################################################
	def get_state_dict_from_input(
		self
	) -> Dict[str, Dict[str, Any]]:
		"""
		Given the input CIF files and chain_id
			for apo and holo states, extract the
			seq and auth residue numbering.

		Returns:
		----------
		state_dict: dict containing the SEARES seq and auth
			residue numbering for the apo and holo states.
			{
				"apo": {
					"seq": str,
					"res_num": np.ndarray
				},
				"holo": {
					"seq": str,
					"res_num": np.ndarray
				}
			}
		"""
		apo_seqres, apo_res_num, apo_missing = self.get_seq_from_cif(
			cif_file = self.input_dict["apo_cif_file"],
			chain_id = self.input_dict["apo_asym_id"]
		)
		holo_seqres, holo_res_num, holo_missing = self.get_seq_from_cif(
			cif_file = self.input_dict["holo_cif_file"],
			chain_id = self.input_dict["holo_asym_id"]
		)
		state_dict = {
			"apo": {
				"seq": apo_seqres,
				"res_num": apo_res_num,
				"missing_mask": apo_missing
			},
			"holo": {
				"seq": holo_seqres,
				"res_num": holo_res_num,
				"missing_mask": holo_missing
			}
		}
		return state_dict

	################################################################################
	def get_asym_auth_asym_map(
		self,
		cif_file: str
	) -> Dict[str, Dict]:
		"""
		Parse the MMCIF dict to create a mapping between the
			asym_id and auth_asym_id.
		"""
		mmcif_dict = MmcifDictParser( cif_file = cif_file )
		asym_to_auth_asym = mmcif_dict.get_chain_mapping()
		return asym_to_auth_asym


	def get_seq_from_cif(
		self,
		cif_file: str,
		chain_id: str,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Given the path to the CIF file, parse and extract the,
			SEQRES sequence
			PDB residue numbering
			Auth residue numbering
			for the specified chain_id.
		The chain_id is assumed to be the asym_id.
		We create a binary mask for the chain of interest which is
			used to extract the other relevant info.
		Remove the residues corresponding to expression tags
			(auth residue numbering <1)
		In the cif file, the pdb_seq_num amd auth_seq_num both are
			alomost identical and correspond to the author provided
			residue numbering.
			They differ in the fact that auth_seq_num contains '?' for
				missing residues.
			e.g. 1VR6
			We use the pdb_seq_num for our purpose.
		_pdbx_poly_seq_scheme is invariant to model_id.

		Inputs:
		----------
		cif_file: path to the CIF file.
		chain_id: asym_id for the chain of interest.

		Returns:
		----------
		chain_seqres: SEQRES seq for the chain of interest.
		chain_res_num: auth residue numbering for the chain
			of interest.
		"""
		mmcif_dict = MmcifDictParser( cif_file = cif_file ).mmcif_dict

		# Create a binary mask for chain of interest.
		label_asym_id = np.array( mmcif_dict["_pdbx_poly_seq_scheme.asym_id"] )
		chain_mask = np.where( label_asym_id == chain_id, 1, 0 )

		# Get the sequence (3-letter aa)
		seqres_seq = mmcif_dict["_pdbx_poly_seq_scheme.mon_id"]
		seqres_seq = np.array(
			[aa_3_to_1( aa ) for aa in seqres_seq]
		)

		# Auth residue numbering
		pdb_res_num = np.array(
			mmcif_dict["_pdbx_poly_seq_scheme.pdb_seq_num"]
			).astype( np.int32 )
		# Auth residue numbering with missing residue annotation.
		auth_res_num = mmcif_dict["_pdbx_poly_seq_scheme.auth_seq_num"]
		auth_res_num = list( map( str, auth_res_num ) )
		auth_res_num = np.array( auth_res_num )

		# Select the seq and residues for the chain of interest.
		chain_seqres = seqres_seq[chain_mask.astype( bool )]
		chain_pdb_res_num = pdb_res_num[chain_mask.astype( bool )]
		chain_auth_res_num = auth_res_num[chain_mask.astype( bool )]

		# Create a mask to exclude residues belonging to expression tags.
		res_mask = np.where( chain_pdb_res_num > 0, 1, 0 )

		# Remove the expression tag.
		chain_seqres = "".join( chain_seqres[res_mask.astype( bool )] )
		chain_pdb_res_num = chain_pdb_res_num[res_mask.astype( bool )]
		chain_auth_res_num = chain_auth_res_num[res_mask.astype( bool )]

		# Create a mask for missing residues.
		missing_mask = np.where( chain_auth_res_num != "?", 1, 0 )

		return chain_seqres, chain_pdb_res_num, missing_mask

	################################################################################
	def get_representative_seq(
		self,
		state_dict: Dict[str, Dict[str, Any]]
	) -> Dict[str, Any]:
		"""
		Given the SEQRES seq and auth residue numbering for
			the apo and holo states,
		Find the coomon sequence between the apo and holo states.
			We use the pairwise sew alignemnet (PSA).
			This works fine as long as the apo and holo states
				belong to the same protein.
			Otherwise we may have removal of residues due
				to gapped alignment.
		Select the residues corresponding to the common seq
			in apo and holo states.
		Remove residues that are missing in either of the
			apo and holo states.
		Create a new 1-indexed residue numbering.

		Inputs:
		----------
		state_dict: dict containing the SEARES seq and auth
			residue numbering for the apo and holo states.

		Returns:
		----------
		rep_dict: dict containing the representative seq
			and the residue umbering for the apo, holo,
			and the representative seq.
			{
				rep_seq: str
				rep_holo_res_num: np.ndarray,
				rep_apo_res_num: np.ndarray,
				seq_id: np.ndarray
			}
		"""
		apo_seqres = state_dict["apo"]["seq"]
		holo_seqres = state_dict["holo"]["seq"]
		apo_res_num = state_dict["apo"]["res_num"]
		holo_res_num = state_dict["holo"]["res_num"]
		# 0 - Missing residues
		apo_missing = state_dict["apo"]["missing_mask"]
		holo_missing = state_dict["holo"]["missing_mask"]

		alignments = pairwise2.align.globalms(
			apo_seqres,
			holo_seqres,
			2,    # match
			-1,   # mismatch
			-5,   # gap open
			-0.5, # gap extend
			penalize_end_gaps = False,
			one_alignment_only = True,
		)

		if len( alignments ) == 0:
			print( "Could not align the apo-holo sequences..." )
			return None
		else:
			aln = alignments[0]

		apo_aln = aln.seqA
		holo_aln = aln.seqB

		rep_seq = []
		rep_apo_res_num, rep_holo_res_num = [], []
		rep_apo_mask, rep_holo_mask = [], []

		apo_idx = 0
		holo_idx = 0

		for aa_apo, aa_holo in zip( apo_aln, holo_aln ):

			if aa_apo != "-":
				apo_res = apo_res_num[apo_idx]
				apo_mask = apo_missing[apo_idx]
				apo_idx += 1

			if aa_holo != "-":
				holo_res = holo_res_num[holo_idx]
				holo_mask = holo_missing[holo_idx]
				holo_idx += 1

			# Keep only aligned identical residues
			if aa_apo != "-" and aa_holo != "-" and aa_apo == aa_holo:
				rep_seq.append( aa_apo )
				rep_apo_res_num.append( apo_res )
				rep_holo_res_num.append( holo_res )
				rep_apo_mask.append( apo_mask )
				rep_holo_mask.append( holo_mask )

		# Create a mask that ignores missing residues in either apo/holo seq.
		missing_mask = np.array( [
			r1*r2 != 0 for r1, r2 in zip( rep_apo_mask, rep_holo_mask )
		] ).astype( bool )

		# Select residues that are observed in both apo and holo states.
		rep_seq = "".join( np.array( rep_seq )[missing_mask] )
		rep_apo_res_num = np.array( rep_apo_res_num )[missing_mask]
		rep_holo_res_num = np.array( rep_holo_res_num )[missing_mask]

		seq_id = np.arange( 1, len( rep_seq )+1, 1 ).astype( int )

		if len( rep_holo_res_num ) != len( rep_apo_res_num ):
			raise ValueError(
				"Auth residue numbering varies for apo and holo state. " +
				f"\napo: {rep_apo_res_num}" + 
				f"\nholo: {rep_holo_res_num}"
			)

		rep_dict = {
			"rep_seq": rep_seq,
			"rep_holo_res_num": rep_holo_res_num,
			"rep_apo_res_num": rep_apo_res_num,
			"seq_id": seq_id
		}
		return rep_dict

	################################################################################
	def save_rep_seq(
		self,
		rep_dict: Dict[str, Any]
	):
		"""
		Save the representative seq for apo and
			holo states as .pdb files.

		Inputs:
		----------
		rep_dict: dict containing the representative seq
			and the residue umbering for the apo, holo,
			and the representative seq.
		"""
		apo_asym_to_auth_asym = self.get_asym_auth_asym_map(
			cif_file = self.input_dict["apo_cif_file"]
		)
		holo_asym_to_auth_asym = self.get_asym_auth_asym_map(
			cif_file = self.input_dict["holo_cif_file"]
		)

		apo_asym_id = self.input_dict["apo_asym_id"]
		apo_auth_asym_id = apo_asym_to_auth_asym["asym"][apo_asym_id]
		holo_asym_id = self.input_dict["holo_asym_id"]
		holo_auth_asym_id = holo_asym_to_auth_asym["asym"][holo_asym_id]

		self.save_to_pdb(
			cif_file = self.input_dict["apo_cif_file"],
			pdb_file = self.input_dict["apo_out_file"],
			rep_seq = rep_dict["rep_seq"],
			auth_chain_id = apo_auth_asym_id,
			model_id = self.input_dict["apo_model_id"],
			auth_res_num = rep_dict["rep_apo_res_num"],
			seq_id = rep_dict["seq_id"]
		)
		self.save_to_pdb(
			cif_file = self.input_dict["holo_cif_file"],
			pdb_file = self.input_dict["holo_out_file"],
			rep_seq = rep_dict["rep_seq"],
			auth_chain_id = holo_auth_asym_id,
			model_id = self.input_dict["holo_model_id"],
			auth_res_num = rep_dict["rep_holo_res_num"],
			seq_id = rep_dict["seq_id"]
		)

	################################################################################
	def _set_residue_id( self, residue, new_id ):
		residue.id = new_id
		if residue.is_disordered() == 2:
			for child in residue.disordered_get_list():
				child.id = new_id


	def save_to_pdb(
		self,
		cif_file: str,
		pdb_file: str,
		rep_seq: str,
		auth_chain_id: str,
		model_id: int,
		auth_res_num: np.ndarray,
		seq_id: np.ndarray
	):
		"""
		Given a CIF file, save a subset of the structure
			to a .pdb file.
		We select the chain and the required residues (auth numbering)
			to be saved.
		Additionally, the residues are renumbered ith the provided
			numbering (seq_id).
		Arbitrarily renumbering can lead to conflicts in BioPython
			which expects the residue numbering to be unique.
			To avoid conflict we first provide a placeholder residue
				numbering to all residues and later update with the
				desired residue numbering.

		Inputs:
		----------
		cif_file: path to the CIF file from where to extract
			the chain of interest.
		pdb_file: path to the file to save the extracted chain.
		rep_seq: representative sequence (1-letter aa).
		authchain_id: auth_asym_id for the required chain.
		auth_res_num: auth residue numbering.
		seq_id: 1-indexed residue numbering for the rep_seq.
		"""
		p = Parser( pdb_file = cif_file )
		structure = p.structure

		if len( auth_res_num ) != len( seq_id ):
			raise ValueError(
				"The auth_res_num and seq_id lengths do not match. " +
				f"\nauth_res_num = {auth_res_num}" +
				f"\nseq_id = {seq_id}"
			)
		auth_to_idx = {r: i for i, r in enumerate( auth_res_num )}

		if model_id == -10:
			model = structure[0]
		else:
			model = structure[model_id-1]

		for chain in model:
			if chain.id != auth_chain_id:
				continue
			temp_id = -100
			for residue in chain:
				hetfield, resseq, icode = residue.id

				if resseq in auth_res_num:
					resname = residue.resname
					idx = auth_to_idx[resseq]
					rep_aa = rep_seq[idx]
					if rep_aa != aa_3_to_1( aa = resname ):
						raise ValueError(
							f"Amino acid in the representative seq {rep_aa} " +
							f"does not match that in the structure {resname} " +
							f"at auth_ress_num = {auth_res_num[idx]}..."
						)
					new_res_id = seq_id[idx]
					self._set_residue_id( residue = residue, new_id = ( hetfield, new_res_id, icode ) )
				else:
					# Residues that are not part of the rep_seq are
					# 	asseigned negative numbers - ChainSlect removes
					#	residues with negative numbering.
					self._set_residue_id( residue = residue, new_id = ( hetfield, temp_id, icode ) )
					temp_id -= 1

		io = PDBIO()
		io.set_structure( structure )
		if model_id == -10:
			# Single model structures (X-ray)
			io.save(
				pdb_file,
				select = ChainSelect(
					chain_id = auth_chain_id
				)
			)
		else:
			# Multi-model structures (NMR)
			io.save(
				pdb_file,
				select = ModelSelect(
					model_id = model_id-1,
					chain_id = auth_chain_id
				)
			)

	def compute_tm_score(
		self
	):
		"""
		Compute the TM-score vetween the apo and holo states.

		Inputs:
		----------
		rep_dict: dict containing the representative seq
			and the residue umbering for the apo, holo,
			and the representative seq.
		"""
		tmp_dir = os.path.join( "./ms_rep_tmp_dir/" )
		os.makedirs( tmp_dir, exist_ok = True )

		stdout_file = usalign(
			usalign_script = self.usalign_script,
			model_id1 = 0,
			model1_file = self.input_dict["apo_out_file"],
			model_id2 = 1,
			model2_file = self.input_dict["holo_out_file"],
			tmp_dir = tmp_dir,
			mm = 0
		)
		rmsd, tm = get_alignment_score( stdout_file = stdout_file )
		return rmsd, tm

################################################################################
# Multistate benchmark creation pipeline
################################################################################
class MultiStateBenchmark():
	"""
	Prepare a multi-state benchmark comprising monomers known
		to exist in holo and apo states with available PDB
		structures.
	"""
	def __init__( self ):
		self.config_dict = get_config_dict( is_multimer = False )
		self.cpu_cores = 10

		# Select entries with RMSD > the cutoff.
		self.apo_holo_rmsd_cutoff = 10
		# Select entries with domain motion only
		self.only_domain = True
		self.tm_cutoff = 0.5
		self.aa_for_xl = [
			"LYS", "ARG", "HIS", "ASP", "GLU",
			"ASN", "GLN", "SER", "THR", "TYR",
			"MET", "CYS", "LEU", "ILE", "ALA",
			"VAL", "GLY", "PRO", "TRP", "PHE"
			]


	def forward( self ):
		"""
		Parse the multi-state .csv file.
		Filter entries that do not fit the selection criterion.
		Obtain XLs from JWalk.
		"""
		self.create_required_file_paths()
		self.create_required_dir()

		self.run_multistate_pipeline()


	################################################################################
	################################################################################
	def create_required_file_paths( self ):
		"""
		Create the required directory and file paths.
		"""
		self.base_dir = os.path.join(
			os.path.abspath( self.config_dict.models.base_dir )
			)
		self.benchmark_name = "multistate"

		benchmark_dir_path = get_benchmark_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name
		)
		meta_dir_path = get_meta_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name
		)

		# Directory paths
		self.dir_paths = {
			"benchmark_dir": benchmark_dir_path,
			"meta_dir": meta_dir_path,
			"pdb_dir": os.path.join(
				meta_dir_path, "struct",
			),
			"jwalk_dir": os.path.join(
				meta_dir_path, "jwalk_output",
			)
		}
		# File paths
		self.file_paths = {
			"multistate_input_file": self.config_dict.benchmark.datasets.multi_state,
			"multistate_dict": os.path.join(
				meta_dir_path, "multistate_dict.npy"
			),
			"benchmark_csv_file": get_benchmark_csv_file(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				raw_file = False
			),
			"raw_benchmark_csv_file": get_benchmark_csv_file(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				raw_file = True
			)
		}


	def create_required_dir( self ):
		"""
		Create the required directories.
		"""
		for k,v in self.dir_paths.items():
			os.makedirs( v, exist_ok = True )


	def create_system_dir(
		self,
		sys_name_list: List[str]
		):
		"""
		Create system-specific dir.

		Inputs:
		----------
		sys_name_list: list of sytem names in the benchmark.
		"""
		for sys_name in sys_name_list:
			sys_dir_path = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name
			)
			os.makedirs( sys_dir_path, exist_ok = True )

	################################################################################
	################################################################################
	def run_multistate_pipeline( self ):
		"""
		The overall pipeline comprises of two stages:
			Metadata extraction
			Representative sequence creation
			Simulating XLs
			Check inter-state XL satisfaction
		"""
		if os.path.exists( self.file_paths["multistate_dict"] ):
			multistate_dict = np.load(
				self.file_paths["multistate_dict"], allow_pickle = True
			).item()
			print( "Multistate dict already exists..." )
		else:
			multistate_dict = self.metadata_extraction()
			self.get_pdb_resolution( multistate_dict = multistate_dict )
			print( "\nCreating the representative seq for the multistate system..." )
			multistate_dict = self.create_representative_seq_for_multistate(
				multistate_dict = multistate_dict
			)
		xl_file_paths = self.create_xl_file_paths(
			multistate_dict = multistate_dict
		)
		self.simulated_xl_generation(
			multistate_dict = multistate_dict,
			xl_file_paths = xl_file_paths
		)

		sys_configs = self.create_sys_config_dict(
			multistate_dict = multistate_dict
		)

		self.check_xl_satisfaction(
			multistate_dict = multistate_dict,
			sys_configs = sys_configs,
			xl_file_paths = xl_file_paths
		)

		self.save_benchmark_csv(
			multistate_dict = multistate_dict
		)

	################################################################################
	def metadata_extraction( self ) -> Dict[str, Any]:
		"""
		Parse the multistate input file and extract the PDB IDs for
			the required entries.
			Exclude NMR structures.
		Create system-specific directory.
		Download the biological assembly for the selected PDB IDs.
		Parse the mmcif file to obtain the relevant metadata for both
			states of each system.
		Extract the required chain from the PDB file and save on disk.
		"""
		multistate_dict = self.create_multistate_dict()

		self.create_system_dir(
			sys_name_list = list( multistate_dict.keys() )
		)
		self.download_multistate_struct(
			multistate_dict = multistate_dict
		)

		return multistate_dict

	################################################################################
	def create_representative_seq_for_multistate(
		self,
		multistate_dict: Dict[str, Any]
	):
		"""
		For all multistate systems, create a representative sequence
			to be modeled.
		Extract the coordinates for the representative seq from the
			apo and holo states and save on disk.
		Ignore entries for which,
			Could not align apo and holo state sequences.
			apo and holo states are too similar.
		"""
		remove_sys = []
		for sys_name in multistate_dict:
			print( sys_name )
			sys_dir_path = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name
			)

			if "representative" in multistate_dict[sys_name]:
				print( f"Representative seq already exissts for {sys_name}..." )
				continue

			apo_pdb = multistate_dict[sys_name]["apo"]["pdb_id"]
			apo_asym_id = multistate_dict[sys_name]["apo"]["chain_id"]
			apo_model_id = multistate_dict[sys_name]["apo"]["model_id"]
			apo_cif_file = os.path.join(
					self.dir_paths["pdb_dir"], f"{apo_pdb}.cif"
				)
			apo_pdb_file = os.path.join(
					sys_dir_path, f"{sys_name}_S1.pdb"
				)

			holo_pdb = multistate_dict[sys_name]["holo"]["pdb_id"]
			holo_asym_id = multistate_dict[sys_name]["holo"]["chain_id"]
			holo_model_id = multistate_dict[sys_name]["holo"]["model_id"]
			holo_cif_file = os.path.join(
					self.dir_paths["pdb_dir"], f"{holo_pdb}.cif"
				)
			holo_pdb_file = os.path.join(
					sys_dir_path, f"{sys_name}_S2.pdb"
				)

			obj = ApoHoloStates(
				apo_cif_file = apo_cif_file,
				holo_cif_file = holo_cif_file,
				apo_asym_id = apo_asym_id,
				apo_model_id = apo_model_id,
				holo_asym_id = holo_asym_id,
				holo_model_id = holo_model_id,
				apo_out_file = apo_pdb_file,
				holo_out_file = holo_pdb_file
			)
			rep_dict = obj.forward()
			if rep_dict is None:
				remove_sys.append( sys_name )
				continue
			# Ignore entries for which apo and holo states are too similar.
			if rep_dict["tm"] > self.tm_cutoff:
				remove_sys.append( sys_name )
				continue
			sys_length = len( rep_dict["rep_seq"] )
			if sys_length > self.config_dict.benchmark.globals.max_sys_length:
				print( f"{sys_name} exceeds max length: {sys_length}..." )
				remove_sys.append( sys_name )
				continue

			multistate_dict[sys_name]["representative"] = {
				"seq": rep_dict["rep_seq"],
				"seq_id": rep_dict["seq_id"],
				"tm": rep_dict["tm"],
				"rmsd": rep_dict["rmsd"]
			}
			print( f"TM-score = {rep_dict['tm']}; RMSD = {rep_dict['rmsd']}" )
		for sys_name in remove_sys:
			_ = multistate_dict.pop( sys_name )

		np.save(
			self.file_paths["multistate_dict"],
			multistate_dict,
			allow_pickle = True
		)

		print( "Select multistate entries = ", len( multistate_dict ) )
		return multistate_dict

	################################################################################
	def simulated_xl_generation(
		self,
		multistate_dict: Dict[str, Any],
		xl_file_paths: Dict[str, Dict]
	):
		"""
		Simulate XLs using JWalk.
		Map PDB residue numbering to seq_id.
		Map the chain IDs to proteins (incorporating ambiguity).
		Remove overlapping XLs and subsample XLs for each state.
		Save XLs on disk for,
			State 1/2 separately.
			State 1+2.
		"""
		remove = []
		for sys_name in multistate_dict:
			sys_dir_path = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name
			)

			if self.xl_file_exist( sys_xl_file_path = xl_file_paths[sys_name] ):
				print( f"XL files already exist for {sys_name}..." )
				continue

			processed_xls_file = os.path.join(
				self.dir_paths["jwalk_dir"],
				f"{sys_name}_xls.npy"
			)
			if os.path.exists( processed_xls_file ):
				xls_dict = np.load( processed_xls_file, allow_pickle = True ).item()
			else:
				xls_dict = {}
				for state in multistate_dict[sys_name]:
					if state == "representative":
						continue
					state_id = multistate_dict[sys_name][state]["state_id"]
					# Get XLs for each state.
					xls_df = self.xl_simulator(
						sys_name = state_id,
						jwalk_dir = self.dir_paths["jwalk_dir"],
						pdb_struct_dir = sys_dir_path
					)

					xls_dict[state_id] = xls_df
				np.save( processed_xls_file, xls_dict, allow_pickle = True )

			xls_df1, xls_df2 = self.select_xls_for_multistate(
				sys_name = sys_name,
				xls_dict = xls_dict,
				sys_dir_path = sys_dir_path
			)

			xls_df1 = self.map_chains_to_prot(
				sys_name = sys_name,
				xls_df = xls_df1
			)
			xls_df2 = self.map_chains_to_prot(
				sys_name = sys_name,
				xls_df = xls_df2
			)

			# Xls from both states together.
			xls_s1_2 = pd.concat(
				[xls_df1, xls_df2], axis = 0
			)
			xls_s1_2 = xls_s1_2.reset_index( drop = True )
			# No XLs left
			if xls_df1.shape[0] < 5 or xls_df2.shape[0] < 5:
				remove.append( sys_name )
			else:
				self.save_multistate_xls(
					xls_s1_df = xls_df1,
					xls_s2_df = xls_df2,
					xls_s1_2_df = xls_s1_2,
					sys_xl_file_path = xl_file_paths[sys_name]
				)
		for k in remove:
			multistate_dict.pop( k )

	################################################################################
	def xl_file_exist(
		self,
		sys_xl_file_path: Dict[str, str]
	):
		"""
		Check if the Xl file paths exist or not.
			Returns true, if all 3 xl_file paths exist.
		"""
		check = []
		for k in sys_xl_file_path:
			xl_file = sys_xl_file_path[k]
			check.append( os.path.exists( xl_file ) )
		return all( check )


	def create_xl_file_paths(
		self,
		multistate_dict: Dict[str, str]
	) -> Dict[str, Dict]:
		"""
		Create the file path to the XL file for all selected systems.
		"""
		xl_file_paths = {}
		for sys_name in multistate_dict:
			sys_dir_path = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name
			)

			state1_file = os.path.join(
				sys_dir_path, "interprotein_xls_S1.csv"
				)
			state2_file = os.path.join(
				sys_dir_path, "interprotein_xls_S2.csv"
				)
			state1_2_file = os.path.join(
				sys_dir_path, "interprotein_xls_S1_2.csv"
				)
			xl_file_paths[sys_name] = {
				"state1": state1_file,
				"state2": state2_file,
				"state1_2": state1_2_file
			}
		return xl_file_paths

	################################################################################
	################################################################################
	def parse_multistate_csv(
		self
	) -> pd.DataFrame:
		"""
		Load and return the multi-state benchmark.
		Input file has a ";" separated format.
		"""
		df = pd.read_csv( self.file_paths["multistate_input_file"], sep = ";" )
		return df


	def preprocess_multistate_benchmark(
		self,
		multistate_df: pd.DataFrame
	) -> pd.DataFrame:
		"""
		We select entries from the benchmark which:
			Involve domain movements.
			Have a rmsd_apo_holo >10 angstrom.
		Return the filtered dataframe.
		"""
		drop_rows = []
		for i in multistate_df.index:
			if self.only_domain:
				if "domain" not in multistate_df["motion_type"][i]:
					drop_rows.append( i )
			if multistate_df["rmsd_apo_holo"][i] < self.apo_holo_rmsd_cutoff:
				drop_rows.append( i )
		selected_df = multistate_df.drop( drop_rows, axis = 0 )
		selected_df = selected_df.reset_index( drop = True )
		return selected_df

	################################################################################
	def create_multistate_dict( self ) -> Dict[str, str]:
		"""
		Preprocess the entries in the multistate benchmark.
		For the selected multi-state entries, extract the following metadata,
			PDB IDs for apo and holo state
		The apo_id/holo_id have the following format,
			{PDB_ID}-{MODEL_ID}_{CHAIN_ID}
			PDB_ID -> PDB accession
			MODEL_ID -> model_id for NMR structures
			CHAIN_ID -> chain identifier
		We define the system name as: "ms{INDEX}
			INDEX -> 0-indexed integer identifier
		"""
		multistate_dict = {}
		multistate_df = self.parse_multistate_csv()
		selected_df = self.preprocess_multistate_benchmark(
			multistate_df = multistate_df
		)
		print(
			"Selected multistate entries...\n",
			selected_df.shape, "\n",
			selected_df.head() )

		for i in selected_df.index:
			apo_id = selected_df["apo_id"][i]
			holo_id = selected_df["holo_id"][i]

			# # Has a large missing region (4lp5).
			# if apo_id == "4LP5_A" and holo_id == "4P2Y_A":
			# 	continue

			apo_pdb, apo_chain = apo_id.split( "_" )
			holo_pdb, holo_chain = holo_id.split( "_" )

			if "-" in apo_pdb:
				apo_pdb, apo_model_id = apo_pdb.split( "-" )
				apo_model_id = int( apo_model_id )
			else:
				# Placeholder value for X-ray structures
				apo_model_id = -10

			if "-" in holo_pdb:
				holo_pdb, holo_model_id = holo_pdb.split( "-" )
				holo_model_id = int( holo_model_id )
			else:
				holo_model_id = -10

			sys_name = f"ms{i}"
			multistate_dict[sys_name] = {
				"apo": {
					"pdb_id": apo_pdb.lower(),
					"chain_id": apo_chain,
					"model_id": apo_model_id,
					"state_id": f"{sys_name}_S1",
					"motion_type": selected_df["motion_type"][i]
				},
				"holo": {
					"pdb_id": holo_pdb.lower(),
					"chain_id": holo_chain,
					"model_id": holo_model_id,
					"state_id": f"{sys_name}_S2",
					"motion_type": selected_df["motion_type"][i]
				}
			}
		return multistate_dict

	################################################################################
	def download_per_pdb( self, pdb_id: str ):
		"""
		"""
		for ext in ["cif", "pdb"]:
			pdb_file = os.path.join(
					self.dir_paths["pdb_dir"], f"{pdb_id}.{ext}"
				)
			if os.path.exists( pdb_file ):
				continue
			download_pdb(
				pdb_id = pdb_id,
				ext = ext,
				file_name = pdb_file,
				download_assembly = False
			)


	def download_multistate_struct(
		self,
		multistate_dict: Dict[str, str]
	):
		"""
		Download the structure from PDB in .cif/.pdb format for the apo and
			holo states for all entries.
		We consider the biological assembly.
		"""
		print( "\nDownloading structures for multistate benchmark..." )
		pdb_ids_list = []
		for sys_name in multistate_dict:
			# print( sys_name )
			for state in ["apo", "holo"]:
				pdb_id = multistate_dict[sys_name][state]["pdb_id"]
				pdb_ids_list.append( pdb_id )

		with Pool( self.cpu_cores ) as p:
			for result in tqdm.tqdm(
				p.imap( self.download_per_pdb, pdb_ids_list ),
				total = len( pdb_ids_list ),
				desc = "Downloading structure"
			):
				pass

	################################################################################
	def get_pdb_resolution(
		self,
		multistate_dict: Dict[str, str]
	):
		"""
		# When downloading the biological assembly, the cif file
		# 	does not contain the resolution of the structure.
		Get the resolutions for all separately
			from the PDB REST API.
		"""
		print( "\nFetching the resolution for biological assemblies..." )
		resolution_dict = {}
		def get_resolution( entry_id: str ):
			"""
			Fetch the resolution for the given entry_id from the PDB REST API.
			"""
			rest = PdbRestApi( entry_id = entry_id )
			entry_data = rest.entry_data
			if entry_data == None:
				# Sanity check: at this stage the PDB entry exists so
				# 	 the REST API must return the entry details.
				raise ValueError(
					f"Could not fetch data from the PDB REST API for {entry_id}..."
				)
			if "resolution_combined" in entry_data["rcsb_entry_info"]:
				resolution = entry_data["rcsb_entry_info"]["resolution_combined"]
			else:
				resolution = 0.0
			return entry_id, resolution

		resolution_file = os.path.join(
			self.dir_paths["meta_dir"], "resolution_dict.json"
		)
		if os.path.exists( resolution_file ):
			print( "Resolution dict already exists..." )
		else:
			pdb_ids_list = []
			for sys_name in multistate_dict:
				for state in ["apo", "holo"]:
					pdb_id = multistate_dict[sys_name][state]["pdb_id"]
					pdb_ids_list.append( pdb_id )

			with ThreadPoolExecutor( self.cpu_cores ) as executor:
				futures = [
					executor.submit( get_resolution, entry_id )
					for entry_id in pdb_ids_list
				]
				for future in futures:
					entry_id, resolution = future.result()
					resolution_dict[entry_id] = resolution
		
			resolution_file = os.path.join(
				self.dir_paths["meta_dir"], "resolution_dict.json"
			)
			write_json( dict_ = resolution_dict, file_path = resolution_file )

		return resolution_dict

	################################################################################
	################################################################################
	def xl_simulator(
		self,
		sys_name: str,
		jwalk_dir: str,
		pdb_struct_dir: str
	) -> Dict[str, pd.DataFrame]:
		"""
		Simulate Xls using JWalk.
		We simulate XLs involving several amino acids as shown below.
			We set the num_inter_xl to 1 so as to select all XLs per aa.
		No FP XL included.
		Merge all TP XLs obtained from all aa.

		Inputs:
		----------
		sys_name_list: list of system anmes; these correspond to the
			apo/holo state IDs for all multistate entries.
		jwalk_dir: dir path to store the JWalk output.
		pdb_struct_dir: path to the dir containing the PDB file for the system.

		Returns:
		----------
		xl_df: pd.DataFrame containing Xls simulated using JWalk.
		"""
		print( "\nSimulating XLs using JWalk..." )

		xl_df = pd.DataFrame( {} )
		for aa1 in self.aa_for_xl:
			for aa2 in self.aa_for_xl:
				print( f"\n{sys_name} Computing XLs for aa: {aa1}-{aa2} " + "-"*20 )
				sim_obj = SimulateCrosslinks(
					jwalk_exec = self.config_dict.benchmark.jwalk.jwalk_exec,
					pdb_ids_list = [sys_name],
					jwalk_dir = jwalk_dir,
					pdb_struct_dir = pdb_struct_dir,
					struct_format = "pdb",
					short_linker = self.config_dict.benchmark.jwalk.short_linker,
					long_linker = self.config_dict.benchmark.jwalk.long_linker,
					num_xls = 1,
					cores = self.cpu_cores,
					aa1 = aa1,
					aa2 = aa2
				)
				sim_obj.reinit_logs = True
				sim_obj.inter_xls = False
				sim_obj.forward()
				xls_dict_aa = sim_obj.xls_dict

			# If no Xls selected.
			if sys_name not in xls_dict_aa:
				print( "No XLs obtained..." )
				continue
			# Merge XLs from all the specified XLs.
			if xl_df.shape[0] == 0:
				xl_df = xls_dict_aa[sys_name]["short_xls"]
			else:
				xl_df = pd.concat(
					[
						xl_df,
						xls_dict_aa[sys_name]["short_xls"]
					],
					axis = 0
				)

			xl_df = xl_df.reset_index( drop = True )
		print( f"Total XLs obtained for {sys_name}: {xl_df.shape}" )
		return xl_df

	################################################################################
	def sample_xls(
		self,
		xls_df: pd.DataFrame,
		num_xls: int
	) -> pd.DataFrame:
		"""
		Randomly select a subset of num_xls XLs.

		Inputs:
		----------
		xls_df: pd.DataFrame containing XLs in the format,
			prot1,res1,prot2,res2
		num_xls: max XLs to sample.

		Returns:
		----------
		xls: pd.DataFrame containing subsampled XLs.
		"""
		np.random.seed( self.config_dict.prng_seed )
		if xls_df.shape[0] <= num_xls:
			xls = xls_df
		else:
			indexes = list( xls_df.index )
			# Sample a subset of XLs without replacement.
			
			sampled_idx = np.random.choice(
				a = indexes,
				size = num_xls,
				replace = False
			)
			xls = xls_df.iloc[sampled_idx]
		return xls

	################################################################################
	################################################################################
	def compute_xl_satisfaction(
		self,
		coords: Dict[int, np.ndarray],
		xl_df: pd.DataFrame,
	) -> tuple[float, List[int]]:
		"""
		Compute XL satisfaction for the given state (coordinates)
			and XLs.

		Inputs:
		----------
		coords: dict containing the residue numbers mapped to
			the coordinates.
		xl_df: pd.dataFrame containing XLs.

		Returns:
		----------
		xl_satisfaction: fraction of XLs satisfied.
		xl_sat_idx: a list of indices for the satisfied XLs.
		"""

		total = xl_df.shape[0]
		satisfied = 0
		xl_sat_idx = []
		for i in xl_df.index:
			r1 = int( xl_df["res1"][i] )
			r2 = int( xl_df["res2"][i] )

			dist = np.linalg.norm( coords[r2] - coords[r1] )
			if dist <= self.config_dict.benchmark.jwalk.short_linker:
				satisfied += 1
				xl_sat_idx.append( i )

		if total == 0:
			xl_satisfaction = -1
		else:
			xl_satisfaction = satisfied/total
		return xl_satisfaction, xl_sat_idx

	################################################################################
	def filter_cross_xls(
		self,
		pdb_file: str,
		xl_df: pd.DataFrame
	) -> pd.DataFrame:
		"""
		Remove XLs which are satisfied by the other given structure.
			This is meant to use for removing XLs from state1 that
			are satisfied by state2 and vice-versa.

		Inputs:
		----------
		pdb_file: file path for the structure.
		xl_df: pd.dataFrame containing XLs.

		Returns:
		----------
		xl_df: dataframe containing XLs not satisfied by the
			given structure.
		"""
		coords = {}
		struct = Parser( pdb_file = pdb_file ).structure
		for chain in struct[0]:
			for residue in chain:
				if residue.id[0] != " ":
					continue
				res_id = int( residue.id[1] )
				coords[res_id] = residue["CA"].get_coord()

		# Drop XLs satisfied by the given structure.
		_, drop_rows = self.compute_xl_satisfaction(
			coords = coords,
			xl_df = xl_df
		)
		xl_df = xl_df.drop( drop_rows, axis = 0 )
		xl_df = xl_df.reset_index( drop = True )
		return xl_df

	################################################################################
	def select_xls_for_multistate(
		self,
		sys_name: str,
		xls_dict: Dict[str, pd.DataFrame],
		sys_dir_path: str
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""
		Select Xls for the two states.
			Segregte the XLs into three categories:
				State1 XLs only
				State2 XLs only
				Common XLs
			Randomly select a subset fo 8 XLs for each state.
			The two states must have no overlapping XL.
				Ignore the common XLs.

		Inputs:
		----------
		xls_dict: dict containing the JWalk simulated XLs as a pd.DataFrame
			for each state.
		num_xls: no. of XLs to be selected.

		Returns:
		----------
		xls_s1: pd.DataFrame containing XLs for state1.
		xls_s2: pd.DataFrame containing XLs for state2.
		"""
		state1, state2 = list( xls_dict.keys() )
		xls_s1_all = xls_dict[state1]
		xls_s2_all = xls_dict[state2]

		state1_file = os.path.join( sys_dir_path, f"{sys_name}_S1.pdb" )
		state2_file = os.path.join( sys_dir_path, f"{sys_name}_S2.pdb" )

		xls_s1 = self.filter_cross_xls(
			pdb_file = state2_file,
			xl_df = xls_s1_all
		)
		xls_s2 = self.filter_cross_xls(
			pdb_file = state1_file,
			xl_df = xls_s2_all
		)

		# Sample a subset of XLs.
		xls_s1 = self.sample_xls(
			xls_df = xls_s1,
			num_xls = self.config_dict.benchmark.jwalk.num_inter_xls
		)
		xls_s2 = self.sample_xls(
			xls_df = xls_s2,
			num_xls = self.config_dict.benchmark.jwalk.num_inter_xls
		)
		xls_s1 = xls_s1.reset_index( drop = True )
		xls_s2 = xls_s2.reset_index( drop = True )

		# Add a "label" column; indicating TP/FP.
		labels = [1]*xls_s1.shape[0]
		xls_s1["label"] = labels
		labels = [1]*xls_s2.shape[0]
		xls_s2["label"] = labels

		print( f"State1 XLs: {xls_s1.shape[0]} \t State2 XLs: {xls_s2.shape[0]}" )
		return xls_s1, xls_s2

	################################################################################
	def map_chains_to_prot(
		self,
		sys_name: str,
		xls_df: pd.DataFrame,
	) -> pd.DataFrame:
		"""
		The prot1/2 contains chain_ids labels need to be mapped to protein labels.
		e.g. A -> {sys_name}_1; B -> {sys_name}_2

		Inputs:
		----------
		xls_df: pd.DataFrame containing XLs for state1.

		Returns:
		----------
		xls_df: pd.DataFrame containing XLs for state1 with
			chain IDs mapped to protein names.
		"""
		for i in xls_df.index:
			xls_df.loc[i, "prot1"] = f"{sys_name}_1"
			xls_df.loc[i, "prot2"] = f"{sys_name}_1"
		return xls_df

	################################################################################
	def save_multistate_xls(
		self,
		sys_xl_file_path: Dict[str, str],
		xls_s1_df: pd.DataFrame,
		xls_s2_df: pd.DataFrame,
		xls_s1_2_df: pd.DataFrame
	):
		"""
		Save the XLs for state1/2 on disk in the system data dir.

		Inputs:
		----------
		data_dir: path to the system specific data dir.
		xls_s1: pd.DataFrame containing XLs for state1.
		xls_s2: pd.DataFrame containing XLs for state2.
		"""
		state1_file = sys_xl_file_path["state1"]
		state2_file = sys_xl_file_path["state2"]
		state1_2_file = sys_xl_file_path["state1_2"]
		xls_s1_df.to_csv( state1_file, index = False )
		xls_s2_df.to_csv( state2_file, index = False )
		xls_s1_2_df.to_csv( state1_2_file, index = False )

	################################################################################
	################################################################################
	def create_sys_config_dict(
		self,
		multistate_dict: Dict[str, str],
	):
		"""
		Create the system config dict required for running model
			predictions.
		Save the config_dict on disk in the system sata dir.
		"""
		sys_configs = {}
		for sys_name in multistate_dict:
			rep_seq = multistate_dict[sys_name]["representative"]["seq"]
			residues = multistate_dict[sys_name]["representative"]["seq_id"]
			start_res, end_res = residues[0], residues[-1]

			sys_configs[sys_name] = {
				"name": sys_name,
				"entity": [
					{
						"entity_id": 1,
						"copy_num": 1,
						"start": int( start_res ),
						"end": int( end_res ),
						"sequence": rep_seq
					}
				]
			}
			sys_config_path = get_sys_config_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name
			)
			write_json(
				dict_ = sys_configs[sys_name],
				file_path = sys_config_path
			)
		return sys_configs

	################################################################################
	################################################################################
	def check_xl_satisfaction(
		self,
		multistate_dict,
		sys_configs: Dict[str, Dict],
		xl_file_paths: Dict[str, Dict]
	):
		"""
		Check the Xl satisfaction of:
			State1 for state2 XLs.
			State2 for state1 XLs.
		"""
		xl_sat_dict = {}
		for sys_name in sys_configs:
			xl_sat_dict[sys_name] = {}
			sys_dir_path = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name
			)

			state1_pdb = os.path.join( sys_dir_path, f"{sys_name}_S1.pdb" )
			state2_pdb = os.path.join( sys_dir_path, f"{sys_name}_S2.pdb" )
			xls_s1_df = pd.read_csv( xl_file_paths[sys_name]["state1"] )
			xls_s2_df = pd.read_csv( xl_file_paths[sys_name]["state2"] )

			for i, pdb_file in enumerate( [state1_pdb, state2_pdb], start = 1 ):
				for j, xl_df in enumerate( [xls_s1_df, xls_s2_df], start = 1 ):

					coords = {}
					struct = Parser( pdb_file = pdb_file ).structure
					for chain in struct[0]:
						for residue in chain:
							if residue.id[0] != " ":
								continue
							res_id = int( residue.id[1] )
							coords[res_id] = residue["CA"].get_coord()

					xl_sat, _ = self.compute_xl_satisfaction(
						coords = coords,
						xl_df = xl_df
					)

					xl_sat_dict[sys_name][f"S{i}_S{j}"] = xl_sat
		print( "\nXL satisfaction across all systems..." )
		for sys_name in xl_sat_dict:
			print( sys_name, "-"*20 )
			for k in xl_sat_dict[sys_name]:
				print( f"\t{k} -> {xl_sat_dict[sys_name][k]}" )
		return xl_sat_dict

	################################################################################
	################################################################################
	def save_benchmark_csv(
		self,
		multistate_dict
	):
		"""
		Save a .csv file for the benchmark.
		This is needed for the downstream scripts to run.
		"""
		# These are minimum fields needed.
		flat_dict = {k:[] for k in [
			"PDB ID", "Auth Asym ID", "Stoichiometry", "Total length",
			"Apo PDB ID", "Holo PDB ID", "TM-score", "RMSD", "Motion type"
			]}

		for sys_name in multistate_dict:
			flat_dict["PDB ID"].append( sys_name )
			# We don't care for the exact chain_id.
			# 	We have two different states here.
			flat_dict["Auth Asym ID"].append( "A" )
			flat_dict["Stoichiometry"].append( 1 )
			flat_dict["Total length"].append(
				len( multistate_dict[sys_name]["representative"]["seq"] )
			)
			flat_dict["Apo PDB ID"].append(
				multistate_dict[sys_name]["apo"]["pdb_id"]
			)
			flat_dict["Holo PDB ID"].append(
				multistate_dict[sys_name]["holo"]["pdb_id"]
			)
			flat_dict["TM-score"].append(
				multistate_dict[sys_name]["representative"]["tm"]
			)
			flat_dict["RMSD"].append(
				multistate_dict[sys_name]["representative"]["rmsd"]
			)
			flat_dict["Motion type"].append(
				multistate_dict[sys_name]["holo"]["motion_type"]
			)

		df = pd.DataFrame( flat_dict )
		df.to_csv( self.file_paths["benchmark_csv_file"], index = False )
		df.to_csv( self.file_paths["raw_benchmark_csv_file"], index = False )

################################################################################
################################################################################
if __name__ == "__main__":
	MultiStateBenchmark().forward()
	print( "May the Force be with you..." )
