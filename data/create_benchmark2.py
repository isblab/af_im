"""
Contains module to create the required files for the
	complexes in the benchmark.
"""
from typing import List, Tuple, Dict, Any
import os, math
import numpy as np
import pandas as pd

from config import get_config_dict
from utils.utils import ( open_file_handler,
							read_json,
							write_json,
							run_subprocess )
from utils.paths import (
	get_meta_dir_path,
	get_sys_data_dir_path,
	get_xl_file_path,
	get_sys_config_path
)


class CreateBenchmark():
	"""
	Create input files for all systems (complexes)
		in the benchmark.
	"""
	def __init__( self ):
		self.config_dict = get_config_dict()
		self.dataset_configs = self.config_dict.benchmark
		self.benchmark_name = self.dataset_configs.globals.benchmark_name
		np.random.seed( self.config_dict.prng_seed )

		self.chain_entity_map = {}
		self.xl_datasets = {}


	def forward( self ):
		"""
		Map all chain IDs to the corresponding entity ID
			to model ambiguity.
		Create the following input files for each system:
			XLs dataset containing:
				TP Xls with short inker.
				TP Xls with long inker.
				FP Xls with short inker.
			Config file containing the the sequence and
				restraint info for the system.
		Note: True positive (TP/tp); False positive (FP/fp); Cross-link (XL)
		"""
		self.create_required_paths()
		self.required_dir_exist()
		self.load_metadata()

		# Create chain ID to entity DI mapping.
		self.create_chain_entity_map()
		# Map chain IDs to entity IDs.
		self.map_xl_chain_to_entity()
		# Create datasets for all required xl types.
		self.create_xl_dataset()
		# Create inputs for all complexes in the benchmark.
		self.create_system_inputs_for_modeling()
		self.write_benchmark_csv()
		print( "\n May the Force be with you..." )

	################################################################################
	################################################################################
	def create_required_paths( self ):
		"""
		Create paths for all required directories and files.
		"""
		## --------------------------
		# Global paths
		## --------------------------
		# Base directory for all benchmarks.
		self.base_dir = os.path.join(
			os.path.abspath( self.dataset_configs.globals.base_dir )
			)
		# Benchmark specific dir.
		self.benchmark_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_benchmark" )
		# Dir to store benchmark metadata including structure file,
		# 	seqres dict, benchamrk csv and the required intermediate files.
		self.meta_dir = get_meta_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name
		)
		# Dir containing PDB structures.
		self.pdb_struct_dir = os.path.join( self.meta_dir, "struct" )
		# PDB IDs remaining after metadat collection.
		self.benchmark_pdbs_file =  os.path.join( self.meta_dir,
										f"{self.benchmark_name}_benchmark_pdb_ids.txt" )

		self.seqres_dict_file = os.path.join( self.meta_dir, "seqres_dict.npy" )
		self.xls_dict_file = os.path.join( self.meta_dir, "jwalk_xls.npy" )


	def required_dir_exist( self ):
		"""
		Check if the required dir and files exist.
		"""
		if not os.path.exists( self.base_dir ):
			raise FileNotFoundError( f"Base dir: {self.base_dir} does not exist..." )

		if not os.path.exists( self.meta_dir ):
			raise FileNotFoundError( f"Metadata dir: {self.meta_dir} does not exist..." )

		if not os.path.exists( self.seqres_dict_file ):
			raise FileNotFoundError( f"Seqres dict file does not exist..." )

		if not os.path.exists( self.xls_dict_file ):
			raise FileNotFoundError( f"Simulated XLs dict does not exist..." )


	def load_metadata( self ):
		"""
		Load all required metadata on disk.
		"""
		f = open_file_handler( self.benchmark_pdbs_file, "r" )
		self.benchmark_pdb_ids = f.readlines()[0].split( "," )
		f.close()
		self.seqres_dict = np.load( self.seqres_dict_file, allow_pickle = True ).item()
		self.jwalk_xls_dict = np.load( self.xls_dict_file, allow_pickle = True ).item()

	################################################################################
	################################################################################
	def create_chain_entity_map( self ):
		"""
		For a given PDB ID (sys_name), create a dict mapping the auth_asym_ids to
			their respective entity_id.
		"""
		for sys_name in self.benchmark_pdb_ids:
			if sys_name not in self.seqres_dict:
				raise KeyError( f"{sys_name} does not exists in self.seqres_dict..." )

			self.chain_entity_map[sys_name] = {}

			entity_ids = list( self.seqres_dict[sys_name].keys() )
			for entity_id in entity_ids:
				auth_asym_ids = list( self.seqres_dict[sys_name][entity_id].keys() )

				for aa_id in auth_asym_ids:
					self.chain_entity_map[sys_name][aa_id] = entity_id


	def map_xl_chain_to_entity( self ):
		"""
		XL data is usually obtained at the resolution of entiies
			(protein/molecule) not instances (chains).
			This leads to ambiguity in XL data when >1 copy (chain)
				for an entity is present.
		Hence, we map the chain IDs for all XLs to the respective
			entity to introduce ambiguity in our dataset.
		e.g. For a system contaiing chains A-B,C:
			A -> Protein_1; B -> Protein_1; C -> Protein_2
		This method modifies the existing self.jwalk_xls_dict.

		An XL is removed, if the chain ID does not match any of the
			chain IDs associated with the entity.
		"""
		for sys_name in self.benchmark_pdb_ids:
			for xl_type in ["short_xls", "long_xls", "fp_xls"]:
				if xl_type not in self.jwalk_xls_dict[sys_name]:
					raise KeyError( f"XL type - {xl_type} does" +
						"not exist for sys_name = {sys_name}..." )
				xl_df = self.jwalk_xls_dict[sys_name][xl_type]
				drop_rows = []
				for i in xl_df.index:
					chain1 = xl_df.loc[i, "prot1"]
					chain2 = xl_df.loc[i, "prot2"]

					if chain1 not in self.chain_entity_map[sys_name]:
						drop_rows.append( i )
						continue
					if chain2 not in self.chain_entity_map[sys_name]:
						drop_rows.append( i )
						continue

					entity_id1 = self.chain_entity_map[sys_name][chain1]
					prot1 = f"{sys_name}_{entity_id1}"
					entity_id2 = self.chain_entity_map[sys_name][chain2]
					prot2 = f"{sys_name}_{entity_id2}"

					xl_df.loc[i, "prot1"] = prot1
					xl_df.loc[i, "prot2"] = prot2
				xl_df = xl_df.drop( drop_rows )
				xl_df = xl_df.reset_index( drop = True )
				self.jwalk_xls_dict[sys_name][xl_type] = xl_df

	################################################################################
	################################################################################
	def sample_tp_xls(
		self,
		tp_xls_df: pd.DataFrame,
		num_tp_xls: int
		) -> pd.DataFrame:
		"""
		Randommly sample a specified subset of TP XLs.
		If the no. of TP XLs is <= the required no. of XLs,
			select all.

		Inputs:
		----------
		tp_xls_df: pd.dataFrame containing TP XLs in the following format
			prot1,res1,prot2,res2
		num_tp_xls: max no. of TP XLs required.

		Returns:
		----------
		tp_xls: pd.DataFrame containing the sampled subset of XLs.
		"""
		if tp_xls_df.shape[0] <= num_tp_xls:
			tp_xls = tp_xls_df
		else:
			indexes = list( tp_xls_df.index )
			# Sample a subset of XLs without replacement.
			sampled_idx = np.random.choice( a = indexes,
											size = num_tp_xls,
											replace = False )
			tp_xls = tp_xls_df.iloc[sampled_idx]
		return tp_xls


	def sample_fp_xls(
		self,
		fp_xls_df: pd.DataFrame,
		num_fp_xls: int
		) -> pd.DataFrame:
		"""
		Select a specified no. of FP XLs.
		Sort in descending order based on the XL distance.
			FP XLs are pre-sorted in descending order, so
				just select from the top.

		Inputs:
		----------
		fp_xls_df: pd.dataFrame containing FP XLs in the following format
			prot1,res1,prot2,res2
		num_fp_xls: max no. of FP XLs required.

		Returns:
		----------
		fp_xls: pd.DataFrame containing the sampled subset of FP XLs.
		"""
		fp_xls = fp_xls_df[:num_fp_xls]
		return fp_xls


	def xl_mixer(
		self,
		tp_xls_df: pd.DataFrame,
		fp_xls_df: pd.DataFrame,
		frac_tp: float,
		frac_fp: float,
		max_xls: int
		) -> pd.DataFrame:
		"""
		Create an XL datset comprising the requires no. of TP and FP XLs.
		Select the specified no. of TP short linker
			XLs and FP XLs.

		Inputs:
		----------
		tp_xls_df: pd.dataFrame containing TP XLs in the following format
			prot1,res1,prot2,res2
		fp_xls_df: pd.dataFrame containing FP XLs in the following format
			prot1,res1,prot2,res2
		frac_tp: fraction of TP XLs required.
		frac_fp: fraction of FP XLs required.
		max_xls: max no. of XLs required.

		Returns:
		----------
		xls_df: pd.DataFrame containing the sampled subset of
			TP and FP XLs based on the specified fractions.
		"""
		num_tp_xls = math.ceil( frac_tp*max_xls )
		num_fp_xls = math.ceil( frac_fp*max_xls )

		tp_xls = self.sample_tp_xls(
			tp_xls_df = tp_xls_df,
			num_tp_xls = num_tp_xls )

		fp_xls = self.sample_fp_xls(
			fp_xls_df = fp_xls_df,
			num_fp_xls = num_fp_xls )

		xls_df = pd.concat( [tp_xls, fp_xls] )

		tp_fp_label = [1]*tp_xls.shape[0] + [0]*fp_xls.shape[0]
		xls_df["label"]  = tp_fp_label
		xls_df = xls_df.reset_index( drop = True )

		return xls_df


	def create_xl_dataset( self ):
		"""
		For each system, create the following three XL datasets:
			1. short linker XLs: containing TP XLs with Ca-ca distance
				<short_linker length specified in the config file.
			2. long linker XLs: containing TP XLs with Ca-ca distance
				>short_linker length and <= long_linker length
				specified in the config file.
			3. FP XLs: create a dataset comprising only FP Xls.
		"""
		frac_tp, frac_fp = self.dataset_configs.jwalk.frac_tp_fp
		max_xls = self.dataset_configs.jwalk.num_inter_xls

		for sys_name in self.jwalk_xls_dict:
			# Short linker XL dataset.
			#--------------------
			short_xls = self.xl_mixer(
				tp_xls_df = self.jwalk_xls_dict[sys_name]["short_xls"],
				fp_xls_df = self.jwalk_xls_dict[sys_name]["fp_xls"],
				frac_tp = frac_tp,
				frac_fp = frac_fp,
				max_xls = max_xls
			)

			# Long linker XL dataset.
			#--------------------
			long_xls = self.xl_mixer(
				tp_xls_df = self.jwalk_xls_dict[sys_name]["long_xls"],
				fp_xls_df = self.jwalk_xls_dict[sys_name]["fp_xls"],
				frac_tp = frac_tp,
				frac_fp = frac_fp,
				max_xls = max_xls
			)

			# FP XLs dataset.
			#--------------------
			fp_xls = self.xl_mixer(
				tp_xls_df = self.jwalk_xls_dict[sys_name]["short_xls"],
				fp_xls_df = self.jwalk_xls_dict[sys_name]["fp_xls"],
				frac_tp = 0.0,
				frac_fp = 1.0,
				max_xls = max_xls
			)

			self.xl_datasets[sys_name] = {
				"short_xls": short_xls,
				"long_xls": long_xls,
				"fp_xls": fp_xls
			}

	################################################################################
	################################################################################
	def get_sys_xl_file_paths( self, sys_name: str ) -> Tuple[str, str, str]:
		"""
		Get the file path for the .csv file containing
			XLs of the required xl_type.

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.

		Returns:
		----------
		short_xl_file: .csv file path for XLs with short linker.
		long_xl_file: .csv file path for XLs with long linker.
		fp_xl_file: .csv file path for FP XLs.
		"""
		short_xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = "short",
			frac_fp = self.dataset_configs.jwalk.frac_tp_fp[1]
			)
		long_xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = "long",
			frac_fp = self.dataset_configs.jwalk.frac_tp_fp[1]
			)
		# fp_xl_file = get_xl_file_path(
		# 	base_dir = self.base_dir,
		# 	benchmark_name = self.benchmark_name,
		# 	sys_name = sys_name,
		# 	xl_type = "fp",
		# 	frac_fp = self.dataset_configs.frac_tp_fp[1]
		# 	)
		return short_xl_file, long_xl_file #, fp_xl_file

	################################################################################
	def create_sys_dir( self, sys_name: str ) -> str:
		"""		
		Create and return the dir path for a given system.
		If the directory does not already exist, it is created anew.

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.

		Returns:
		----------
		sys_dir: absolute path to the system-specific data directory.
		"""
		sys_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name
		)
		os.makedirs( sys_dir, exist_ok = True )
		return sys_dir


	def save_struct_to_sys_dir( self, sys_name: str,
								sys_dir: str ):
		"""
		Copy the structure file for the given system to the sys dir.
		Depending on the configured structure format, one or more files
		may be copied:
			- pdb: copies <sys_name>.pdb
			- cif: copies <sys_name>.cif
			- both: copies both .pdb and .cif files

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		sys_dir: absolute path to the system-specific data directory.
		"""
		struct_format = self.dataset_configs.globals.struct_format
		struct_format = ["pdb", "cif"] if struct_format == "both" else [struct_format]
		for ext in struct_format:
			struct_file_src = os.path.join( self.pdb_struct_dir,
										f"{sys_name}.{ext}" )
			dest = sys_dir
			cmd = ["cp", struct_file_src, dest]
			run_subprocess( cmd )

	################################################################################
	def get_entities( self, sys_name: str ) -> List[Dict]:
		"""
		Extract entity-level sequence information for a given system.
		Create a list of entities associated part of the given system.
		Create all entities part of the system. Have the following:
			entity_id (starts from 1).
			copy_num --> no. of copies for an entity.
			start, end UniProt residue positions.
			Complete UniProt sequence.
		For homomers, residue numbering and sequence information are
			taken from the first available chain, assuming that all
			copies are identical.

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.

		Returns:
		----------
		entities: a list of entity dicts for a given system containing
			the input metadata.
			The dict follows the structure:
			{
				entity_id
				copy_num
				start
				end
				sequence
			}
		"""
		entities = []

		for entity_id in self.seqres_dict[sys_name]:
			auth_asym_ids = list( self.seqres_dict[sys_name][entity_id].keys() )
			copy_num = len( auth_asym_ids )

			# For homomers, consider the 1st chain for getting Uniprot positions.
			chain_id = auth_asym_ids[0]
			start = self.seqres_dict[sys_name][entity_id][chain_id]["start_seq_id"]
			end = self.seqres_dict[sys_name][entity_id][chain_id]["end_seq_id"]
			seq = self.seqres_dict[sys_name][entity_id][chain_id]["seq"]

			if len( seq ) != ( end - start + 1 ):
				raise ValueError( f"Sequence length ({len( seq )}) " +
								f"does not match the residue numbering ({start}-{end}..." )

			entities.append(
				{
					# "name": sys_name,
					"entity_id": int( entity_id ),
					"copy_num": copy_num,
					"start": start,
					"end": end,
					"sequence": seq
				}
			 )

		return entities


	def create_sys_config_dict(
		self,
		sys_name: str,
		entities: List[Dict]
		) -> Dict[str, Any]:
		"""
		Create a config dict for the givrn system. It contains input
			metadata and XL restraint info.
		The dict follows the structure:
			{
				"name",
				"entity": {
					{},
					{}
				},
				"short_xl_restraint": {},
					restraint info for XLs with short linker length.
				"long_xl_restraint": {},
					restraint info for XLs with long linker length.
				"fp_xl_restraint": {}
					restraint info for FP XLs.
			}

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		entities: a list of entity dicts for a given system containing
			the input metadata.
		xl_file: path for the XLs .csv file.

		Returns:
		----------
		sys_dict: dict containing the configs for the system to be modeled.
		"""
		short_xl_file, long_xl_file = self.get_sys_xl_file_paths(
			sys_name = sys_name
		)
		sys_dict = {
		# f"System_{sys_name}": {
			"name": sys_name,
			"entity": entities,
			"short_xl_restraint": {
				"xl_max_bound": self.dataset_configs.jwalk.short_linker,
				"file_name": short_xl_file,
			},
			"long_xl_restraint": {
				"xl_max_bound": self.dataset_configs.jwalk.long_linker,
				"file_name": long_xl_file,
			},
			# "fp_xl_restraint": {
			# 	# FP XLs are evaluated at short_linker length.
			# 	"xl_max_bound": self.dataset_configs.jwalk.short_linker,
			# 	"file_name": fp_xl_file,
			# }
		}
		# }
		return sys_dict


	def save_sys_config_dict( self, sys_name: str ):
		"""
		For the given system, create config file contaiing
			input metadata and XL restraint info.
		Save in the system-specifc dir as a .json file.

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		sys_dir: absolute path to the system-specific data directory.
		"""
		entities = self.get_entities( sys_name = sys_name )
		sys_conf_dict = self.create_sys_config_dict(
			sys_name = sys_name,
			entities = entities )
		sys_conf_file = get_sys_config_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
		)
		write_json( sys_conf_dict, sys_conf_file )

	################################################################################
	def save_xls_in_sys_dir(
		self,
		sys_name: str ):
		"""
		For the given system, save XLs for all xl types in the
			system-specific dir.
		We assume that the system dir alreadu exists.

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		"""
		xl_file_paths = self.get_sys_xl_file_paths( sys_name = sys_name )
		xl_types = ( "short_xls", "long_xls")
		# xl_types = ( "short_xls", "long_xls", "fp_xls" )
		for xl_type, xl_file in zip( xl_types, xl_file_paths ):
			xl_df = self.xl_datasets[sys_name][xl_type]
			xl_df.to_csv( xl_file, index = False )

	################################################################################
	def create_system_inputs_for_modeling( self ):
		"""
		For all the complexes included in the benchmark:
			Create a dir for storing system specific fiels.
			Copy the structure file to the system dir.
			Save XLs for all xl types to the system dir.
			Create and save config file containing input
				metadata and restraint info.
		We use the PDB ID as the system name.
		"""
		for sys_name in self.benchmark_pdb_ids:
			sys_dir = self.create_sys_dir( sys_name = sys_name )
			self.save_struct_to_sys_dir(
				sys_name = sys_name,
				sys_dir = sys_dir )
			self.save_xls_in_sys_dir( sys_name = sys_name )
			self.save_sys_config_dict( sys_name = sys_name )

	################################################################################
	################################################################################
	def write_benchmark_csv( self ):
		"""
		Write the following info for the benchmark to a .csv file:
			PDB ID
			Polymer entity
			Auth Asym ID
			Stoichiometry
			Residue positions
			Total length
			Total Short linker XLs
			Total Long linker XLs
			Total FP XLs
			Selected Short linker XLs
			Selected Long linker XLs
			Selected FP XLs
		"""
		flat_dict = {k:[] for k in [
			"PDB ID", "Polymer entity",
			"Auth Asym ID",
			"Stoichiometry",
			"Residue positions",
			"Total length",
			"Total short linker XLs",
			"Total Long linker XLs",
			"Total FP XLs",
			"Selected short linker XLs",
			"Selected long linker XLs",
			"Selected FP XLs"
			]}
		for sys_name in self.benchmark_pdb_ids:
			entity_ids = list( self.seqres_dict[sys_name].keys() )
			auth_asym_ids = []
			stoichiometry = []
			pos = []
			total_length = 0
			for entity_id in entity_ids:
				aa_ids = list( self.seqres_dict[sys_name][entity_id].keys() )
				auth_asym_ids.append( ":".join( aa_ids ) )
				stoichiometry.append( f"{len( aa_ids )}" )
				start = self.seqres_dict[sys_name][entity_id][aa_ids[0]]["start_seq_id"]
				end = self.seqres_dict[sys_name][entity_id][aa_ids[0]]["end_seq_id"]
				pos.append( f"{start}-{end}" )

				for aa_id in aa_ids:
					seq = self.seqres_dict[sys_name][entity_id][aa_id]["seq"]
					total_length += len( seq )

			flat_dict["PDB ID"].append( sys_name )
			flat_dict["Polymer entity"].append( ",".join( map( str, entity_ids ) ) )
			flat_dict["Auth Asym ID"].append( ",".join( auth_asym_ids ) )
			flat_dict["Stoichiometry"].append( ",".join( stoichiometry ) )
			flat_dict["Residue positions"].append( ",".join( pos ) )
			flat_dict["Total length"].append( total_length )
			# Log the total no. of XLs.
			flat_dict["Total short linker XLs"].append(
				self.jwalk_xls_dict[sys_name]["short_xls"].shape[0]
			)
			flat_dict["Total Long linker XLs"].append(
				self.jwalk_xls_dict[sys_name]["long_xls"].shape[0]
			)
			flat_dict["Total FP XLs"].append(
				self.jwalk_xls_dict[sys_name]["fp_xls"].shape[0]
			)
			# Log the no. of XLs that were selected.
			flat_dict["Selected short linker XLs"].append(
				self.xl_datasets[sys_name]["short_xls"].shape[0]
			)
			flat_dict["Selected long linker XLs"].append(
				self.xl_datasets[sys_name]["long_xls"].shape[0]
			)
			flat_dict["Selected FP XLs"].append(
				self.xl_datasets[sys_name]["fp_xls"].shape[0]
			)

		df = pd.DataFrame( flat_dict )
		file = os.path.join( self.meta_dir,
							f"{self.benchmark_name}_benchmark.csv" )
		df.to_csv( file, index = False )


if __name__ == "__main__":
	CreateBenchmark().forward()
