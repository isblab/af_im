"""
Contains module to obtain metadata for benchmark dataset creation.
1. PDB IDs from any benchmark (see below).
2. We need the following details about the complex:
	PDB structure (both .pdb and .cif).
		We want the SEQRES sequence.
		PDB residue numbers.
3. Obtain data: need support for both simulated and real.
	Simulated
		XLs -> JWalk
"""
from typing import List
import os, glob, copy, time, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from config import get_config_dict
from utils.utils import ( open_file_handler,
							read_json, write_json,
							write_configdict_to_json )
from utils.api_utils import PdbRestApi
from utils.mappings import (
	create_pdb_num_to_seq_id_mapping,
	map_xls_to_seq_id
	)
from api_data_modules import( DownloadPdbStructure,
								SeqResDict )

from simulate_data import ( SimulateCrosslinks ) 


class Metadata():
	"""
	Obtain all required metadata for the benchmark dataset.
	"""
	def __init__( self ):
		self.config_dict = get_config_dict()
		self.dataset_configs = self.config_dict.benchmark
		self.benchmark_name = self.dataset_configs.globals.benchmark_name
		np.random.seed( self.config_dict.prng_seed )

		self.seqres_dict = {}
		# Dict containing mapping  between pdb_seq_num and seq_id.
		self.pdb_num_seq_id_map = {}
		# Dict to store inter-protein XLs.
		self.xls_dict = {}

		self.logs = {}


	def forward( self ):
		"""
		Note: True positive (TP/tp); False positive (FP/fp); Cross-link (XL)
		"""
		self.create_required_paths()
		self.create_required_dir()

		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )

		self.get_pdb_ids()

		self.run_dataset_creation_pipeline()
		self.save_dataset_configs()
		self.write_logs_to_csv()
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
		self.base_dir = os.path.join(
			os.path.abspath( self.dataset_configs.globals.base_dir )
			)
		# Benchmark specific dir.
		self.benchmark_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_benchmark" )
		# Dir to store benchmark metadata including PDB API files,
		# 	structure file, SIFTS mapping, benchamrk csv and the required intermediate files.
		self.meta_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_metadata" )

		# Dict mapping PDB ID to its respective benchmark.
		self.pdb_benchmark_map_file = os.path.join( self.meta_dir, "pdb_benchmark_mapping.json" )
		# PDB IDs remaining after metadta collection.
		self.benchmark_pdbs_file =  os.path.join( self.meta_dir,
										f"{self.benchmark_name}_benchmark_pdb_ids.txt" )
		# Logs file path.
		self.logs_file = os.path.join( self.meta_dir, f"Logs_{self.benchmark_name}.json" )
		self.logs_csv_file = os.path.join( self.meta_dir, f"Logs_{self.benchmark_name}.csv" )
		self.dataset_configs_file = os.path.join( self.meta_dir,
								f"Dataset_configs_{self.benchmark_name}.json" )

		## --------------------------
		# For DownloadPdbStructure module
		## --------------------------
		# Dir contaiing PDB structure.
		self.pdb_struct_dir = os.path.join( self.meta_dir, "struct" )
		self.pdb_struct_dwnld_file = os.path.join( self.meta_dir,
												"pdb_struct_dwnld.txt" )

		## --------------------------
		# For SeqResDict module
		## --------------------------
		self.seqres_dict_file = os.path.join( self.meta_dir, "seqres_dict.npy" )
		self.resolution_dict_file = os.path.join( self.meta_dir, "resolution_dict.json" )
		self.pdb_num_seq_id_map_file = os.path.join( self.meta_dir, "pdb_num_seq_id_map.npy" )

		## --------------------------
		# For Simulated data
		## --------------------------
		self.jwalk_output_dir = os.path.join( self.meta_dir, "jwalk_output/" )
		self.xls_dict_file = os.path.join( self.meta_dir, "jwalk_xls.npy" )



	def create_required_dir( self ):
		"""
		Create the required directories if not already existing.
		"""
		for name in [self.base_dir, self.meta_dir,
			self.benchmark_dir, self.pdb_struct_dir,
			self.jwalk_output_dir
		]:
			os.makedirs( name, exist_ok = True )

	################################################################################
	################################################################################
	def get_pdb_ids( self ):
		"""
		Obtain the PDB IDs from the input files of the required benchmark.
		"""
		if self.benchmark_name == "crosslink":
			self.benchmark_pdb_ids_list = self.get_pdb_ids_for_xl_benchmark()
		else:
			raise ValueError( "Unsupported benchmark specified..." )


	def get_pdb_ids_for_xl_benchmark( self ) -> List[str]:
		"""
		For the merged simulated XL benchmark, obtain PDB IDs from:
			PDB benchmark from AFUnmasked
			Protein-protein and protein-peptide benchmark from FoldBench
			SAbDab database
			Antigen-Antibody benchmark from FoldBench
			PINDER
			AFM benchmark
		"""
		afu = self.parse_pdb_afu_benchmark()
		print( f"PDB IDs from AF Unmasked PDB benchmark: {len( afu )}" )
		foldbench = self.parse_foldbench_benchmark()
		print( f"PDB IDs from FoldBench benchmark: {len( foldbench )}" )
		sabdab = self.parse_sabdab_benchmark()
		print( f"PDB IDs from SAbDab benchmark: {len( sabdab )}" )
		pinder = self.parse_pinder_benchmark()
		print( f"PDB IDs from PINDER benchmark: {len( pinder )}" )
		afmb = self.parse_afmb_benchmark()
		print( f"PDB IDs from AFM benchmark: {len( afmb )}" )

		# Remove duplicate PDB IDs and sort.
		pdb_ids = sorted(
			list(
				set( afu + foldbench + sabdab + pinder + afmb )
			)
		)
		return pdb_ids


	def parse_pdb_afu_benchmark( self ) -> List[str]:
		"""
		Using the PDB benchmark provided in the AF Unmasked paper.
		"""
		fh = open_file_handler( self.dataset_configs.datasets.afu, "r" )
		pdb_ids = fh.readlines()[0].strip().split( "," )
		fh.close()
		pdb_ids = [id_.lower() for id_ in pdb_ids]

		return pdb_ids


	def parse_sabdab_benchmark( self ) -> List[str]:
		"""
		A .tsv file was download from SAbDab database
			(https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab)
			using the following filters:
				Non-redundant at 60% sequence identity.
				In complex: bound-only
				resolution cutoff: 3.0
		Select entries for which the antigen type is protein/peptide.
		"""
		df = pd.read_csv( self.dataset_configs.datasets.sabdab, sep = "\t" )
		groups = df.groupby( ["pdb"] )

		pdb_ids = []
		for group in groups:
			pdb = group[0]
			# Must have atleast 1 protein/peptide antigen.
			ag_type = set( group[1]["antigen_type"]
				).intersection( set( ["protein", "peptide"] ) )
			if len( ag_type ) != 0:
				pdb_ids.append( pdb[0].lower() )
		return pdb_ids


	def parse_foldbench_benchmark( self ) -> List[str]:
		"""
		.csv file was obtained from https://github.com/BEAM-Labs/FoldBench.git
		Will parse the following as specified:
			Protein-protein benchmark.
			Protein-peptide benchmark.
			Antigen-antibody benchmark.
		"""
		prot_prot = pd.read_csv( self.dataset_configs.datasets.foldbench_prot_prot )
		prot_pep = pd.read_csv( self.dataset_configs.datasets.foldbench_prot_pep )
		abag = pd.read_csv( self.dataset_configs.datasets.foldbench_abag )

		pdb_ids = prot_prot["pdb_id"].str.split( "-" ).str[0].tolist()
		pdb_ids += prot_pep["pdb_id"].str.split( "-" ).str[0].tolist()
		pdb_ids += abag["pdb_id"].str.split( "-" ).str[0].tolist()

		return pdb_ids


	def parse_pinder_benchmark( self ) -> List[str]:
		"""
		PINDER dataset was obtained from the PINDER package as
			specified in #Issue41.
			(https://github.com/pinder-org/pinder)
		Parse the PINDER dataset and return the PDB IDs.
		"""
		f = open_file_handler( self.dataset_configs.datasets.pinder_xl, "r" )
		pdb_ids = f.readlines()[0].split( "," )
		return pdb_ids


	def parse_afmb_benchmark( self ) -> List[str]:
		"""
		afmb --> AFM benchmark was obtained from the following repository:
		https://gitlab.com/ElofssonLab/afm-benchmark.git
		We parse all the heter- and homo-mer complexes in the benchmark.
		Each file contains entries separated by a '\n' character.
		"""
		pdb_ids = []
		for complex_type in ["homo", "hete"]:
			for stoic in [2, 3, 4, 5, 6]:
				file_name = f"ID_{stoic}mer_{complex_type}.csv"
				file_path = os.path.join( self.dataset_configs.datasets.afmb, file_name )
		f = open_file_handler( file_path, "r" )
		for line in f.readlines():
			pdb_ids.append( line.strip().lower() )
		return pdb_ids

	################################################################################
	################################################################################
	def run_dataset_creation_pipeline( self ):
		"""
		Pipeline all modules for benchamrk dataset creation.
		Get .pdb and .cif structures for all complexes.
		Get data - simulated or real.
		"""
		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "-------- Download PDB Structures --------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_download_pdb_structure_module()

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- MMCIF Dict ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.run_seqres_dict_module()
		print( f"PDB IDs remaining: {len( self.benchmark_pdb_ids_list )}" )

		print( "\n" + "".join( ["-" for i in range( 40 )] ) )
		print(  "---------- Simulated Data ----------" )
		print( "".join( ["-" for i in range( 40 )] ) + "\n" )
		self.simulate_experimental_data()

		w = open_file_handler( self.benchmark_pdbs_file, "w" )
		w.writelines( ",".join( list( self.xls_dict.keys() ) ) )
		w.close()

	################################################################################
	################################################################################
	def run_download_pdb_structure_module( self ):
		"""
		Download .pdb and .cif structures for all complexes.
		"""
		warnings.filterwarnings( "ignore" ) 
		if os.path.exists( self.pdb_struct_dwnld_file ):
			print( "Structures for benchmark already downloaded..." )
			f = open_file_handler( self.pdb_struct_dwnld_file, "r" )
			self.benchmark_pdb_ids_list = f.readlines()[0].split( "," )
			f.close()

		else:
			obj = DownloadPdbStructure(
				pdb_ids_list = self.benchmark_pdb_ids_list,
				pdb_struct_dir = self.pdb_struct_dir,
				struct_format = self.dataset_configs.globals.struct_format,
				download_assembly = self.dataset_configs.globals.download_assembly,
				cores = self.dataset_configs.globals.cores,
				max_trials = self.dataset_configs.globals.max_trials,
				wait_time = self.dataset_configs.globals.wait_time
				)
			obj.forward()

			self.benchmark_pdb_ids_list = copy.copy( obj.downloaded_struct )
			self.logs["DownloadPdbStructure"] = copy.deepcopy( obj.struct_logs )

			del obj
			
			w = open_file_handler( self.pdb_struct_dwnld_file, "w" )
			w.writelines( ",".join( self.benchmark_pdb_ids_list ) )
			w.close()
			write_json( self.logs, self.logs_file )

	################################################################################
	################################################################################
	def run_seqres_dict_module( self ):
		"""
		Obtain the cif dict containing entity_id, chain_id, seq,
			and res no. for all entries.
		"""
		if os.path.exists( self.seqres_dict_file ):
			print( "CIF dict already present..." )
			self.seqres_dict = np.load( self.seqres_dict_file,
										allow_pickle = True ).item()
			self.pdb_num_seq_id_map = np.load( self.pdb_num_seq_id_map_file,
										allow_pickle = True ).item()

		else:
			obj = SeqResDict(
				pdb_ids_list = self.benchmark_pdb_ids_list,
				pdb_struct_dir = self.pdb_struct_dir,
				cores = self.dataset_configs.globals.cores,
				max_sys_length = self.dataset_configs.globals.max_sys_length,
				frac_coverage = self.dataset_configs.globals.frac_coverage
				)
			obj.forward()

			self.seqres_dict = copy.deepcopy( obj.seqres_dict )
			self.resolution_dict = copy.deepcopy( obj.resolution_dict )
			self.logs["SeqResDict"] = copy.deepcopy( obj.cif_logs )

			if self.dataset_configs.globals.download_assembly:
				self.update_resolution_dict()

			del obj

			# self.create_pdb_num_to_seq_id_mapping()
			self.pdb_num_seq_id_map = create_pdb_num_to_seq_id_mapping(
				seqres_dict = self.seqres_dict
			)
			
			np.save( self.seqres_dict_file,
					self.seqres_dict,
					allow_pickle = True )
			np.save( self.pdb_num_seq_id_map_file,
					self.pdb_num_seq_id_map,
					allow_pickle = True )
			write_json( self.resolution_dict, self.resolution_dict_file )

			write_json( self.logs, self.logs_file )

		self.benchmark_pdb_ids_list = list( self.seqres_dict.keys() )


	def update_resolution_dict( self ):
		"""
		When downloading the biological assembly, the cif file
			does not contain the resolution of the structure.
		So we need to get the resolutions for all separately from the PDB REST API.
		"""
		print( "Fetching the resolution for biological assemblies..." )
		def get_resolution( entry_id: str ):
			"""
			Fetch the resolution for the given entry_id from the PDB REST API.
			"""
			rest = PdbRestApi( entry_id = entry_id )
			entry_data = rest.entry_data
			if entry_data == None:
				# Sanity check: at this stage the PDB entry exists so
				# 	 the REST API must return the entry details.
				raise ValueError( f"Could not fetch data from the PDB REST API for {entry_id}..." )
			if "resolution_combined" in entry_data["rcsb_entry_info"]:
				resolution = entry_data["rcsb_entry_info"]["resolution_combined"]
			else:
				resolution = 0.0
			return entry_id, resolution

		with ThreadPoolExecutor( self.dataset_configs.globals.cores ) as executor:
			futures = [
				executor.submit(get_resolution, entry_id)
				for entry_id in self.resolution_dict.keys()
			]
			# future = executor.submit( get_resolution, self.resolution_dict.keys() )
			for future in futures:
				entry_id, resolution = future.result()
				self.resolution_dict[entry_id] = resolution


	# def create_pdb_num_to_seq_id_mapping( self ):
	# 	"""
	# 	Craete a mapping between the seq_id and pdb_seq_num obtained
	# 		from the .cif file.
	# 	We map the pdb_seq_num to seq_id.
	# 		This is because pdb_seq_num may be discontinous in some cases
	# 			(8g0q_B, 8g0q_D) however, seq_id is always continous.
	# 	pdb_id: {
	# 		"chain_id": dict( zip( pdb_seq_num, seq_id ) )
	# 	}
	# 	"""
	# 	for pdb_id in self.seqres_dict:
	# 		self.pdb_num_seq_id_map[pdb_id] = {}
	# 		for entity_id in self.seqres_dict[pdb_id]:
	# 			for chain_id in self.seqres_dict[pdb_id][entity_id]:
	# 				chain = self.seqres_dict[pdb_id][entity_id][chain_id]
	# 				start_seq_id = chain["start_seq_id"]
	# 				end_seq_id = chain["end_seq_id"]

	# 				seq_id = chain["seq_id"]
	# 				pdb_seq_num = chain["res_num"]
	# 				seq = chain["seq"]

	# 				if ( end_seq_id-start_seq_id+1 ) != len( seq ):
	# 					raise ValueError( f"Mismatch in length of seq_id and sequence for PDB: {pdb_id}..." )
	# 				if len( seq_id ) != len( pdb_seq_num ):
	# 					raise ValueError( f"Mismatch in length of seq_id and pdb_seq_num for PDB: {pdb_id}..." )
	# 				self.pdb_num_seq_id_map[pdb_id][chain_id] = dict( zip( pdb_seq_num, seq_id ) )

	################################################################################
	################################################################################
	def simulate_experimental_data( self ):
		"""
		Simulate experimental dat for the required protein complexes.
		Currently supporting XL data (JWalk).
		"""
		# if self.dataset_configs["jwalk"]["enabled"]:
		self.simulate_xls()


	def simulate_xls( self ):
		"""
		Simulate XL data using JWalk.
		Obtain inter-protein XLs.
		Map residue numbers to UniProt numbering based on SIFTS mapping.
		"""
		if os.path.exists( self.xls_dict_file ):
			print( "JWalk simulated XLs already exist..." )
			self.xls_dict = np.load( self.xls_dict_file,
							allow_pickle = True ).item()
		else:
			obj = SimulateCrosslinks(
				jwalk_exec = self.dataset_configs.jwalk.jwalk_exec,
				pdb_ids_list = self.benchmark_pdb_ids_list,
				pdb_struct_dir = self.pdb_struct_dir,
				jwalk_dir = self.jwalk_output_dir,
				struct_format = "pdb",
				short_linker = self.dataset_configs.jwalk.short_linker,
				long_linker = self.dataset_configs.jwalk.long_linker,
				num_xls = self.dataset_configs.jwalk.num_inter_xls,
				cores = self.dataset_configs.globals.cores,
				)
			obj.forward()

			self.xls_dict = copy.deepcopy( obj.xls_dict )
			self.logs["Jwalk"] = copy.deepcopy( obj.jwalk_logs )

			del obj

			np.save( self.xls_dict_file,
					self.xls_dict,
					allow_pickle = True )

		self.map_xls_to_seq_id()
		np.save( self.xls_dict_file,
				self.xls_dict,
				allow_pickle = True )
		write_json( self.logs, self.logs_file )


	def map_xls_to_seq_id( self ):
		"""
		Given a dataframe for inter-protein XLs, map the pdb_seq_num
			to the corresponding seq_id.
		Columns: Protein1, Residue1, Protein1, Residue1
		"""
		drop_pdb = []
		for pdb_id in self.xls_dict:
			for xl_type in ["short_xls", "long_xls", "fp_xls"]:
				# df = self.xls_dict[pdb_id][xl_type]
				df = map_xls_to_seq_id(
					xl_df = self.xls_dict[pdb_id][xl_type],
					pdb_num_seq_id_map = self.pdb_num_seq_id_map[pdb_id]
				)
				# drop_index = []
				# for i in df.index:
				# 	chain1 = df.iloc[i, 0]
				# 	res1 = int( df.iloc[i, 1] )
				# 	chain2 = df.iloc[i, 2]
				# 	res2 = int( df.iloc[i, 3] )

				# 	if chain1 not in self.pdb_num_seq_id_map[pdb_id]:
				# 		drop_index.append( i )
				# 		continue
				# 	if chain2 not in self.pdb_num_seq_id_map[pdb_id]:
				# 		drop_index.append( i )
				# 		continue
				# 	chain1_map = self.pdb_num_seq_id_map[pdb_id][chain1]
				# 	chain2_map = self.pdb_num_seq_id_map[pdb_id][chain2]

				# 	# Ignore XLs for residues not in pdb_seq_num
				# 	# 	(not selected for modeling).
				# 	if res1 not in chain1_map:
				# 		drop_index.append( i )
				# 		continue
				# 	if res2 not in chain2_map:
				# 		drop_index.append( i )
				# 		continue

				# 	df.iloc[i, 1] = chain1_map[res1]
				# 	df.iloc[i, 3] = chain2_map[res2]
				# df = df.drop( drop_index )
				if len( df ) == 0:
					drop_pdb.append( pdb_id )
					break
				df = df.reset_index( drop = True )
		# Remove PDB IDs for which no inter-protein XL was obtained for the selected chains.
		for pdb_id in drop_pdb:
			del self.xls_dict[pdb_id]


	def save_dataset_configs( self ):
		"""
		Save the datset configs to a JSON file on disk.
		"""
		write_configdict_to_json( self.dataset_configs, self.dataset_configs_file )


	def write_logs_to_csv( self ):
		"""
		Write the logs dict to a csv file.
		"""
		flat_dict = {k:[] for k in ["Module",
									"Description",
									"Count"]}

		for module in self.logs:
			for desc in self.logs[module]:
				desc_val = self.logs[module][desc]
				flat_dict["Module"].append( module )
				flat_dict["Description"].append( desc )
				if isinstance( desc_val, List ):

					val = desc_val[1]
				else:
					val = desc_val
				flat_dict["Count"].append( val )
			[flat_dict[k].append( "" ) for k in flat_dict]
		df = pd.DataFrame( flat_dict )
		df.to_csv( self.logs_csv_file, index = False )


if __name__ == "__main__":
	Metadata().forward()
