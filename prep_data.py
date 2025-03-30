"""
Script to obtain and create the required files
	for the benchmark dataset.
"""
import os
import glob
import copy
from typing import List, Dict, Tuple
from multiprocessing import Pool
from functools import partial
import pandas as pd
import tqdm

from utils import ( run_subprocess, open_file_handler,
					read_json, write_json )
from api_utils import ( PdbRestApi, get_uniprot_seq, download_pdb,
						download_sifts_mapping, parse_sifts_xml )

# Using CASP15 dataset as our benchmark.
# 	targetlist.csv for CASP15 must be present in ./raw/.
# Shifted to AF Unmasked PDB benchmark.
# 	.txt file containing pdb_ids must be present in ./raw/.

class CreateBenchmark():
	"""
	Create input for the benchmark dataset.
	"""
	def __init__( self ):
		self.jwalk_exec = "jwalk"
		self.xl_length = 35

		# self.casp_input_csv = "./raw/targetlist_mod.csv"
		# PDB benchmark from AF Unmasked paper.
		self.afu_pdb_benchmark = "./raw/af_unmasked_pdb_benchmark.txt"
		self.bm_v5_5 = "./raw/Table_BM5.5.xlsx"
		self.base_dir = os.path.join( "./benchmark/" )
		self.benchmark_dir = os.path.join( self.base_dir, "imp_dl_benchmark/" )
		self.meta_dir = os.path.join( self.base_dir, "metadata" )
		self.pdb_api_dir = os.path.join( self.meta_dir, "pdb_api" )
		self.pdb_struct_dir = os.path.join( self.meta_dir, "pdb_struct" )
		self.sifts_xml_dir = os.path.join( self.meta_dir, "sifts_xml" )
		self.sifts_dict_dir = os.path.join( self.meta_dir, "sifts_dict" )


		self.pdb_benchmark_dict = {}
		self.sifts_pdb_to_uni = {}
		self.uni_seq_dict = {}
		self.uni_seq_file = os.path.join( self.base_dir, "uni_seq.json" )
		self.benchmark_config_dict = {}
		self.benchmark_csv = os.path.join( self.base_dir, "benchmark.csv" )

		self.initialize_dir()


	def forward( self ):
		"""
		For our benchmark we consider only protein multimer entries.
		For each entry we need the:
			Sequence of the constituent proteins.
			Structure file.
			PDB to UniProt mapping.
			Stoichiometry.
		"""
		# get PDB IDs from the two benchmarks.
		afu_benchmark = self.parse_pdb_afu_benchmark()
		# bm_benchmark = self.parse_bm_benchmark()

		# benchmark_pdb_ids = afu_benchmark + bm_benchmark
		benchmark_pdb_ids = afu_benchmark
		print( f"Total PDB IDs obtained: {len( benchmark_pdb_ids )}" )

		print( "\n------------------------------------------------" )
		print( "Downloading info from PDB REST API...\n" )
		self.where_the_magic_happens( benchmark_pdb_ids )

		print( "\n------------------------------------------------" )
		print( "Filter and Segregate the PDB IDs...\n" )
		selected_pdb_ids = self.filter_and_segregate()

		print( "\n------------------------------------------------" )
		print( "Remove the unwanted PDB IDs...\n" )
		self.clean_benchmark_dict( selected_pdb_ids )

		print( "\n------------------------------------------------" )
		print( "Download UniProt sequences...\n" )
		if not os.path.exists( self.uni_seq_file ):
			self.dwnld_uni_seq( selected_pdb_ids )
		else:
			self.uni_seq_dict = read_json( self.uni_seq_file )

		print( "\n------------------------------------------------" )
		print( "Download the structure from PDB in .pdb format...\n" )
		self.dwnld_pdb_struct( selected_pdb_ids )

		print( "\n------------------------------------------------" )
		print( "Obtain PDB-UniProt mapping using SIFTS...\n" )
		self.map_pdb_to_uniprot( selected_pdb_ids )

		print( "\n------------------------------------------------" )
		print( "Create config files...\n" )
		self.create_sys_config_dict()

		print( "\n------------------------------------------------" )
		print( "Simulate XL data...\n" )
		num_xls = self.simulate_xl_data()

		print( "\n------------------------------------------------" )
		print( "Save benchmark to disk..." )
		self.write_benchmark_to_csv( num_xls )



	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def initialize_dir( self ):
		"""
		Create the required directories if not already existing.
		"""
		os.makedirs( self.base_dir, exist_ok = True )
		os.makedirs( self.meta_dir, exist_ok = True )
		os.makedirs( self.pdb_api_dir, exist_ok = True )
		os.makedirs( self.pdb_struct_dir, exist_ok = True )
		os.makedirs( self.benchmark_dir, exist_ok = True )
		os.makedirs( self.sifts_xml_dir, exist_ok = True )
		os.makedirs( self.sifts_dict_dir, exist_ok = True )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def parse_pdb_afu_benchmark( self ) -> List:
		"""
		Using the PDB benchmark provided in the AF Unmasked paper.
		"""
		fh = open_file_handler( self.afu_pdb_benchmark, "r" )
		pdb_ids = fh.readlines()[0].strip().split( "," )
		fh.close()

		print( f"PDB IDs from AF Unmasked PDB benchmark: {len( pdb_ids )}" )
		return pdb_ids


	def parse_bm_benchmark( self ) -> List:
		"""
		Using the docking benchmark v_5.5 (https://zlab.wenglab.org/benchmark/).
		Selecting only medium and difficult complexes from here.
		PDB IDs are present as: '1AHW_AB:C', '1DQJ_AB:C'
		We want --> '1ahw', '1dqj'
		"""
		pdb_ids = []
		df = pd.read_excel( self.bm_v5_5 )
		complexes = df["Complex"].tolist()
		start_idx = complexes.index( "Medium Difficulty (60)" )

		for cplex in complexes[start_idx+1:]:
			if "Difficult" in cplex:
				continue
			pdb_id = cplex.split( "_" )[0].lower()
			pdb_ids.append( pdb_id )
		print( f"PDB IDs from BM v_5.5 benchmark: {len( pdb_ids )}" )
		return pdb_ids


	def get_entry_entity_files( self, entry_id: str ) -> Tuple[str, str]:
		"""
		Return the file paths for the entry and entity dict.
		"""
		entry_file = os.path.join( self.pdb_api_dir, f"entry_dict_{entry_id}.json" )
		entity_file = os.path.join( self.pdb_api_dir, f"entity_dict_{entry_id}.json" )
		return entry_file, entity_file


	def save_entry_entity_dict( self, rest_api: PdbRestApi,
								entry_file: str,
								entity_file: str ):
		"""
		Save the entry and entity dict on disk.
		Assuming the entity dict contains info. for all entities to be saved.
		"""
		if not os.path.exists( entry_file ):
			write_json( rest_api.entry_data, entry_file )
		if not os.path.exists( entity_file ):
			write_json( rest_api.entity_data, entity_file )


	def instantiate_pdb_rest_api( self, entry_id: str,
									entry_file: str,
									entity_file: str ) -> PdbRestApi:

		"""
		Instantiate the PDB REST API object.
		"""
		rest_api = PdbRestApi( entry_id = entry_id,
								entry_file = entry_file,
								entity_file = entity_file )
		return rest_api


	def entry_from_pdb_rest_api( self, entry_id: str ):
		"""
		Given a PDB ID, fetch the following info. from the PDB REST API:
			All entities.
			All polymer entities.
			UniProt IDs.
			Auth asym IDs.
			Stoichiometry.
			Aligned UniProt start-end position.
			Aligned PDB start-end position.
		entry_id corresponds to the PDB ID.
		"""
		entry_file, entity_file = self.get_entry_entity_files( entry_id )

		# Instantiate the PDB REST API object.
		rest_api = self.instantiate_pdb_rest_api( entry_id,
													entry_file,
													entity_file )
		# Get all entity IDs in the.
		all_entities = rest_api.get_all_entities()
		# Get all polymer entity IDs in the.
		polymer_entities = rest_api.get_polymer_entities()

		pdb_dict = {k:[] for k in [
								"entity_ids", "polymer_entity_ids", "uniprot_ids",
								"stoichiometry", "uni_pos", "pdb_pos"]}
		stoichiometry = []
		uniprot_ids = []
		auth_asym_ids = []
		uni_pos, pdb_pos = [], []
		total_length = 0
		for entity_id in polymer_entities:
			rest_api.retrieve_polymer_entity_data( entity_id )
			uniprot_ids.extend( rest_api.get_uniprot_ids_for_entity( entity_id ) )
			asym_ids = rest_api.get_asym_ids_for_entity( entity_id )
			auth_asym_ids.append( "-".join( rest_api.get_auth_asym_ids_for_entity( entity_id ) ) )
			pos, length = rest_api.get_poly_entity_align( entity_id )
			uni_pos.append( pos[0] )
			pdb_pos.append( pos[0] )
			total_length += length

			stoichiometry.append( f"{len( asym_ids )}" )

		pdb_dict["entity_ids"] = ",".join( all_entities )
		pdb_dict["polymer_entity_ids"] = ",".join( polymer_entities )
		pdb_dict["uniprot_ids"] = ",".join( uniprot_ids )
		pdb_dict["auth_asym_ids"] = ",".join( auth_asym_ids )
		pdb_dict["stoichiometry"] = "-".join( stoichiometry )
		pdb_dict["uni_pos"] = ",".join( uni_pos )
		pdb_dict["pdb_pos"] = ",".join( pdb_pos )
		pdb_dict["total_length"] = total_length

		self.save_entry_entity_dict( rest_api,
									entry_file,
									entity_file )
		return entry_id, pdb_dict


	def where_the_magic_happens( self, benchamrk_pdb_ids: List ):
		"""
		For all the benchmark PDB IDs, get the required info.
		(See self.entry_from_pdb_rest_api doc-str)
		"""
		for idx, pdb_id in enumerate( benchamrk_pdb_ids ):
			# Skip obsolete PDB IDs - 8h4x.
			# Skip PDBs with insertion code - 8a82, 7yls
			if pdb_id in ["8h4x", "8a82", "7yls"]:
				continue
			print( f"{idx} --> {pdb_id}" )

			pdb_id, pdb_dict = self.entry_from_pdb_rest_api( pdb_id )

			self.pdb_benchmark_dict[pdb_id] = {}
			for k in pdb_dict:
				self.pdb_benchmark_dict[pdb_id][k] = pdb_dict[k]
		print( "Info. obtained for PDB entries: ", len( self.pdb_benchmark_dict ) )


	def filter_benchmark( self ):
		"""
		Remove a PDB entry if:
			1. It contains a non-polymeric entity.
				entity_ids > polymer_entity_ids.
			2. No. of UniProt IDs does not match the no. of polymer_entity_ids.
			3. No aligned PDB-UniProt positions available.
			4. Total system length > 1450.
		"""
		selected_pdb_ids = []
		c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0
		for pdb_id in self.pdb_benchmark_dict:
			# entity_ids = self.pdb_benchmark_dict[pdb_id]["entity_ids"].split( "," )
			polymer_entity_ids = self.pdb_benchmark_dict[pdb_id]["polymer_entity_ids"].split( "," )
			uniprot_ids = self.pdb_benchmark_dict[pdb_id]["uniprot_ids"].split( "," )
			uni_pos = self.pdb_benchmark_dict[pdb_id]["uni_pos"].split( "," )
			pdb_pos = self.pdb_benchmark_dict[pdb_id]["pdb_pos"].split( "," )

			# Removing non-polymeric entities.
			# if len( entity_ids ) > len( polymer_entity_ids ):
			# 	c1 += 1
			# 	continue

			if len( polymer_entity_ids ) < len( uniprot_ids ):
				c2 += 1
				continue

			if len( polymer_entity_ids ) > len( uniprot_ids ):
				c3 += 1
				continue

			# Aligned PDB-UniProt positiosn not present.
			# if any( [len( u ) == 0 for u in uni_pos] ) or any( [len( p ) == 0 for p in pdb_pos] ):
			if not all( uni_pos ) and not all( pdb_pos ):
				c4 += 1
				continue

			if self.pdb_benchmark_dict[pdb_id]["total_length"] > 1450:
				c5 += 1
				continue

			selected_pdb_ids.append( pdb_id )
		print( c1, "  ", c2, "  ", c3, "  ", c4, "  ", c5 )
		return selected_pdb_ids


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def segregate_benchmark( self, selected_pdb_ids ):
		"""
		Segregate all remaining complexes into:
			Heteromers: >1 polymer_entity_ids all with 1 chain.
			Homomers: only >=1 polymer_entity_ids with >1 chain.
		"""
		segregated_pdb_ids = {k: [] for k in ["hetero", "homo"]}
		for pdb_id in selected_pdb_ids:

			stoichiometry = self.pdb_benchmark_dict[pdb_id]["stoichiometry"]
			stoichiometry = stoichiometry.split( "-" )

			hetero = all( [s == "1" for s in stoichiometry] )

			if hetero:
				segregated_pdb_ids["hetero"].append( pdb_id )
			else:
				segregated_pdb_ids["homo"].append( pdb_id )
		return segregated_pdb_ids


	def filter_and_segregate( self ):
		"""
		Remove a PDB entry if:
			1. It contains a non-polymeric entity.
				entity_ids > polymer_entity_ids.
			2. No. of UniProt IDs does not match the no. of polymer_entity_ids.

		Segregate all remaining complexes into:
			Heteromers: >1 polymer_entity_ids all with 1 chain.
			Homoromers: only >=1 polymer_entity_ids with >1 chain.
		"""
		selected_pdb_ids = self.filter_benchmark()
		print( f"Selected PDB IDs: {len( selected_pdb_ids )}" )
		selected_pdb_ids = self.segregate_benchmark( selected_pdb_ids )

		return selected_pdb_ids


	def clean_benchmark_dict( self, selected_pdb_ids: List ):
		"""
		Remove PDB entries which were not selected.
		Remove PDB entries for which UniProt IDs could not be downloaded.
		"""
		tmp = copy.deepcopy( self.pdb_benchmark_dict )

		self.pdb_benchmark_dict = {}
		for pdb_id in tmp:
			if pdb_id in selected_pdb_ids["hetero"]:
				self.pdb_benchmark_dict[pdb_id] = tmp[pdb_id]
			elif pdb_id in selected_pdb_ids["homo"]:
				self.pdb_benchmark_dict[pdb_id] = tmp[pdb_id]

		print( "Remaiing entries: ", len( self.pdb_benchmark_dict ) )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def get_unique_uni_seq( self, selected_pdb_ids: Dict[str, List] ):
		"""
		Get unique UniProt IDs.
		"""
		unique_uni_ids = []
		for category in selected_pdb_ids:
			for pdb_id in selected_pdb_ids[category]:

				uniprot_ids = self.pdb_benchmark_dict[pdb_id]["uniprot_ids"].split( "," )

				for uni_id in uniprot_ids:
					if uni_id not in unique_uni_ids:
						unique_uni_ids.append( uni_id )

		return unique_uni_ids


	def dwnld_uni_seq( self, selected_pdb_ids: Dict[str, List] ):
		"""
		Download unique UniProt sequences for all UniProt accessions in the
			selected PDB IDs (hetero/homo-mers).
		"""
		unique_uni_ids = self.get_unique_uni_seq( selected_pdb_ids )
		total = len( unique_uni_ids )
		for idx, uni_id in enumerate( unique_uni_ids ):
		# with Pool( 5 ) as p:
		# 	for result in tqdm.tqdm(
		# 							p.imap_unordered( partial( get_uniprot_seq, max_trials = 10,
		# 																wait_time = 5,
		# 																return_id = True ),
		# 												unique_uni_ids ),
		# 							total = total ):
			print( f"Downloading seq for Uni ID: {uni_id} -- {idx}/{total}... " )
			uni_seq = get_uniprot_seq( uni_id,
									max_trials = 10,
									wait_time = 5,
									return_id = False )

			# uni_id, uni_seq = result
			if len( uni_seq ) != 0:
				self.uni_seq_dict[uni_id] = uni_seq
			else:
				print( f"{uni_id} --> {uni_seq}" )

		write_json( self.uni_seq_dict, self.uni_seq_file )


	def dwnld_pdb_struct( self, selected_pdb_ids: Dict[str, List] ):
		"""
		Download the structure as a .pdb file.
		"""
		for category in selected_pdb_ids:
			total = len( selected_pdb_ids[category] )
			for idx, pdb_id in enumerate( selected_pdb_ids[category] ):
				print( f"Downloading PDB struct for: {pdb_id} -- {idx}/{total}... " )
				pdb_file = os.path.join( self.pdb_struct_dir, f"{pdb_id}.pdb" )
				sys_pdb_path = os.path.join( self.benchmark_dir, f"{pdb_id}/{pdb_id}.pdb" )

				# If system already selected and created, do not download again.
				if os.path.exists( sys_pdb_path ):
					continue
				# If the .pdb file in metadata dir does not exist.
				if not os.path.exists( pdb_file ):
					result = download_pdb( pdb_id, "pdb", pdb_file )
					# If .pdb file doesn't exist, try .cif file.
					if not result:
						result = download_pdb( pdb_id, "cif", pdb_file )
						# Throw an error if can't download a .cif also.
						if not result:
							print( f"Warning: Unable to download PDB: {pdb_id}" )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def map_pdb_to_uniprot( self, selected_pdb_ids: Dict[str, List] ):
		"""
		Get PDB to UniProt mapping using SIFTS.
		"""
		for category in selected_pdb_ids:
			total = len( selected_pdb_ids[category] )
			for idx, pdb_id in enumerate( selected_pdb_ids[category] ):
				print( f"Obtaining SIFTS mapping for: {pdb_id} -- {idx}/{total}... " )

				xml_file_path = os.path.join( self.sifts_xml_dir, f"{pdb_id}.xml" )
				if not os.path.exists( xml_file_path ):
					download_sifts_mapping( pdb_id,
											xml_file_path,
											max_trials = 5,
											wait_time = 5 )
				sifts_dict = parse_sifts_xml( xml_file_path )
				write_json( sifts_dict,
							os.path.join( self.sifts_dict_dir, f"{pdb_id}.json" ) )

				pdb_uni_res = self.get_pdb_to_uni_res_map( sifts_dict )
				self.sifts_pdb_to_uni[pdb_id] = pdb_uni_res


	def get_pdb_to_uni_res_map( self, sifts_dict: Dict ) -> Dict[str, Dict[str, str]]:
		"""
		Create a dict with PDB positions as keys and UniProt positions 
			as values for all chains.
		"""
		pdb_uni_res = {}
		for chain in sifts_dict:
			# if chain not in pdb_uni_res:
			# 	pdb_uni_res[chain] = {}
			pdb_pos = sifts_dict[chain]["resolved"]["PDB position"]
			uni_pos = sifts_dict[chain]["resolved"]["Uniprot position"]
			if len( pdb_pos ) != len( uni_pos ):
				raise ValueError( "Error in PDB and UniProt mapping for" +
									f" PDB {pdb_id}, chain {chain}. \n"
									f"PDB residues = {len( pdb_pos )}" +
									f" and UniProt residues = {len( uni_pos )}" )
			pdb_uni_res[chain] = dict( zip( pdb_pos, uni_pos ) )

		return pdb_uni_res


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def create_system_dir( self, pdb_id: str ):
		"""
		Create directories for all benchmark systems.
		"""
		sys_dir = os.path.join( self.benchmark_dir, f"{pdb_id}" )
		os.makedirs( sys_dir, exist_ok = True )


	def get_entities( self, stoichiometry: List, uniprot_ids: Dict, uni_pos: List
					) -> List[Dict]:
		"""
		Create all entities part of the system. Have the following:
			entity_id (starts from 1).
			UniProt ID.
			copy_num --> no. of copies for an entity.
			start, end UniProt residue positions.
			Complete UniProt sequence.
		"""
		entities = []
		for i in range( len( stoichiometry ) ):
			uni_id = uniprot_ids[i]
			copy_num = int( stoichiometry[i] )
			# Residue positions are stored as "{start}-{end}"
			start, end = list( map( int, uni_pos[i].split( "-" ) ) )

			if uni_id not in self.uni_seq_dict:
				raise KeyError( f"{uni_id} not present in the self.uni_seq_dict..." )
			# The downstream script will select the required sequence.
			seq = self.uni_seq_dict[uni_id]

			entities.append(
				{
					"entity_id": i+1,
					"uni_id": uni_id,
					"copy_num": copy_num,
					"start": start,
					"end": end,
					"sequence": seq
				}
			 )

		return entities


	def create_sys_dict_entry( self, sys_num: int, sys_name: str, entities: Dict
								) -> Dict:
		"""
		Create a config dict containing:
			"system_{index}": {
					"name",
					"entity": {
						{},
						{}
					},
					"data_gathering": {
						"xl_restraint": {}
					}
			}
		"""
		sys_dict = {
					f"System_{sys_num}": {
							"name": sys_name,
							"entity": entities,
							"data_gathering": {
								"xl_restraint": {
									"xl_max_bound": self.xl_length,
									"file_name": f"{sys_name}_interprotein_xls.csv",
								}
							}
						}
					}
		return sys_dict


	def create_sys_config_dict( self ):
		"""
		For all the benchmark entries create a config dict.
		We use the PDB ID as the system name.
		"""
		for sys_num, sys_name in enumerate( self.pdb_benchmark_dict ):
			sys_dict = {}
			# Stoichiometry is stored as '-' separated values.
			stoichiometry = self.pdb_benchmark_dict[sys_name]["stoichiometry"].split( "-" )
			uniprot_ids = self.pdb_benchmark_dict[sys_name]["uniprot_ids"].split( "," )
			uni_pos = self.pdb_benchmark_dict[sys_name]["uni_pos"].split( "," )

			sys_config_file = os.path.join( self.benchmark_dir,
											f"{sys_name}/sys_conf_{sys_name}.json" )

			self.create_system_dir( sys_name )
			entities = self.get_entities( stoichiometry, uniprot_ids, uni_pos )
			sys_dict = self.create_sys_dict_entry( sys_num, sys_name, entities )

			src_path = os.path.join( self.pdb_struct_dir, f"{sys_name}.pdb" )
			dest_path = os.path.join( self.benchmark_dir, f"{sys_name}/" )
			if os.path.exists( src_path ):
				cmd = ["cp", f"{src_path}", f"{dest_path}"]
				run_subprocess( cmd )

			write_json( sys_dict, sys_config_file )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def run_jwalk( self, pdb_file: str ):
		"""
		Run Jwalk to obtain XLs given a .pdb file.
	
		Input:
		----------
		pdb_file --> Path to the .pdb file.

		returns:
		----------
		None
		"""
		cmd = [
		f"{self.jwalk_exec}",
		"-i", f"{pdb_file}",
		]

		run_subprocess( cmd )


	def parse_jwalk_output( self, name: str ) -> pd.DataFrame:
		"""
		Jwalk writes a .txt file containing all the XLs.
		It provides the following info:
			Index, Model (input file name)
			Atom1 --> AA-RES-CHAIN-CA
			Atom2 --> AA-RES-CHAIN-CA
			SASD --> Solvent accessible surface distance.
			Eculidean distance --> distance between CA atoms.
		Fetch all intra and inter-protein XLs for which the Euclidean distance is less than xl_length bound.

		Input:
		----------
		name --> System name.

		Returns:
		----------
		None
		"""
		file_path = glob.glob( "./Jwalk_results/*.txt" )
		if len( file_path ) == 0:
			raise Exception( f"Jwalk results .txt file does not exist for {name}..." )
		file_path = file_path[0]
		df = pd.read_csv( file_path, sep = "\s+" ) # delim_whitespace = True

		# Remove XLs with SASD distances higher than the XL_length.
		# 	Using SASD provides more accurate XLs.
		df = df.loc[df["SASD"] <= self.xl_length]

		inter = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( "-" ).str[2]]

		# Extract chain ID and res no.
		interprotein_xls = pd.DataFrame()
		for i in [1, 2]:
			interprotein_xls[f"prot{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[2]
			interprotein_xls[f"res{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[1]

		interprotein_xls = interprotein_xls.reset_index( drop = True )

		return interprotein_xls


	def get_uni_pos_for_xls( self, pdb_id: str, df: pd.DataFrame
							) -> pd.DataFrame:
		"""
		Given a dataframe for inter-protein XLs, map the PDB positions
			to the corresponding UniProt positions.
			Columns: Protein1, Residue1, Protein1, Residue1
		"""
		for i in range( df.shape[0] ):
			chain1 = df.loc[i, "prot1"]
			res1 = int( df.loc[i, "res1"] )
			chain2 = df.loc[i, "prot2"]
			res2 = int( df.loc[i, "res2"] )

			df.iloc[i, 1] = self.sifts_pdb_to_uni[pdb_id][chain1][res1]
			df.iloc[i, 3] = self.sifts_pdb_to_uni[pdb_id][chain2][res2]
		return df


	def get_chain_entity_map( self, pdb_id: str ) -> Dict[str, int]:
		"""
		For a given PDB ID, create dict mapping the auth_asym_ids to
			their respective entity_id.
		"""
		chain_entity_map = {}
		entity_ids = self.pdb_benchmark_dict[pdb_id]["polymer_entity_ids"].split( "," )
		auth_asym_ids = self.pdb_benchmark_dict[pdb_id]["auth_asym_ids"].split( "," )

		for i in range( len( auth_asym_ids ) ):
			for aa_id in auth_asym_ids[i].split( "-" ):
				chain_entity_map[aa_id] = entity_ids[i]

		return chain_entity_map


	def map_chain_to_protein( self, pdb_id: str, df: pd.DataFrame
								) -> pd.DataFrame:
		"""
		Given the inter-protein XL pairs, map the chains to the respective proteins.
		Each protein is identified by an entity_id, so we replace chains with -
			f"prot_{entity_id}".
		"""
		chain_entity_map = self.get_chain_entity_map( pdb_id )

		for i in range( df.shape[0] ):
			chain1 = df.iloc[i, 0]
			chain2 = df.iloc[i, 2]

			prot = f"prot_{chain_entity_map[chain1]}"
			df.iloc[i, 0] = prot
			prot = f"prot_{chain_entity_map[chain2]}"
			df.iloc[i, 2] = prot
		return df


	def simulate_xl_data( self ):
		"""
		Simulate XLs for the system uisng Jwalk.
		Save the intraprotein and interprotein XLs as csv files.

		Input:
		----------
		name --> System name.

		Returns:
		----------
		None
		"""
		print( "Simulating XL data..." )
		num_xls = {}
		# Remove PDB IDs for which JWalk couldn't be run or for which all residues were not mapped.
		no_xls, exclude = [], []
		for idx, pdb_id in enumerate( self.pdb_benchmark_dict ):
			print( f"\n--> {idx} --> {pdb_id}" )
			# Not all PDB positions are mapped to UniProt.
			# if pdb_id in ["8g0k", "8ck8"]:
			# 	continue
			sys_dir = os.path.join( self.benchmark_dir, f"{pdb_id}/" )

			# Move to system dir.
			os.chdir( sys_dir )
			if os.path.exists( f"./{pdb_id}.pdb" ):
				struct_file = os.path.join( f"{pdb_id}.pdb" )
			elif os.path.exists( f"./{pdb_id}.cif" ):
				struct_file = os.path.join( f"{pdb_id}.cif" )
			else:
				raise FileNotFoundError( f"PDB/CIF file not found for entry {pdb_id}..." )

			try:
				# Run Jwalk.
				if not os.path.exists( "./Jwalk_results/" ):
					self.run_jwalk( struct_file )
				else:
					print( f"Jwalk_results already present in {pdb_id} dir..." )
			except:
				print( f"Couldn't run JWalk for {pdb_id}..." )
				exclude.append( pdb_id )

			# Obtain intraprotein and interprotein XLs from Jwalk output.
			interprotein_xls = self.parse_jwalk_output( pdb_id )
			try:
				# Map PDB positions to UniProt positions for all XLs.
				interprotein_xls = self.get_uni_pos_for_xls( pdb_id, interprotein_xls )
				# Convert the chain IDs to proteins.
				interprotein_xls = self.map_chain_to_protein( pdb_id, interprotein_xls )
				print( f"Inter-XLs = {len( interprotein_xls )}" )

				if len( interprotein_xls ) > 0:
					# just logging the no. of XLs.
					num_xls[pdb_id] = len( interprotein_xls )
					# Save on disk.
					interprotein_xls.to_csv( f"./{pdb_id}_interprotein_xls.csv", index = False )
				else:
					no_xls.append( pdb_id )
			except:
				print( "Not all PDB residues mapped to UniProt..." )
				exclude.append( pdb_id )

			os.chdir( "../../../" )

		for pdb_id in exclude+no_xls:
			self.pdb_benchmark_dict.pop( pdb_id )

		return num_xls


	def write_benchmark_to_csv( self, num_xls ):
		"""
		Create a .csv file for all the PDB IDs in the benchmark.
		Ignore the ones with 0 interprotein-XLs.
		"""
		dum = list( self.pdb_benchmark_dict.keys() )[0]
		keys = ["pdb_id"] + list( self.pdb_benchmark_dict[dum].keys() )
		flat_dict = {k:[] for k in keys}
		flat_dict["Interprotein-XLs"] = []

		for pdb_id in self.pdb_benchmark_dict:
			flat_dict["pdb_id"].append( pdb_id )
			for k, v in self.pdb_benchmark_dict[pdb_id].items():
				flat_dict[k].append( v )
			flat_dict["Interprotein-XLs"].append( num_xls[pdb_id] )

		df = pd.DataFrame( flat_dict )
		df.to_csv( self.benchmark_csv, index = False )


if __name__ == "__main__":
	CreateBenchmark().forward()
