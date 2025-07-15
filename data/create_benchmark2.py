"""
Create input for the benchmark dataset for modeling.
"""
from typing import List, Dict
import os
import numpy as np
import pandas as pd
from utils.utils import ( open_file_handler,
							read_json,
							write_json,
							run_subprocess )


class CreateBenchmark():
	"""
	Create input files for modeling.
	"""
	def __init__( self ):
		self.benchmark_name = "afu"

		self.chain_entity_map = {}


	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.required_dir_exist()
		self.load_metadata()

		self.create_chain_entity_map()

		self.create_system_inputs_for_modeling()
		self.write_benchmark_csv()
		print( "\n May the Force be with you..." )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def create_required_paths( self ):
		"""
		Create paths for all required directories and files.
		"""
		## --------------------------
		# Global paths
		## --------------------------
		# Base directory for all benchmarks.
		self.base_dir = os.path.join( os.path.abspath( "../benchmark/" ) )
		# Benchmark specific dir.
		self.benchmark_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_benchmark" )
		# Dir to store benchmark metadata including structure file,
		# 	seqres dict, benchamrk csv and the required intermediate files.
		self.meta_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_metadata" )
		# Dir containing PDB structures.
		self.pdb_struct_dir = os.path.join( self.meta_dir, "struct" )
		# PDB IDs remaining after metadat collection.
		self.benchmark_pdbs_file =  os.path.join( self.meta_dir,
										f"{self.benchmark_name}_benchmark_pdb_ids.txt" )

		self.seqres_dict_file = os.path.join( self.meta_dir, "seqres_dict.npy" )
		# .csv file to store relevant details for the benchmark complexes.
		self.output_benchmark_csv = os.path.join( self.meta_dir,
												f"{self.benchmark_name}_benchmark.csv" )

		self.dataset_configs_file = os.path.join( self.meta_dir,
								f"Dataset_configs_{self.benchmark_name}.csv" )
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
		self.dataset_configs = read_json( self.dataset_configs_file )
		f = open_file_handler( self.benchmark_pdbs_file, "r" )
		self.benchmark_pdb_ids = f.readlines()[0].split( "," )
		f.close()
		self.seqres_dict = np.load( self.seqres_dict_file, allow_pickle = True ).item()
		self.xls_dict = np.load( self.xls_dict_file, allow_pickle = True ).item()


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def create_chain_entity_map( self ):
		"""
		For a given PDB ID (sys_name), create dict mapping the auth_asym_ids to
			their respective entity_id.
		"""
		for sys_name in self.benchmark_pdb_ids:
			self.chain_entity_map[sys_name] = {}

			entity_ids = list( self.seqres_dict[sys_name].keys() )
			for entity_id in entity_ids:
				auth_asym_ids = list( self.seqres_dict[sys_name][entity_id].keys() )

				for aa_id in auth_asym_ids:
					self.chain_entity_map[sys_name][aa_id] = entity_id



	def map_xl_chain_to_entity( self, sys_name: str, xl_df: pd.DataFrame ):
		"""
		Map the chain IDs for all XLs to the respective entity.
		e.g. For a system contaiing chains A-B,C:
			A -> Protein_1; B -> Protein_1; C -> Protein_2
		"""
		for i in range( xl_df.shape[0] ):
			chain1 = xl_df.loc[i, "prot1"]
			chain2 = xl_df.loc[i, "prot2"]

			entity_id1 = self.chain_entity_map[sys_name][chain1]
			prot1 = f"{sys_name}_{entity_id1}"
			entity_id2 = self.chain_entity_map[sys_name][chain2]
			prot2 = f"{sys_name}_{entity_id2}"

			xl_df.iloc[i, 0] = prot1
			xl_df.iloc[i, 2] = prot2



	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def get_entities( self, sys_name: str ) -> List[Dict]:
		"""
		Create all entities part of the system. Have the following:
			entity_id (starts from 1).
			copy_num --> no. of copies for an entity.
			start, end UniProt residue positions.
			Complete UniProt sequence.
		"""
		entities = []
		entity_id = 1

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
					"name": sys_name,
					"entity_id": int( entity_id ),
					"copy_num": copy_num,
					"start": start,
					"end": end,
					"sequence": seq
				}
			 )

		return entities


	def create_sys_dict( self, sys_num: int,
						sys_name: str,
						entities: Dict,
						xl_file: str ) -> Dict:
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
		f"System_{sys_name}": {
				"name": sys_name,
				"entity": entities,
				"data_gathering": {
					"xl_restraint": {
						"xl_max_bound": self.dataset_configs["jwalk"]["xl_max_bound"],
						"file_name": xl_file,
					}
				}
			}
		}
		return sys_dict


	def create_sys_dir( self, sys_name: str ) -> str:
		"""
		Create the system dir and return the path to sys dir.
		"""
		sys_dir = os.path.join( self.benchmark_dir, sys_name )
		os.makedirs( sys_dir, exist_ok = True )
		return sys_dir


	def save_struct_to_sys_dir( self, sys_name: str,
								sys_dir: str ):
		"""
		Save the structure file in sys dir.
		"""
		struct_format = self.dataset_configs["global"]["struct_format"]
		struct_format = ["pdb", "cif"] if struct_format == "both" else struct_format
		for ext in struct_format:
			struct_file_src = os.path.join( self.pdb_struct_dir,
										f"{sys_name}.{ext}" )
			dest = sys_dir
			cmd = ["cp", struct_file_src, dest]
		run_subprocess( cmd )


	def save_tp_sys_dict_to_sys_dir( self,
									sys_num: int,
									sys_name: str,
									sys_dir: str,
									entities: List[Dict]
									):
		"""
		For all TP xls, create sys config dict and get the Jwalk XLs.
		Save both in sys_dir.
		"""
		# Save TP XLs to sys dir.
		xl_file = f"interprotein_xls_tp.csv"
		sys_dict_file = os.path.join( sys_dir, f"sys_config_tp_{sys_name}.json" )
		sys_dict = self.create_sys_dict( sys_num = sys_num,
											sys_name = sys_name,
											entities = entities,
											xl_file = xl_file )
		write_json( sys_dict, sys_dict_file )

		tp_xls_file = os.path.join( sys_dir, xl_file )
		tp_xls = self.xls_dict[sys_name]["tp_xls"]
		self.map_xl_chain_to_entity( sys_name = sys_name,
										xl_df = tp_xls )
		tp_xls.to_csv( tp_xls_file, index = False )



	# def add_fp_xls( self, sys_name: str ):
	# 	"""
	# 	Add 5% FP XLs to TP XLs.
	# 	"""


	# def save_fp_sys_dict_to_sys_dir( self,
	# 								sys_num: int,
	# 								sys_name: str,
	# 								sys_dir: str,
	# 								entities: List[Dict]
	# 								):
	# 	"""
	# 	For all TP xls, create sys config dict and get the Jwalk XLs.
	# 	Save both in sys_dir.
	# 	"""
	# 	# Save TP XLs to sys dir.
	# 	xl_file = f"{sys_name}_fp_xls.csv"
	# 	sys_dict_file = os.path.join( sys_dir, f"sys_config_fp_{sys_name}.json" )
	# 	sys_dict = self.create_sys_dict_entry( sys_num, sys_name, entities )
	# 	write_json( sys_dict, sys_dict_file )

	# 	tp_xls_file = os.path.join( sys_dir, f"interprotein_xls_tp.csv" )
	# 	tp_xls = self.xls_dict[pdb_id]["tp_xls"]
	# 	tp_xls.to_csv( tp_xls_file, index = False )



	def create_system_inputs_for_modeling( self ):
		"""
		For all the complexes included in the benchmark,
			create inputs files required for modeling.
		We use the PDB ID as the system name.
		"""
		for sys_num, sys_name in enumerate( self.benchmark_pdb_ids ):
			sys_config_file = os.path.join( self.benchmark_dir,
											f"{sys_name}/sys_conf_{sys_name}.json" )

			sys_dir = self.create_sys_dir( sys_name = sys_name )
			self.save_struct_to_sys_dir( sys_name = sys_name,
										sys_dir = sys_dir )
			entities = self.get_entities( sys_name )
			self.save_tp_sys_dict_to_sys_dir( sys_num = sys_num,
											sys_name = sys_name,
											sys_dir = sys_dir,
											entities = entities )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def write_benchmark_csv( self ):
		"""
		Write the following info for the benchmark to a .csv file:
			PDB ID
			Polymer entity IDs
			Uniprot IDs
			Stoichiometry
			Uniprot positions
			Total length
			Total inter-XLs
		"""
		flat_dict = {k:[] for k in ["PDB ID", "Polymer entity",
									"Auth Asym ID",
									"Stoichiometry",
									"Residue positions",
									"Total length",
									"TP-interprotein XLs",
									"FP-interprotein XLs"]}
		for sys_name in self.benchmark_pdb_ids:
			entity_ids = list( self.seqres_dict[sys_name].keys() )
			auth_asym_ids = []
			stoichiometry = []
			pos = []
			total_length = 0
			for entity_id in entity_ids:
				aa_ids = list( self.seqres_dict[sys_name][entity_id].keys() )
				auth_asym_ids.append( "-".join( aa_ids ) )
				stoichiometry.append( f"{len( aa_ids )}" )
				start = self.seqres_dict[sys_name][entity_id][aa_ids[0]]["start_seq_id"]
				end = self.seqres_dict[sys_name][entity_id][aa_ids[0]]["end_seq_id"]
				pos.append( f"{start}-{end}" )

				for aa_id in aa_ids:
					seq = self.seqres_dict[sys_name][entity_id][aa_id]["seq"]
					total_length += len( seq )

			flat_dict["PDB ID"].append( sys_name )
			flat_dict["Polymer entity"].append( ",".join( entity_ids ) )
			flat_dict["Auth Asym ID"].append( ",".join( auth_asym_ids ) )
			flat_dict["Stoichiometry"].append( ",".join( stoichiometry ) )
			flat_dict["Residue positions"].append( ",".join( pos ) )
			flat_dict["Total length"].append( total_length )
			flat_dict["TP-interprotein XLs"].append( 
						self.xls_dict[sys_name]["tp_xls"].shape[0]
						)
			flat_dict["FP-interprotein XLs"].append( 
						self.xls_dict[sys_name]["fp_xls"].shape[0]
						)
		df = pd.DataFrame( flat_dict )
		df.to_csv( self.output_benchmark_csv, index = False )


if __name__ == "__main__":
	CreateBenchmark().forward()

