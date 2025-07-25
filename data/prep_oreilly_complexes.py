"""
Prepare input files for the complexes reported in
	O'Reilly F et al (2023) 10.15252/msb.202311544.
They provide XL-data and AF2 predicted structures (.pdb file).
We use the following 5 complexes as used in
	Zhnag Y et al (2025) 10.1101/2024.10.24.619977.

This script bypasses the default dataset creation pipeline.
	Extract the sequence and residue positions for all
		complexes from the PDB file.
	Create the system dir for all complexes with the
		structure, XL file,a nd sys config saved.
"""
from typing import List, Dict
import os
import numpy as np
import pandas as pd

from utils.utils import ( run_subprocess,
							open_file_handler,
							read_json, write_json )
from utils.pdb_utils import ( Parser )


class OreillyComplexes():
	"""
	Obtain the sequence and residue positions for all chains
		for the AF2 complexes from O'Reilly F et al (2023).
	"""
	def __init__( self ):
		self.benchmark_name = "oreilly"  # "afu", "sabdab", "oreilly"
		self.oreilly_complexes = ["gata_gatc", "gcvpa_gcvpb",
								"phes_phet", "roca_putc", "sucd_succ"]
		self.xl_max_bound = 30
		self.pdb_dict = {}
		self.xls_dict = {}



	def forward( self ):
		"""
		"""
		self.create_required_paths()
		self.create_required_dir()
		self.wrapper()
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
		# Base directory for all benchmarks.
		self.base_dir = os.path.join( os.path.abspath( "../benchmark/" ) )
		## --------------------------
		self.data_dir = os.path.join( os.path.abspath( "../raw/oreilly_complexes/" ) )

		# Benchmark specific dir.
		self.benchmark_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_benchmark" )
		# Dir to store benchmark metadata including PDB API files,
		# 	structure file, SIFTS mapping, benchamrk csv and the required intermediate files.
		self.meta_dir = os.path.join( self.base_dir, f"{self.benchmark_name}_metadata" )

		# Dir containing .pdb files for the O'Reilly complexes.
		self.pdb_struct_dir = os.path.join( self.data_dir, "pdbs" )

		# Dir containing XL files for the O'Reilly complexes.
		self.xls_dir = os.path.join( self.data_dir, "crosslinks" )

		# .csv file to store relevant details for the benchmark complexes.
		self.output_benchmark_csv = os.path.join( self.meta_dir,
												f"{self.benchmark_name}_benchmark.csv" )


	def create_required_dir( self ):
		"""
		Create the required directories if not already existing.
		"""
		os.makedirs( self.base_dir, exist_ok = True )
		os.makedirs( self.meta_dir, exist_ok = True )
		os.makedirs( self.benchmark_dir, exist_ok = True )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def wrapper( self ):
		"""
		For all complexes,
			Obtain the sequence and residue positions.
			Crosslinks with chains maped to protein name.
			Create and save required files to sys_dir.
		"""
		for sys_name in self.oreilly_complexes:
			self.pdb_dict[sys_name] = self.extract_from_struct( sys_name = sys_name )
			self.xls_dict[sys_name] = self.map_xls_to_prot( sys_name = sys_name )
			self.create_and_save_to_sys_dir( sys_name = sys_name )



	def extract_from_struct( self, sys_name: str ):
		"""
		Obtain the chain ID, sequence, and residue positions from the pdb files.
		"""
		pdb_file = os.path.join( self.pdb_struct_dir, f"{sys_name}.pdb" )
		p = Parser( pdb_file = pdb_file )

		chain_dict = {}

		chains = set()
		positions = []
		sequence = []
		for model in p.get_models():
			for residue, chain_id in p.get_residues_from_model( model ):
				if chain_id not in chain_dict:
					chain_dict[chain_id] = {k:[] for k in ["seq", "position"]}

				res_symbol = p.extract_perresidue_quantity( residue, "res_name" )
				chain_dict[chain_id]["seq"].append( res_symbol )
				res_pos = p.extract_perresidue_quantity( residue, "res_pos" )
				chain_dict[chain_id]["position"].append( res_pos )
		return chain_dict



	def map_xls_to_prot( self, sys_name: str ) -> pd.DataFrame:
		"""
		Convert the chain IDs in the XL .csv file to protein name.
		Assuming, the complex anme contains proteins in the
			same order as the chains in structure.
		Columns: res1,prot1,res2,prot2
		"""
		xl_file = os.path.join( self.xls_dir, f"{sys_name}.csv" )
		df = pd.read_csv( xl_file )

		chain_map = dict( zip(
			self.pdb_dict[sys_name].keys(),
			sys_name.split( "_" )
			) )
		for i in df.index:
			chain1 = df.iloc[i, 1]
			chain2 = df.iloc[i, 3]
			df.iloc[i, 1] = f"{chain_map[chain1]}_1"
			df.iloc[i, 3] = f"{chain_map[chain1]}_2"
		# Reorder columns to prot1,res1,prot2,res2
		df = df.reindex( columns = ["prot1", "res1", "prot2", "res2"] )
		return df



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

		prots = sys_name.split( "_" )
		idx = 0
		for entity_id, chain_id in enumerate( self.pdb_dict[sys_name] ):
			copy_num = 1

			seq = "".join( self.pdb_dict[sys_name][chain_id]["seq"] )
			positions = self.pdb_dict[sys_name][chain_id]["position"]
			start = positions[0]
			end = positions[-1]

			if len( seq ) != ( end - start + 1 ):
				raise ValueError( f"Sequence length ({len( seq )}) " +
								f"does not match the residue numbering ({start}-{end}..." )
			entities.append(
				{
					"name": prots[idx],
					"entity_id": int( entity_id+1 ),
					"copy_num": copy_num,
					"start": start,
					"end": end,
					"sequence": seq
				}
			 )
			idx += 1

		return entities



	def create_sys_dict( self,
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
		sys_name = "-".join( sys_name.split( "_" ) )

		sys_dict = {
		f"System_{sys_name}": {
				"name": sys_name,
				"entity": entities,
				"data_gathering": {
					"xl_restraint": {
						"xl_max_bound": self.xl_max_bound,
						"file_name": xl_file,
					}
				}
			}
		}
		return sys_dict



	def create_and_save_to_sys_dir( self, sys_name: str ):
		"""
		Create dir for the given system and save the pdb file,
			xl file, and sys config to it.
		"""
		# Create the system dir.
		sys_dir = os.path.join( self.benchmark_dir, sys_name )
		os.makedirs( sys_dir, exist_ok = True )

		# Copy the structure (.pdb) and XL file to sys_dir.
		pdb_file = os.path.join( self.pdb_struct_dir, f"{sys_name}.pdb" )
		pdb_dest = os.path.join( sys_dir, f"{sys_name}.pdb" )
		xl_dest = os.path.join( sys_dir, f"interprotein_xls.csv" )
		self.xls_dict[sys_name].to_csv( xl_dest, index = False )

		run_subprocess( ["cp", f"{pdb_file}", f"{pdb_dest}"] )

		# Create sys_dict and save to sys_dir.
		entities = self.get_entities( sys_name )
		sys_dict = self.create_sys_dict( sys_name = sys_name,
										entities = entities,
										xl_file = f"interprotein_xls.csv" )

		sys_dict_file = os.path.join( sys_dir, f"sys_config_{sys_name}.json" )
		write_json( sys_dict, sys_dict_file )


	##------------------------------------------------------------##
	##------------------------------------------------------------##
	def write_benchmark_csv( self ):
		"""
		Write the following info for the benchmark to a .csv file:
			PDB ID
			Polymer entity IDs
			Uniprot IDs
			Stoichiometry
			residue positions
			Total length
			Total inter-XLs
		"""
		flat_dict = {k:[] for k in ["PDB ID", "Polymer entity",
									"Auth Asym ID",
									"Stoichiometry",
									"Residue positions",
									"Total length",
									"Total TP XLs",
									"Selected TP XLs",
									"Total FP XLs",
									"Selected FP XLs"]}
		for sys_name in self.oreilly_complexes:
			# Arbitarily creating entity_id's as the complexes are heterodimeric.
			entity_ids = ["1", "2"]
			stoichiometry = []
			auth_asym_ids = []
			pos = []
			total_length = 0

			aa_ids = list( self.pdb_dict[sys_name].keys() )
			auth_asym_ids.append( "-".join( aa_ids ) )
			stoichiometry.append( f"{len( aa_ids )}" )
			for aa_id in aa_ids:
				positions = self.pdb_dict[sys_name][aa_id]["position"]
				start = positions[0]
				end = positions[-1]
				pos.append( f"{start}-{end}" )
				total_length += len( self.pdb_dict[sys_name][aa_id]["seq"] )

			flat_dict["PDB ID"].append( sys_name )
			flat_dict["Polymer entity"].append( ",".join( entity_ids ) )
			flat_dict["Auth Asym ID"].append( ",".join( auth_asym_ids ) )
			flat_dict["Stoichiometry"].append( ",".join( stoichiometry ) )
			flat_dict["Residue positions"].append( ",".join( pos ) )
			flat_dict["Total length"].append( total_length )
			flat_dict["Total TP XLs"].append( 
						self.xls_dict[sys_name].shape[0]
						)
			flat_dict["Selected TP XLs"].append( 
						self.xls_dict[sys_name].shape[0]
						)
			flat_dict["Total FP XLs"].append( 0 )
			flat_dict["Selected FP XLs"].append( 0 )
		df = pd.DataFrame( flat_dict )
		df.to_csv( self.output_benchmark_csv, index = False )


if __name__ == "__main__":
	OreillyComplexes().forward()
