import numpy as np

from utils import read_json, write_json

"""
This script assumes a specific directory structure:
	base_dir --> e.g. benchmark/2ayo/
		System specific JSON file.
		Restraint input files --> e.g. For Xl restraaint *_xlr.csv.
"""


class DataGathering():
	def __init__( self ):
		self.sys_name = sys_name
		# Main directory for the modeled system.
		self.base_dir = os.path.abspath( f"./benchmark/{self.sys_name}/" )



	def load_topology_file( self ):
		"""
		Parse the topology file (JSON format).
		"""
		self.topology_file = read_json( sefl.topology_file_path )


	def parse_xl_data( self, batch ):
		"""
		Parse the .csv file containing the XL data.
		"""
		df = pd.read_csv( os.path.abspath( "2ayo_interprotein_xls.csv" ) )

		r1, r2 = np.array( df["res1"] ), np.array( df["res2"] )
		r1, r2 = r1 - 1, r2 -1
		r2 += 404
		xl_dist = torch.zeros( ( 480, 480 ) )
		xl_mask = torch.zeros( ( 480, 480 ) )

		xl_mask[r1, r2] = 1
		xl_dist[r1, r2] = 35

		xl_mask[r2, r1] = 1
		xl_dist[r2, r1] = 35

		batch["xl_restraint"] = {}
		batch["xl_restraint"]["xl_res_mask"] = xl_mask
		# batch["xl_restraint"]["xl_tgt_mask"] = xl_dist

		return batch		


	def preprocess_xl_data( self, df: pd.dataFrame ):
		"""
		1. Sort the XLs according to chain IDs for prot1.
		2. Convert the residue positions to appropriate indices.
			For all residues in a chain, the index must be shifted by the no. 
				of residues in the previous chain.
		3. Split the indices into column vectors.
		"""



	def create_xl_map( self, xl_res1: np.array, xl_res2: np.array ):
		"""
		Create a zero-matrix with the shape defined by the system length.
		Add 1's for XL'd residues, creating a binary XL-map (essentially a contact map).
		"""


	def make_xl_restraint_features( self ):
		"""
		Given the XL'd residues create a binary mask for XL'd residue pairs (xl_res_mask).
		"""

