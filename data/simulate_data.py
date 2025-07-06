"""
Contains classes to obtain simulated experimental data.
"""

from utils.utils import ( run_subprocess )




class SimulateCrosslinks():
	"""
	Simulate cross-linking data using JWalk.
	"""
	def __init__( self, pdb_ids_list: str,
					pdb_struct_dir: str,
					struct_format: str,
					xl_max_bound: int,
					min_inter_xls: int,
					cores: int ):
		self.pdb_ids_list = pdb_ids_list
		self.struct_dir = pdb_struct_dir
		self.struct_format = struct_format
		self.xl_max_bound = xl_max_bound
		self.min_inter_xls = min_inter_xls
		self.cores = cores

		self.xl_dict = []



	def forward( self ):
		"""
		"""
		t_start = time.time()
		self.initialize_logs_dict()

		self.struct_logs["Total_pdb_ids"] = len( self.pdb_ids_list )
		self.dwnld_struct_in_parallel()
		w = open_file_handler( self.downloaded_struct_file, "w" )
		w.writelines( ",".join( self.downloaded_struct ) )
		w.close()

		t_end = time.time()
		time_taken = t_end - t_start
		self.jwalk_logs["pdb_ids_with_inter_xls"] = len( self.xl_dict )
		self.jwalk_logs["time_taken"] = time_taken



	def initialize_logs_dict( self ):
		"""
		Create an empty logs dict with all required keys.
		"""
		self.jwalk_logs = {
		k:[] for k in ["failed_to_run_jwalk", "no_inter_xls",
						"too_few_xls"]
		}



	def run_jwalk( self, struct_file: str ):
		"""
		Run Jwalk to obtain XLs given a .pdb file.
			"""
		cmd = [
		f"{self.jwalk_exec}",
		"-i", f"{pdb_file}",
		]

		run_subprocess( cmd )



	def parse_jwalk_output( self, entry_id: str ) -> pd.DataFrame:
		"""
		Jwalk writes a .txt file containing all the XLs within
			the output dir named Jwalk_results.
		It provides the following info:
			Index, Model (input file name)
			Atom1 --> AA-RES-CHAIN-CA
			Atom2 --> AA-RES-CHAIN-CA
			SASD --> Solvent accessible surface distance.
			Eculidean distance --> distance between CA atoms.
		Fetch all inter-protein XLs for which the SASD is less than xl_max_bound.
		"""
		xl_file_path = glob.glob( f"./Jwalk_results/{entry_id}_*.txt" )
		if len( xl_file_path ) == 0:
			logs["failed_to_run_jwalk"] = entry_id
			interprotein_xls = None
		else:
			xl_file_path = xl_file_path[0]
			df = pd.read_csv( xl_file_path, sep = "\s+" ) # delim_whitespace = True

			# Remove XLs with SASD distances higher than the XL_length.
			# 	Using SASD provides more accurate XLs.
			df = df.loc[df["SASD"] <= self.xl_max_bound]

			inter = df[df["Atom1"].str.split( "-" ).str[2] != df["Atom2"].str.split( "-" ).str[2]]

			# Extract chain ID and res no.
			interprotein_xls = pd.DataFrame()
			for i in [1, 2]:
				interprotein_xls[f"prot{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[2]
				interprotein_xls[f"res{i}"] = inter[f"Atom{i}"].str.split( "-" ).str[1]

			interprotein_xls = interprotein_xls.reset_index( drop = True )

		return interprotein_xls



	def get_xls_for_entry_id( self, entry_id: str ):
		"""
		Given an entry_ids, obtain inter-protein XLs.
		"""
		logs = {}
		struct_file = os.path.join( self.pdb_struct_dir,
									f"{entry_id}.{self.struct_format}" )

		self.run_jwalk( struct_file )

		interprotein_xls = self.parse_jwalk_output( entry_id )

		if interprotein_xls is None:
			logs["failed_to_run_jwalk"] = entry_id
		elif interprotein_xls.shape[0] == 0:
			logs["no_inter_xls"]  = entry_id
			interprotein_xls = None
		elif interprotein_xls.shape[0] < self.min_inter_xls:
			logs["too_few_xls"]  = entry_id
			interprotein_xls = None

		return entry_id, interprotein_xls, logs



	def get_xls_in_parallel( self ):
	"""
	Parallelize obtaining XLs for the given entry_id's.
	"""
	with Pool( self.cores ) as p:
		for result in tqdm.tqdm(
			p.imap_unordered( self.get_xls_for_entry_id, self.pdb_ids_list ),
			total = len( self.pdb_ids_list )
			):
			entry_id, interprotein_xls, logs = result

			if interprotein_xls is None:
				for k in logs:
					self.jwalk_logs[k][0].append( logs[k] )
					self.jwalk_logs[k][1] += len( logs[k] )
			else:
				self.xl_dict[entry_id] = interprotein_xls

