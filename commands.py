"""
Create the required command to run OpenFold.
"""
import os

class OpenfoldCommand():
	"""
	Create the required command to run OpenFold.
	"""
	def __init__( self, script: str, fasta_dir: str,
						config_preset: str, alignment_dir: str,
						output_dir: str, mode: str,
						cpu_cores: int, device: str ):
		self.script = script
		self.fasta_dir = fasta_dir
		self.config_preset = config_preset
		self.alignment_dir = alignment_dir
		self.output_dir = output_dir
		self.mode = mode
		self.seed = 1
		self.cpu_cores = cpu_cores
		self.device = device

		# Arguments for specifying path for the databases.
		self.db_base = "/data/alpha-fold-db/"
		self.databases = [
		[os.path.join( self.db_base, "pdb_mmcif/mmcif_files" )],
		["--uniref90_database_path", os.path.join( self.db_base, "uniref90/uniref90.fasta" )],
		["--mgnify_database_path", os.path.join( self.db_base, "mgnify/mgy_clusters_2022_05.fa" )],
		["--pdb70_database_path", os.path.join( self.db_base, "pdb70/pdb70" )],
		["--uniclust30_database_path", os.path.join( self.db_base,
													"uniclust30/uniclust30_2018_08/uniclust30_2018_08" )],
		["--pdb_seqres_database_path", os.path.join( self.db_base, "pdb_seqres/pdb_seqres.txt")],
		["--uniref30_database_path", os.path.join( self.db_base, "uniref30/UniRef30_2021_03" )],
		["--uniprot_database_path", os.path.join( self.db_base, "uniprot/uniprot.fasta" )],
		["--bfd_database_path", os.path.join( self.db_base,
											"bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" )],
		]

		# Arguments for specifying path for the tools.
		self.tool_base = "/home/kartik/miniforge3/envs/il_ofold/bin/"
		self.tools = [
		["--jackhmmer_binary_path", os.path.join( self.tool_base, "jackhmmer" )],
		["--hhblits_binary_path", os.path.join( self.tool_base, "hhblits" )],
		["--hhsearch_binary_path", os.path.join( self.tool_base, "hhsearch" )],
		["--kalign_binary_path", os.path.join( self.tool_base, "kalign" )],
		]

		self.other_options = [
		["--openfold_checkpoint_path", "openfold/resources/openfold_params/finetuning_ptm_2.pt"],
		["--config_preset", self.config_preset],
		["--data_random_seed", f"{self.seed}"],
		["--output_dir", self.output_dir],
		["--save_outputs"],
		["--cpus", f"{self.cpu_cores}"],
		["--model_device", self.device],
		["----max_template_date", "2023-01-01"]
		["--subtract_plddt"],
		["--cif_output"]
		]


	def get( self ):
		"""
		Return the required command to run OpenFold.
		"""
		command_list = []

		command_list.extend(
						["python3",
						self.script,
						self.fasta_dir,
						self.databases[0][0]]
			 			)

		if self.alignment_dir is not None:
			command_list.extend(
							["--use_precomputed_alignments", self.alignment_dir]
							 )

		# Remove args not required to run monomer/multimer prediction.
		if self.mode == "mono":
			del self.databases[5:8]
		else:
			del self.databases[3:5]

		for db in self.databases[1:]:
			command_list.extend( db )

		for tool in self.tools:
			command_list.extend( tool )

		# Do not require checkpoint for running multimer.
		if self.mode == "mono":
			command_list.extend( self.other_options[0] )

		for opt in self.other_options[1:]:
			command_list.extend( opt )

		return command_list
