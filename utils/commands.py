"""
Create the required command to run OpenFold.
"""
import os, time

class OpenfoldCommand():
	"""
	Create the required command to run OpenFold.
	"""
	def __init__( self, 
			db_dir: str,
			script: str,
			config_preset: str,
			db_preset: str,
			tool_base: str,
			model_ckpt: str,
			fasta_dir: str,
			alignment_dir: str,
			output_dir: str,
			max_template_date: str,
			mode: str,
			seed: int,
			cpu_cores: int,
			device: str
		):
		self.script = script
		self.fasta_dir = fasta_dir
		self.alignment_dir = alignment_dir
		self.output_dir = output_dir

		self.mode = mode

		print( f"Using {db_preset} for OpenFold..." )
		time.sleep( 1 )

		# Arguments for specifying path for the databases.
		self.databases = [
		[os.path.join( db_dir, "pdb_mmcif/mmcif_files" )],
		["--uniref90_database_path", os.path.join( db_dir, "uniref90/uniref90.fasta" )],
		["--mgnify_database_path", os.path.join( db_dir, "mgnify/mgy_clusters_2022_05.fa" )],
		["--pdb70_database_path", os.path.join( db_dir, "pdb70/pdb70" )],
		["--uniclust30_database_path", os.path.join( db_dir,
													"uniclust30/uniclust30_2018_08/uniclust30_2018_08" )],
		["--pdb_seqres_database_path", os.path.join( db_dir, "pdb_seqres/pdb_seqres.txt")],
		["--uniref30_database_path", os.path.join( db_dir, "uniref30/UniRef30_2021_03" )],
		["--uniprot_database_path", os.path.join( db_dir, "uniprot/uniprot.fasta" )],
		["--bfd_database_path", os.path.join( db_dir,
											"bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" )],
		]

		# Arguments for specifying path for the tools.
		self.tools = [
		["--jackhmmer_binary_path", os.path.join( tool_base, "jackhmmer" )],
		["--hhblits_binary_path", os.path.join( tool_base, "hhblits" )],
		["--hhsearch_binary_path", os.path.join( tool_base, "hhsearch" )],
		["--kalign_binary_path", os.path.join( tool_base, "kalign" )],
		]

		self.other_options = [
		["--openfold_checkpoint_path", f"{model_ckpt}"],
		["--preset", f"{db_preset}"],
		["--config_preset", f"{config_preset}"],
		["--data_random_seed", f"{seed}"],
		["--output_dir", output_dir],
		["--save_outputs"],
		["--cpus", f"{cpu_cores}"],
		["--model_device", f"{device}"],
		["--max_template_date", f"{max_template_date}"],
		# ["--use_deepspeed_evoformer_attention"],
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
