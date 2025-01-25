import os


class OpenfoldCommand():
	def __init__( self, script: str, fasta_dir: str, 
						config_preset: str, alignment_dir: str, 
						output_dir: str, mode: str, cpu_cores: str, 
						device: str ):
		self.script = script
		self.fasta_dir = fasta_dir
		self.config_preset = config_preset
		self.alignment_dir = alignment_dir
		self.output_dir = output_dir
		self.mode = mode
		self.cpu_cores = cpu_cores
		self.device = device

		# databases.
		self.db_base = "/data/alpha-fold-db/"
		self.databases = [
		[os.path.join( self.db_base, "pdb_mmcif/mmcif_files" )],
		["--uniref90_database_path", os.path.join( self.db_base, "uniref90/uniref90.fasta" )],
		["--mgnify_database_path", os.path.join( self.db_base, "mgnify/mgy_clusters_2022_05.fa" )],
		["--pdb70_database_path", os.path.join( self.db_base, "pdb70/pdb70" )],
		["--uniclust30_database_path", os.path.join( self.db_base, "uniclust30/uniclust30_2018_08/uniclust30_2018_08" )],
		["--pdb_seqres_database_path", os.path.join( self.db_base, "pdb_seqres/pdb_seqres.txt")],
		["--uniref30_database_path", os.path.join( self.db_base, "uniref30/UniRef30_2021_03" )],
		["--uniprot_database_path", os.path.join( self.db_base, "uniprot/uniprot.fasta" )],
		["--bfd_database_path", os.path.join( self.db_base, "bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" )],
		]

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
		["--output_dir", self.output_dir],
		["--save_outputs"],
		["--cpus", f"{self.cpu_cores}"],
		["--model_device", self.device],
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

		if self.alignment_dir != None:
			command_list.extend( 
							["--use_precomputed_alignments", self.alignment_dir]
							 )

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



# def monomer_cmd( script:str, fasta_dir: str, 
# 				output_dir: str, config_preset: str, 
# 				ckpt_path: str, cpu_cores: int = 4, 
# 				device: str = "cpu" ):
# 	"""
# 	Command to run OpenFold for monomer prediction from 
# 		scratch (without precomputed alignments).

# 	Input:
# 	----------
# 	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
# 	fasta_dir --> (str) path for the input FASTA file.
# 	output_dir --> (str) path for the output directory.

# 	Returns:
# 	----------
# 	(list) command to run monomer prediction in subprocess acceptable format.
# 	"""
# 	return [
# 		"python3", f"{script}",
# 		f"{fasta_dir}",
# 		# "/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
# 		# "--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
# 		# "--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
# 		# "--pdb70_database_path", "/data/alpha-fold-db/pdb70/pdb70",
# 		# "--uniclust30_database_path", "/data/alpha-fold-db/uniclust30/uniclust30_2018_08/uniclust30_2018_08",
# 		# "--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
# 		# "--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
# 		# "--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
# 		# "--hhsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhsearch",
# 		# "--kalign_binary_path", "lib/conda/envs/il_ofold/bin/kalign",
# 		"--config_preset", f"{config_preset}",
# 		"--output_dir", f"{output_dir}",
# 		"--openfold_checkpoint_path", "openfold/resources/openfold_params/finetuning_ptm_2.pt",
# 		"--save_outputs",
# 		"--cpus", f"{cpu_cores}",
# 		"--model_device", f"{device}",
# 		"--cif_output"
# 		]

# def monomer_precomp_aln_cmd( script:str, fasta_dir: str, alignment_dir: str,
# 							output_dir: str, config_preset: str, 
# 							ckpt_path: str, cpu_cores: int = 4, 
# 							device: str = "cpu" ):
# 	"""
# 	Command to run OpenFold for monomer prediction from 
# 		with precomputed alignments..

# 	Input:
# 	----------
# 	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
# 	fasta_dir --> (str) path for the input FASTA file.
# 	alignment_dir --> (str) path to the directory containing precomputed alignments.
# 	output_dir --> (str) path for the output directory.

# 	Returns:
# 	----------
# 	(list) command to run monomer prediction in subprocess acceptable format.
# 	"""
# 	return [
# 		"python3", f"{script}",
# 		f"{fasta_dir}",
# 		"/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
# 		"--use_precomputed_alignments", f"{alignment_dir}",
# 		"--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
# 		"--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
# 		"--pdb70_database_path", "/data/alpha-fold-db/pdb70/pdb70",
# 		"--uniclust30_database_path", "/data/alpha-fold-db/uniclust30/uniclust30_2018_08/uniclust30_2018_08",
# 		"--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
# 		"--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
# 		"--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
# 		"--hhsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhsearch",
# 		"--kalign_binary_path", "lib/conda/envs/il_ofold/bin/kalign",
# 		"--config_preset", f"{config_preset}",
# 		"--output_dir", f"{output_dir}",
# 		"--openfold_checkpoint_path", f"{ckpt_path}",
# 		"--save_outputs",
# 		"--cpus", f"{cpu_cores}",
# 		"--model_device", f"{device}",
# 		"--cif_output"
# 	]


# def multimer_cmd( script:str, fasta_dir: str,
# 					output_dir: str, config_preset: str, 
# 					cpu_cores: int = 4, device: str = "cpu" ):
# 	"""
# 	Command to run OpenFold for multimer prediction from 
# 		scratch (without precomputed alignments).

# 	Input:
# 	----------
# 	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
# 	fasta_dir --> (str) path for the input FASTA file.
# 	output_dir --> (str) path for the output directory.

# 	Returns:
# 	----------
# 	(list) command to run multimer prediction in subprocess acceptable format.
# 	"""
# 	return [
# 		"python3", f"{script}",
# 		f"{fasta_dir}",
# 		"/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
# 		"--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
# 		"--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
# 		"--pdb_seqres_database_path", "/data/alpha-fold-db/pdb_seqres/pdb_seqres.txt",
# 		"--uniref30_database_path", "/data/alpha-fold-db/uniref30/UniRef30_2021_03",
# 		"--uniprot_database_path", "/data/alpha-fold-db/uniprot/uniprot.fasta",
# 		"--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
# 		"--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
# 		"--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
# 		"--hmmsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmsearch",
# 		"--hmmbuild_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmbuild",
# 		"--kalign_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/kalign",
# 		"--config_preset", f"{config_preset}",
# 		"--output_dir", f"{output_dir}",
# 		"--save_outputs",
# 		"--cpus", f"{cpu_cores}",
# 		"--model_device", f"{device}",
# 		"--cif_output"
# 		]


# def multimer_precomp_aln_cmd( script:str, fasta_dir: str, alignment_dir: str,
# 								output_dir: str, config_preset: str, 
# 								cpu_cores: int = 4, device: str = "cpu" ):
# 	"""
# 	Command to run OpenFold for multimer prediction from 
# 		with precomputed alignments.

# 	Input:
# 	----------
# 	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
# 	fasta_dir --> (str) path for the input FASTA file.
# 	alignment_dir --> (str) path to the directory containing precomputed alignments.
# 	output_dir --> (str) path for the output directory.

# 	Returns:
# 	----------
# 	(list) command to run multimer prediction in subprocess acceptable format.
# 	"""
# 	return [
# 		"python3", f"{script}",
# 		f"{fasta_dir}",
# 		"/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
# 		"--use_precomputed_alignments", f"{alignment_dir}",
# 		"--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
# 		"--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
# 		"--pdb_seqres_database_path", "/data/alpha-fold-db/pdb_seqres/pdb_seqres.txt",
# 		"--uniref30_database_path", "/data/alpha-fold-db/uniref30/UniRef30_2021_03",
# 		"--uniprot_database_path", "/data/alpha-fold-db/uniprot/uniprot.fasta",
# 		"--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
# 		"--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
# 		"--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
# 		"--hmmsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmsearch",
# 		"--hmmbuild_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmbuild",
# 		"--kalign_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/kalign",
# 		"--config_preset", f"{config_preset}",
# 		"--output_dir", f"{output_dir}",
# 		"--save_outputs",
# 		"--cpus", f"{cpu_cores}",
# 		"--model_device", f"{device}",
# 		"--cif_output"
# 	]


