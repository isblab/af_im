


def monomer_cmd( script:str, fasta_dir: str, 
				output_dir: str, config_preset: str, 
				ckpt_path: str, cpu_cores: int = 4, 
				device: str = "cpu" ):
	"""
	Command to run OpenFold for monomer prediction from 
		scratch (without precomputed alignments).

	Input:
	----------
	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
	fasta_dir --> (str) path for the input FASTA file.
	output_dir --> (str) path for the output directory.

	Returns:
	----------
	(list) command to run monomer prediction in subprocess acceptable format.
	"""
	return [
		"python3", f"{script}",
		f"{fasta_dir}",
		"/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
		"--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
		"--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
		"--pdb70_database_path", "/data/alpha-fold-db/pdb70/pdb70",
		"--uniclust30_database_path", "/data/alpha-fold-db/uniclust30/uniclust30_2018_08/uniclust30_2018_08",
		"--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
		"--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
		"--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
		"--hhsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhsearch",
		"--kalign_binary_path", "lib/conda/envs/il_ofold/bin/kalign",
		"--config_preset", f"{config_preset}",
		"--output_dir", f"{output_dir}",
		"--openfold_checkpoint_path", "openfold/resources/openfold_params/finetuning_ptm_2.pt",
		"--save_outputs",
		"--cpus", f"{cpu_cores}",
		"--model_device", f"{device}",
		"--cif_output"
		]

def monomer_precomp_aln_cmd( script:str, fasta_dir: str, alignment_dir: str,
							output_dir: str, config_preset: str, 
							ckpt_path: str, cpu_cores: int = 4, 
							device: str = "cpu" ):
	"""
	Command to run OpenFold for monomer prediction from 
		with precomputed alignments..

	Input:
	----------
	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
	fasta_dir --> (str) path for the input FASTA file.
	alignment_dir --> (str) path to the directory containing precomputed alignments.
	output_dir --> (str) path for the output directory.

	Returns:
	----------
	(list) command to run monomer prediction in subprocess acceptable format.
	"""
	return [
		"python3", f"{script}",
		f"{fasta_dir}",
		"/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
		"--use_precomputed_alignments", f"{alignment_dir}",
		"--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
		"--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
		"--pdb70_database_path", "/data/alpha-fold-db/pdb70/pdb70",
		"--uniclust30_database_path", "/data/alpha-fold-db/uniclust30/uniclust30_2018_08/uniclust30_2018_08",
		"--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
		"--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
		"--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
		"--hhsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhsearch",
		"--kalign_binary_path", "lib/conda/envs/il_ofold/bin/kalign",
		"--config_preset", f"{config_preset}",
		"--output_dir", f"{output_dir}",
		"--openfold_checkpoint_path", f"{ckpt_path}",
		"--save_outputs",
		"--cpus", f"{cpu_cores}",
		"--model_device", f"{device}",
		"--cif_output"
	]


def multimer_cmd( script:str, fasta_dir: str,
					output_dir: str, config_preset: str, 
					cpu_cores: int = 4, device: str = "cpu" ):
	"""
	Command to run OpenFold for multimer prediction from 
		scratch (without precomputed alignments).

	Input:
	----------
	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
	fasta_dir --> (str) path for the input FASTA file.
	output_dir --> (str) path for the output directory.

	Returns:
	----------
	(list) command to run multimer prediction in subprocess acceptable format.
	"""
	return [
		"python3", f"{script}",
		f"{fasta_dir}",
		"/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
		"--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
		"--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
		"--pdb_seqres_database_path", "/data/alpha-fold-db/pdb_seqres/pdb_seqres.txt",
		"--uniref30_database_path", "/data/alpha-fold-db/uniref30/UniRef30_2021_03",
		"--uniprot_database_path", "/data/alpha-fold-db/uniprot/uniprot.fasta",
		"--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
		"--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
		"--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
		"--hmmsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmsearch",
		"--hmmbuild_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmbuild",
		"--kalign_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/kalign",
		"--config_preset", f"{config_preset}",
		"--output_dir", f"{output_dir}",
		"--save_outputs",
		"--cpus", f"{cpu_cores}",
		"--model_device", f"{device}",
		"--cif_output"
		]


def multimer_precomp_aln_cmd( script:str, fasta_dir: str, alignment_dir: str,
								output_dir: str, config_preset: str, 
								cpu_cores: int = 4, device: str = "cpu" ):
	"""
	Command to run OpenFold for multimer prediction from 
		with precomputed alignments.

	Input:
	----------
	script --> (str) path for the OpenFold run_pretrained_openfold.py script.
	fasta_dir --> (str) path for the input FASTA file.
	alignment_dir --> (str) path to the directory containing precomputed alignments.
	output_dir --> (str) path for the output directory.

	Returns:
	----------
	(list) command to run multimer prediction in subprocess acceptable format.
	"""
	return [
		"python3", f"{script}",
		f"{fasta_dir}",
		"/data/alpha-fold-db/pdb_mmcif/mmcif_files/",
		"--use_precomputed_alignments", f"{alignment_dir}",
		"--uniref90_database_path", "/data/alpha-fold-db/uniref90/uniref90.fasta",
		"--mgnify_database_path", "/data/alpha-fold-db/mgnify/mgy_clusters_2022_05.fa",
		"--pdb_seqres_database_path", "/data/alpha-fold-db/pdb_seqres/pdb_seqres.txt",
		"--uniref30_database_path", "/data/alpha-fold-db/uniref30/UniRef30_2021_03",
		"--uniprot_database_path", "/data/alpha-fold-db/uniprot/uniprot.fasta",
		"--bfd_database_path", "/data/alpha-fold-db/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
		"--jackhmmer_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/jackhmmer",
		"--hhblits_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hhblits",
		"--hmmsearch_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmsearch",
		"--hmmbuild_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/hmmbuild",
		"--kalign_binary_path", "/home/kartik/miniforge3/envs/il_ofold/bin/kalign",
		"--config_preset", f"{config_preset}",
		"--output_dir", f"{output_dir}",
		"--save_outputs",
		"--cpus", f"{cpu_cores}",
		"--model_device", f"{device}",
		"--cif_output"
	]


