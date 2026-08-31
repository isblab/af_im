"""
Configs for creating the dataset and analysis.
"""
import os, copy
import ml_collections as mlc

db_dir = "/data/alpha-fold-db"
tool_base = "/home/kartik/miniforge3/envs/il_ofold/bin/"
long_sequence_inference = False
use_deepspeed_evoformer_attention = False
is_multimer = True
if is_multimer:
	ofold_config_preset = "model_1_multimer_v3"
else:
	ofold_config_preset = "model_1_ptm"

def get_config_dict(
	db_dir = "/data/alpha-fold-db/",
	tool_base = "/home/kartik/miniforge3/envs/il_ofold/bin/",
	long_sequence_inference = False,
	use_deepspeed_evoformer_attention = False,
	is_multimer = True
	) -> mlc.ConfigDict:
	"""
	returns a config dictionary as an mlc.ConfigDict.
	"""
	if is_multimer:
		ofold_config_preset = "model_1_multimer_v3"
	else:
		ofold_config_preset = "model_1_ptm"

	c = copy.deepcopy( config )

	return c

config = mlc.ConfigDict(
	{
	"prng_seed": 1,  # seed for PRNGs.
	"benchmark": {
		"datasets": {
			# PDB benchmark from AFUnmasked.
			"afu": os.path.join( "../raw/af_unmasked_pdb_benchmark.txt" ),
			# Antibody-antigen complexes from SAbDAb.
			"sabdab": os.path.join( "../raw/sabdab_seqid70_res4.tsv" ),
			# Antibody-antigen complexes from Foldbench.
			"foldbench_abag": os.path.join( "../raw/interface_antibody_antigen.csv" ),
			# Protein-protein complexes from Foldbench.
			"foldbench_prot_prot": os.path.join( "../raw/interface_protein_protein.csv" ),
			# Protein-peptide complexes from Foldbench.
			"foldbench_prot_pep": os.path.join( "../raw/interface_protein_peptide.csv" ),
			# Protein complexes from PINDER-S.
			"pinder_s": os.path.join( "../raw/pinder_s.txt" ),
			# Protein complexes from AF-multimer benchmark.
			"afmb": os.path.join( "../raw/afm_benchmark/" ),
			# Multi-state benchmark (monomers)
			"multi_state": os.path.join(
				os.path.abspath( "./raw/Supplementary_Table_1_91_apo_holo_pairs.csv" )
				)
		},
		"globals": {
			# The PATH specified here wrt the /data/ dir.
			"base_dir": "../benchmark/",
			# Name for the benchmark.
			"benchmark_name": "crosslink",
			# cif/pdb/both
			"struct_format": "both",
			# If True, download the assembly struct else the asym struct.
			"download_assembly": True,
			# Max allowed length of the system.
			"max_sys_length": 1400,
			# Fraction of residues present in each chain of the system.
			"frac_coverage": 0.99,
			# Max cores to be used for parallelization.
			"cores": 100,
			# Max no. of trials for HTTP requests.
			"max_trials": 5,
			# Rest time between consecutive HTTP request trials.
			"wait_time": 10,
		},
		"jwalk": {
			# JWalk executable.
			"jwalk_exec": "jwalk",
			# Ca-Ca distance for short cross-linkers.
			"short_linker": 20,
			# Ca-Ca distance for long cross-linkers.
			"long_linker": 30,
			# [Min, Max] no. of inter-protein XLs (both short and long).
			"num_inter_xls": 8,
			# Fraction of TP and FP XLs to be selected.
			"frac_tp_fp": [0.9, 0.1]
		}
	},
	"msa": {
		"init_model_prefix": "_relaxed",
		"save_feature_dicts": True,
		"ofold_dir": os.path.join( os.path.abspath( "./openfold/" ) ),
		"config_preset": ofold_config_preset,
		# "ofold_params": os.path.join(
		# 					f"/home/kartik/Documents/IMP_Rewired/imp_dl/openfold/openfold/resources/params/params_model_1_multimer_v3.npz"
		# 				),
		"model_checkpoint": None,
		"jax_params_path": os.path.join(
							f"/home/kartik/Documents/IMP_Rewired/imp_dl/openfold/resources/params/params_{ofold_config_preset}.npz"
						),
		"tool_base": tool_base,
		"db_dir": db_dir, # Path to the parent directory containing the alphafold databases.
		"db_preset": "full_dbs", # Use full or reduced database (full_dbs/ reduced_dbs).
		"is_multimer": is_multimer,
		"max_template_date": "2023-01-01",
		"cpu_cores": 50,  # CPU cores to be used for OpenFold run.
		"subtract_plddt": True,  # 100-pLDDT as a proxy for b-factor.
		"long_sequence_inference": long_sequence_inference,
		"use_deepspeed_evoformer_attention": use_deepspeed_evoformer_attention,
		"skip_relaxation": False,
		"databases_n_tools": {
			"template_mmcif_dir": os.path.join( db_dir, "pdb_mmcif/mmcif_files" ),
			"uniref90_database_path": os.path.join( db_dir, "uniref90/uniref90.fasta" ),
			"mgnify_database_path": os.path.join( db_dir, "mgnify/mgy_clusters_2022_05.fa" ),
			"pdb70_database_path": os.path.join( db_dir, "pdb70/pdb70" ),
			"uniclust30_database_path": os.path.join( db_dir, "uniclust30/uniclust30_2018_08/uniclust30_2018_08" ),
			"pdb_seqres_database_path": os.path.join( db_dir, "pdb_seqres/pdb_seqres.txt" ),
			"uniref30_database_path": os.path.join( db_dir, "uniref30/UniRef30_2021_03" ),
			"uniprot_database_path": os.path.join( db_dir, "uniprot/uniprot.fasta" ),
			"bfd_database_path": os.path.join( db_dir,"bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" ),
			"jackhmmer_binary_path": os.path.join( tool_base, "jackhmmer" ),
			"hhblits_binary_path": os.path.join( tool_base, "hhblits" ),
			"hmmbuild_binary_path": os.path.join( tool_base, "hmmbuild" ),
			"hmmsearch_binary_path": os.path.join( tool_base, "hmmsearch" ),
			"hhsearch_binary_path": os.path.join( tool_base, "hhsearch" ),
			"kalign_binary_path": os.path.join( tool_base, "kalign" )
		}
	},
	"models": {
		# The PATH specified here wrt the /data/ dir.
		# "base_dir": "./benchmark/",
		"base_dir": "/data2/kartik/IMP_Rewired/im_bench/imp_dl/benchmark",
	}

#----------#
}
)
