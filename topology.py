"""
Define the configurations for modeling the system.
Essentially, this script will create the topology file for modeling.
"""
import os, copy
import ml_collections as mlc

db_dir = "/data/alpha-fold-db/"
tool_base = "/home/kartik/miniforge3/envs/il_ofold/bin/"
long_sequence_inference = False
use_deepspeed_evoformer_attention = False
is_multimer = True
if is_multimer:
	ofold_config_preset = "model_1_multimer_v3"
else:
	ofold_config_preset = "model_1_ptm"

# For loss functions.
length_scale = 10.0  # angstorm to nm conversion.
eps = 1e-8

def topology_dict(
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
	"objective": "4rhz: " +
		"",
	"system": {},
	# Change the paths according to the system.
	"system_representation": {
		"init_model_prefix": "_relaxed",
		"save_feature_dicts": True,
		"create_restraint_feats": True,
		"ofold_dir": os.path.join( os.path.abspath( "./openfold/" ) ),
		"ofold_script": os.path.abspath( "./openfold/run_pretrained_openfold.py" ),
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
		"seed": 1,  # seed for PRNGs.
		"cpu_cores": 16,  # CPU cores to be used for OpenFold run.
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
	"optimizer": {
		"SGD": {
			"enabled": False,
			"lr": 1e-1,
			"momentum": 0,
			"weight_decay": 0
		},
		"Adam": {
			"enabled": True,
			"lr": 1e-3,
			"amsgrad": False,
			"weight_decay": 0
		},
		"AdamW": {
			"enabled": False,
			"lr": 1e-3,
			"amsgrad": False,
			"weight_decay": 0
		}
	},
	"model": {
		"name": "pose_sampling",
		"long_sequence_inference": long_sequence_inference,
		"use_deepspeed_evoformer_attention": use_deepspeed_evoformer_attention,
		# Currently only splits by chain.
		"rigid_type": "chains",
		# If True predict omgea else predict a quaternion.
		"use_axis_angle": False,
		# K for selecting the TopK closest interafce residue pairs.
		"hist_bins": 16,
		"hist_range": [0, 50],
		# Step size for rotation when using axis-angle.
		"rot_step_size": 0.1,
		# Step size for translation.
		"trans_step_size": 0.1,
		# For feature creation, use the updated conformation.
		"sequential_update": False,
		"num_recycles": 1,  # no. of recycling iterations for OpenFold.
	},
	"loss": {
		# For each loss, enabled allows loss computation and add_penalty allows it be used for backprop.
		"violation": {
			"enabled": True,
			"add_penalty": True,
			"ev": {
				"intra_enabled": False,
				"inter_enabled": True,
				# # Distance between the Ca of intra-chain residues.
				# "intra_chain_dist": 4.0,
				# # Distance between the Ca of inter-chain residues.
				# "inter_chain_dist": 8.0,
				# Tolerance distance to consider a clash.
				"clash_tolerance": 4.0,
				# Controls the smoothness of softplus around the boundary.
				"beta": 0.5,
				"weight": 1.0,
			},
			"sc": {
				"enabled": False,
				# Ca-Ca distance for adjacent residues.
				"inter_res_dist": 4.0,
				"weight": 1.0,
			},
			"length_scale": length_scale,
			"weight": 1.0,
			"eps": eps
		},
		# "violation": {
		# 	"enabled": True,
		# 	"add_penalty": True,
		# 	"violation_tolerance_factor": 12.0,
		# 	"clash_overlap_tolerance": 1.5,
		# 	"average_clashes": True,
		# 	"weight": 1.0,
		# 	"eps": eps
		# },
		"xlr": {
			"enabled": True,
			"add_penalty": True,
			"type": "softplus", # ub_harmonic/pseudo_huber
			"func_form": "mse",   # mse, rmse
			"huber_delta": 5,
			"beta": 0.5,  # for softplus and gated_harmonic loss only.
			"allow_xl_tolerance": False,
			"length_scale": length_scale,
			"weight": 1.0,
			"eps": eps
		},
		"com": {
			"enabled": False,
			"add_penalty": False,
			"weight": 1e-1,
			"eps": eps
		}
	},
	"metrics": {
		"ev": {
			"enabled": True,
			"intra_enabled": False,
			"inter_enabled": True,
			# Distance between the Ca of intra-chain residues.
			"intra_chain_dist": 4.0,
			# Distance between the Ca of inter-chain residues.
			"inter_chain_dist": 8.0,
			"eps": eps
		},
		"xlr": {
			"enabled": True,
			"allow_xl_tolerance": False,
			"length_scale": length_scale,
			"eps": eps
		}
	},
	"analysis": {
		# Run analysis pipeline.
		"enabled": True,
		# Run relaxation and MolProbity validation.
		"enable_relax_validate": True,
		# Metrics to include for model_selection.
		"assessment_metrics": ["loss-violation", "metrics-xlr"],
		"burn_in_frac": 0.2,
		"model_selection": {
			"method": {
				"kmeans": {
					"enabled": False,
					"n_cluster": 2,
					"random_state": 1,
					"n_init": "auto"
				},
				"gmm": {
					"enabled": False,
					"n_components": 2,
					"random_state": 1
				},
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.9, "violation": 1.0}
				},
				"nds": {
					"enabled": False,
				}
			},
			"scale_data": False
		},
		"structural_similarity": {
			# "tool": "usalign",  # usalign/mdanalysis
			"usalign_script": "USalign",
			"mol": "prot",
			"mm": 1,
			"ter": 1,
			"metric": "tm", # rmsd or tm-score
			# TM cutoff from CombFold; 4 A as per this cutoff (doi: 10.1002/prot.26818)
			"similarity_cutoff": 0.7,
			# No. of CPU cores to be used for relaxation.
			"cpu_cores": 50,
			"clean_up": True
		},
		"relax":{
			# Parameters taken from OpenFold configs.
			"max_iterations": 0,  # no max
			"tolerance": 2.39,
			"stiffness": 10.0,
			"exclude_residues": [],
			"max_outer_iterations": 20,
			# If True, use GPU else CPU.
			"use_gpu": True,
			"parallelize": False, # Do not use with GPU.
			# No. of CPU cores to be used for relaxation.
			"cpu_cores": 16,
			"output_format": "pdb",
			"relaxed_model_dir": "relaxed_models",
			# Save models as separate files.
			"save_single_model": True,
			"amber_logs_file": "Logs_amber"
			},
		"molprobity":{
			"clean_up": True # remove all temporary file upon completion.
			},
		"localization_density":{
			"cpu_cores": 10,
			"apix": 1.0,         # angstrom per voxel
			"target_prob": 0.9,  # Probability mass to define the contour for ld visulaization.
			"clean_up": True     # remove all temporary file upon completion.
			}
	},
	"train": {
		# Version for the modeling run.
		"version": None,
		"sample_random_pose": False,  # If True, perform random pose sampling.
		"init_coord": "init",  # zero/ init
		"reinit_frame": "prev_frame",  # "prev_frame": reuses final_atom_positions from previous epoch; "init": initializes as in init_rep.
		"num_frames": 50, # max epochs for sampling.
		"max_norm": 1.0,
		"struct_format": "pdb", # output file format (pdb/cif).
		"device": "cuda:0" # CUDA device to be used.
	}
}
)
