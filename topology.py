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

# For loss functions.
length_scale = 10.0
eps = 1e-8

def topology_dict() -> mlc.ConfigDict:
	"""
	returns a config dictionary as an mlc.ConfigDict.
	"""
	c = copy.deepcopy( config )
	return c

config = mlc.ConfigDict(
	{
	"objective": "8wtd: testing new pose sampling + recycling model. " +
		"",
	"system": {},
	# Change the paths according to the system.
	"system_representation": {
		"init_model_prefix": "_relaxed",
		"ofold_dir": os.path.join( os.path.abspath( "./openfold/" ) ),
		"ofold_script": os.path.abspath( "./openfold/run_pretrained_openfold.py" ),
		"config_preset": "model_1_multimer_v3",
		"ofold_params": os.path.join(
						os.path.abspath(
							f"openfold/openfold/resources/params/params_model_1_multimer_v3.npz"
							)
						),
		"model_checkpoint": None,
		"jax_params_path": os.path.join(
						os.path.abspath(
							f"openfold/openfold/resources/params/params_model_1_multimer_v3.npz"
							)
						),
		"tool_base": tool_base,
		"db_dir": db_dir, # Path to the parent directory containing the alphafold databases.
		"db_preset": "full_dbs", # Use full or reduced database (full_dbs/ reduced_dbs).
		"is_multimer": True,
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
		# structure_module_finetuning, pair_perturbation, single_perturbation
		"name": "pose_recycling",
		"long_sequence_inference": long_sequence_inference,
		"use_deepspeed_evoformer_attention": use_deepspeed_evoformer_attention,
		"rigid_type": "chains",
		## train or eval.
		#"mode": {
		#	"sm": "eval",
		#	"plddt": "eval",
		#	"distogram": "eval"
		#	},
		#"update_params": {
		#	# No. of structure module blocks.
		#	"sm_no_blocks": 8,
		#	},
		## If true, freeze weights for StructureModule.
		#"freeze_sm": False,
		## Required for single_perturbation/pair_perturbation
		#"adapter": {
		#	# linear_perturb/sigmoid_gating/tanh_gating/lora/film
		#	"name": "",
		#	# use bias in Linear layer for adapter.
		#	"bias": False,
		#	# reduced feature dim size for lora adpater
		#	"lora_k": 64,
		#	# controls the magnitude of perturbation.
		#	"alpha": 0.9,
		#	# Only for pair_perturbation. Mask intra-chain contacts in pair_rep.
		#	"inter_mask": True
		#}
	},
	"loss": {
		# For each loss, enabled allows loss computation and add_penalty allows it be used for backprop.
		# FAPE, aupervised_chi, violation, chain_center_of_mass, distogram taken directly from OpenFold configs.
		"fape": {
			"enabled": False,
			"add_penalty": False,
			# For monomer.
			"backbone": {
				"clamp_distance": 10.0,
				"loss_unit_distance": 10.0,
				"weight": 0.5
			},
			# For multimer.
			"intra_chain_backbone": {
				"enabled": True,
				"clamp_distance": 10.0,
				"loss_unit_distance": 10.0,
				"weight": 0.5
			},
			# For multimer.
			"interface_backbone": {
				"enabled": True,
				"clamp_distance": 30.0,
				"loss_unit_distance": 20.0,
				"weight": 0.5
			},
			# For both monomer and multimer.
			"sidechain": {
					"clamp_distance": 10.0,
					"length_scale": length_scale,
					"weight": 0.5
			},
		"eps": 1e-4, # as in OpenFold
		"weight": 1.0,
		},
		"supervised_chi": {
			"enabled": False,
			"add_penalty": False,
			"chi_weight": 0.5,
			"angle_norm_weight": 0.01,
			"eps": eps,
			"weight": 1.0,
		},
		#"violation": {
		#	"enabled": False,
		#	"add_penalty": False,
		#	"violation_tolerance_factor": 12.0,
		#	"clash_overlap_tolerance": 1.5,
		#	"average_clashes": True,
		#	"weight": 0.03,
		#	"eps": eps
		#},
		"chain_center_of_mass": {
			"enabled": False,
			"add_penalty": False,
			"clamp_distance": -4.0,
			"weight": 0.05,
			"eps": eps
		},
		"distogram": {
			"enabled": False,
			"add_penalty": False,
			"min_bin": 2.3125,   # From OpenFold
			"max_bin": 21.6875,   # From OpenFold
			"no_bins": 64,
			"weight": 0.3,
			"eps": eps,  # 1e-6,
		},
		"rigid_chain": {
			"enabled": False,
			"add_penalty": False,
			"length_scale": length_scale,
			"weight": 0.03,
			"eps": eps
		},
		# GaussianDistanceRestraint
		"gdr": {
			"enabled": False,
			"add_penalty": False,
			"length_scale": length_scale,
			"weight": 1e-3,
			"eps": eps
		},
		"violation":{
			"enabled": True,
			"add_penalty": True,
			"ev": {
				"intra_chain_dist": 2.0,
				"inter_chain_dist": 4.0,
				"weight": 1.0
			},
			"sc": {
				"inter_res_dist": 4.0,
				"tolerance_sigma": 0.5,
				"weight": 1.0
			},
			"weight": 1.0,
			"length_scale": length_scale,
			"eps": eps
		},
		"xlr": {
			"enabled": True,
			"add_penalty": True,
			"type": "ub_harmonic", # ub_harmonic/pseudo_huber
			"func_form": "mse",   # mse, rmse
			"huber_delta": 5,
			"length_scale": length_scale,
			"weight": 1.0,
			"eps": eps
		},
	},
	"metrics": {
		"xlr": {
			"enabled": True
		}
	},
	"analysis": {
		# Run analysis pipeline.
		"enabled": False,
		# Run relaxation and MolProbity validation.
		"enable_relax_validate": True,
		# Metrics to include for model_selection.
		"assessment_metrics": ["loss-violation", "metrics-xlr"],
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
					"quantiles": {"xlr": 0.75, "violation": 0.25}
				},
				"nds": {
					"enabled": False,
				}
			},
			"scale_data": False
		},
		"structural_similarity": {
			"usalign_script": "USalign",
			"prot": "prot",
			"mm": 1,
			"ter": 1,
			"similarity_cutoff": 1.0,
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
			}
	},
	"train": {
		# Version for the modeling run.
		"version": 0,
		"skip_pose_sampling": False,
		"fill_none": True,  # If skipping pose sampling, replace final_atom_positions with None.
		"max_epochs": 100, # max epochs for sampling.
		"max_pose_iters": 20, # max epochs for pose sampling.
		"struct_format": "pdb", # output file format (pdb/cif).
		#"allow_mcpa": True, # use multi-chain permutation align
		#"allow_grad_update": True, # allow gradient update - to be deprecated.
		"device": "cuda:0" # CUDA device to be used.
	}
}
)
