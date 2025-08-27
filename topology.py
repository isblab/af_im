"""
Define the configurations for modeling the system.
Essentially, this script will create the topology file for modeling.
"""
import os, copy
import ml_collections as mlc


def topology_dict() -> mlc.ConfigDict:
	"""
	returns a config dictionary as an mlc.ConfigDict.
	"""
	c = copy.deepcopy( config )
	return c

config = mlc.ConfigDict(
	{
	"objective": "8wtd: Test run with Pseudo Huber loss. " +
		"",
	"system": {},
	# Change the paths according to the system.
	"system_representation": {
		"init_model_prefix": "_unrelaxed",
		"ofold_dir": os.path.join( os.path.abspath( "./openfold/" ) ),
		"ofold_script": os.path.abspath( "./openfold/run_pretrained_openfold.py" ),
		"config_preset": "model_1_multimer_v3",
		"ofold_params": os.path.join(
						os.path.abspath(
							f"openfold/resources/params/params_model_1_multimer_v3.npz"
							)
						),
		"model_checkpoint": os.path.abspath(
							"openfold/resources/openfold_params/finetuning_ptm_2.pt"
							),
		"ofold_tools": "/home/kartik/miniforge3/envs/il_ofold/bin/",
		"db_dir": "/data/alpha-fold-db/",
		"db_preset": "full_dbs",
		"mode": "multimer",
		"max_template_date": "2023-01-01",
		"seed": 1,
		"cpu_cores": 16
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
		"name": "structure_module_finetuning",
		# train or eval.
		"mode": {
			"sm": "eval",
			"plddt": "eval",
			"distogram": "eval"
			},
		"update_params": {
			# No. of structure module blocks.
			"sm_no_blocks": 8,
			},
		# If true, freeze weights for StructureModule.
		"freeze_sm": False,
		"adapter": {
			# linear_perturb/sigmoid_gating/tanh_gating/lora/film
			"name": "",
			# use bias in Linear layer for adapter.
			"bias": False,
			# reduced feature dim size for lora adpater
			"lora_k": 64,
			# controls the magnitude of perturbation.
			"alpha": 0.9,
			# Only for pair_perturbation. Mask intra-chain contacts in pair_rep.
			"inter_mask": True
		}
	},
	"loss": {
		# For each loss, enabled allows loss computation and add_penalty allows it be used for backprop.
		# FAPE, aupervised_chi, violation, chain_center_of_mass, distogram taken directly from OpenFold configs.
		"fape": {
			"enabled": True,
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
					"length_scale": 10.0,
					"weight": 0.5
			},
		"eps": 1e-4,
		"weight": 1.0,
		},
		"supervised_chi": {
			"enabled": True,
			"add_penalty": False,
			"chi_weight": 0.5,
			"angle_norm_weight": 0.01,
			"eps": 1e-8,
			"weight": 1.0,
		},
		"violation": {
			"enabled": True,
			"add_penalty": True,
			"violation_tolerance_factor": 12.0,
			"clash_overlap_tolerance": 1.5,
			"average_clashes": True,
			"eps": 1e-8,
			"weight": 0.03
		},
		"chain_center_of_mass": {
			"enabled": True,
			"add_penalty": False,
			"clamp_distance": -4.0,
			"weight": 0.05,
			"eps": 1e-8
		},
		"distogram": {
			"enabled": False,
			"add_penalty": False,
			"min_bin": 2.3125,   # From OpenFold
			"max_bin": 21.6875,   # From OpenFold
			"no_bins": 64,
			"eps": 1e-8,  # 1e-6,
			"weight": 0.3,
		},
		"rigid_chain": {
			"enabled": True,
			"add_penalty": True,
			"length_scale": 10.0,
			"eps": 1e-8,
			"weight": 0.03
		},
		"xlr": {
			"enabled": True,
			"add_penalty": True,
			"type": "pseudo_huber", # ub_harmonic/pseudo_huber
			"func_form": "mse",   # mse, rmse
			"huber_delta": 5,
			"weight": 0.05,
			"eps": 1e-8
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
		# run relaxation and MolProbity validation.
		"enable_relax_validate": True,
		# metrics to include for model_selection.
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
			"clean_up": False
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
			# No. of CPu cores to be used for relaxation.
			"cpu_cores": 16,
			"output_format": "pdb",
			"relaxed_model_dir": "relaxed_models",
			# Save models as separate files.
			"save_single_model": True,
			"amber_logs_file": "Logs_amber"
			},
		"molprobity":{
			"clean_up": False # remove all temporary file upon completion.
			}
	},
	"train": {
		# Version for the modeling run.
		"version": None,
		"mode": "test", # deprecated
		"max_epochs": 100, # max no. of epochs for training.
		"struct_format": "pdb", # output file format (pdb/cif).
		"allow_mcpa": True, # use multi-chain permutation align
		"allow_grad_update": True, # allow gradient update - to be deprecated.
		"device": "cuda:0" # CUDS device to be used.
	}
}
)
