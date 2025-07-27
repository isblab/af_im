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
	"objective": "2ayo with mse_xlr. " +
		"Phishing out PDB IDs..",
	"system": {},
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
		"name": "structure_module_finetuning",
		"mode": {
			"sm": "eval",
			"plddt": "eval",
			"distogram": "eval"
		},
		"update_params": {
			"sm_no_blocks": 8,
		},
		"biases": {
			"pair_bias_type": "additive",
			"pair_bias_scale_factor": 1.0,
		},
		"dropouts": {
			"msa": {
				"enabled": False,
				"loc": "post_evo",
				"prob": 0.0,
			},
			"pair": {
				"enabled": False,
				"loc": "post_evo",
				"prob": 0.0,
			}
		}
	},
	"loss": {
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
			"add_penalty": True,
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
		"xlr": {
				"enabled": True,
				"add_penalty": True,
				"type": "simple_xlr2", # fape_xlr, simple_xlr, mse_xlr, rmse_xlr, disto_xlr
				"func_form": "mse",   # mse, rmse
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
		"assessment_metrics": ["loss-violation", "loss-chain_center_of_mass", "metric-xlr"],
		"clustering": {
			"method": {
				"kmeans": {
					"enabled": True,
					"n_cluster": 2,
					"random_state": 1,
					"n_init": "auto"
				},
				"gmm": {
					"enabled": False,
					"n_components": 2,
					"random_state": 1
				},
				# "hdbscan": {
				# 	"enabled": False,
				# 	"min_cluster_size": 5,
				# 	"min_sample": None,
				# 	"store_centers": False
				# }
			}
		}
	}
	"train": {
		"version": 0,
		"mode": "test",
		"max_epochs": 50,
		"allow_mcpa": True,
		"allow_grad_update": True,
		"device": "cuda:0"
	}
}
)
