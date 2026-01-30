"""
Cotains the configs for an experiment.
"""

xlmerged_hyperparameters = {

    ## -------------------------------------------------------------------------------- ##
    "experiment_2": {
        "modeling_objective": "Pose sampling with uvd features. gated_harmonic loss.",
        "benchmark_name": "xlmerged",
        "modeling_version": 2,
		"input_feats": "uvd",
		"c_hidden": 16,
        "sys_conf_suff": "_tpfp",
        "device": "cuda:0",
        "enable_relax_validate": True,
        "num_frames": 1000,
        "num_recycles": 1,
        "sample_random_pose": False,
        "init_coord": "init",
        "reinit_frame": "prev_frame",
        "xlr": {
            "enabled": True, "add_penalty": True,"type": "gated_harmonic",
            "func_form": "mse", "beta": 5.0, "weight": 1.0
            },
        "model_selection": {
                "quant_filter": {
                    "enabled": True,
                    "quantiles": {"xlr": 0.9, "violation": 1.0}
                }
        }
    },
    ## -------------------------------------------------------------------------------- ##
    "experiment_1.1": {
        "modeling_objective": "Pose sampling only. XLR as gated_harmonic loss with beta=10.0. num_frames=50",
        "benchmark_name": "xlmerged",
        "modeling_version": 1.1,
		"input_feats": "com",
		"c_hidden": 16,
        "sys_conf_suff": "_tpfp",
        "device": "cuda:1",
        "enable_relax_validate": True,
        "num_frames": 50,
        "num_recycles": 1,
        "sample_random_pose": False,
        "init_coord": "init",
        "reinit_frame": "prev_frame",
        "xlr": {
            "enabled": True, "add_penalty": True,"type": "gated_harmonic",
            "func_form": "mse", "beta": 5.0, "weight": 1.0
            },
        "model_selection": {
                "quant_filter": {
                    "enabled": True,
                    "quantiles": {"xlr": 0.9, "violation": 1.0}
                }
        }
    },
	## -------------------------------------------------------------------------------- ##
	"experiment_1": {
		"modeling_objective": "Pose sampling only. XLR as gated_harmonic loss with beta=5.0. num_frames=50",
		"benchmark_name": "xlmerged",
		"modeling_version": 1,
		"input_feats": "com",
		"c_hidden": 16,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"enable_relax_validate": True,
		"num_frames": 50,
		"num_recycles": 1,
		"sample_random_pose": False,
		"init_coord": "init",
		"reinit_frame": "prev_frame",
		"xlr": {
			"enabled": True, "add_penalty": True,"type": "gated_harmonic",
            "func_form": "mse", "beta": 5.0, "weight": 1.0
			},
		"model_selection": {
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.9, "violation": 1.0}
				}
		}
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_0": {
		"modeling_objective": "Pose sampling only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 0,
		"input_feats": "com",
		"c_hidden": 16,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"enable_relax_validate": True,
		"num_frames": 50,
		"num_recycles": 1,
		"sample_random_pose": False,
		"init_coord": "init",
		"reinit_frame": "prev_frame",
		"xlr": {
			"enabled": True, "add_penalty": True,"type": "ub_harmonic",
			"func_form": "mse", "weight": 1.0
			},
		"model_selection": {
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.9, "violation": 1.0}
				}
		}
	},

}
