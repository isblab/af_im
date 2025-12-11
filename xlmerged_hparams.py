"""
Cotains the configs for an experiment.
"""

xlmerged_hyperparameters = {

	## -------------------------------------------------------------------------------- ##
	"experiment_1": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100. No recycles. TP+FP XLs.",
		"benchmark_name": "xlmerged",
		"modeling_version": 1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": {"neff": [100], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [128]}
		},
		"column_masking": {
			"enabled": False,
			"params": {"mask_frac": [0.3] }
		},
		"model_selection": {
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.9, "violation": 1.0}
				}
		}
	},

}