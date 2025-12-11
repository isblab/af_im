"""
Cotains the configs for an experiment.
"""

pinderS_hyperparameters = {

	## -------------------------------------------------------------------------------- ##
	"experiment_2": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=25. No recycles. TP XLs only.",
		"benchmark_name": "pinderS",
		"modeling_version": 2,
		"sys_conf_suff": "",
		"device": "cuda:0",
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
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": False}
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
	## -------------------------------------------------------------------------------- ##
	"experiment_1": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100. No recycles. TP+FP XLs.",
		"benchmark_name": "pinderS",
		"modeling_version": 1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
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
	## -------------------------------------------------------------------------------- ##
	"experiment_0.2": {
		"modeling_objective": "Pose sampling+MSA subsampling. Assessing performance on PINDER-S benchmark. No recycling.",
		"benchmark_name": "pinderS",
		"modeling_version": 0.2,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
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
		"reinit_rep": ["init", "init"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
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
	## -------------------------------------------------------------------------------- ##
	"experiment_0.1": {
		"modeling_objective": "MSA subsampling alone. Assessing performance on PINDER-S benchmark. No recycling.",
		"benchmark_name": "pinderS",
		"modeling_version": 0.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
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
	## -------------------------------------------------------------------------------- ##
	"experiment_0": {
		"modeling_objective": "Pose sampling alone. Assessing performance on PINDER-S benchmark. No recycling.",
		"benchmark_name": "pinderS",
		"modeling_version": 0,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
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
		"reinit_rep": ["init", "init"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
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
	}

}