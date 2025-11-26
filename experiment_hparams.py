"""
Cotains the configs for an experiment.
"""

experiment_hyperparameters = {

	## -------------------------------------------------------------------------------- ##
	"experiment_31.1": {
		"modeling_objective": "Assessing effect of recycles. Column masking+recycles=3.",
		"benchmark_name": "experiment",
		"modeling_version": 31.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 3,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
		},
		"column_masking": {
			"enabled": True,
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
	"experiment_31": {
		"modeling_objective": "Assessing effect of recycles. Column masking+no recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 31,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
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
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
		},
		"column_masking": {
			"enabled": True,
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
	"experiment_30.1": {
		"modeling_objective": "Assessing effect of recycles. MSA subsampling+recycles=3.",
		"benchmark_name": "experiment",
		"modeling_version": 30.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 3,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "init",
		"reinit_step": "init",
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
	"experiment_30": {
		"modeling_objective": "Assessing effect of recycles. MSA subsampling+no recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 30,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
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
		"reinit_frame": "init",
		"reinit_step": "init",
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
	"experiment_29.3": {
		"modeling_objective": "inference_mode=eval; activate_dropout=structure_module." +\
			" No MSA subampling, column masking, pose sampling. TP+FP XLs." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=prev_frame.",
		"benchmark_name": "experiment",
		"modeling_version": 29.3,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"inference_mode": "train",
		"activate_dropouts": "structure_module",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_29.2": {
		"modeling_objective": f"inference_mode=eval; activate_dropout=structure_module." +\
			" No MSA subampling, column masking, pose sampling. TP+FP XLs." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=init.",
		"benchmark_name": "experiment",
		"modeling_version": 29.2,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"inference_mode": "train",
		"activate_dropouts": "structure_module",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "init",
		"reinit_step": "init",
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_29.1": {
		"modeling_objective": f"inference_mode=eval; activate_dropout=evoformer." +\
			" No MSA subampling, column masking, pose sampling. TP+FP XLs." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=prev_frame.",
		"benchmark_name": "experiment",
		"modeling_version": 29.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"inference_mode": "train",
		"activate_dropouts": "evoformer",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_29": {
		"modeling_objective": f"inference_mode=eval; activate_dropout=evoformer." +\
			" No MSA subampling, column masking, pose sampling. TP+FP XLs." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=init.",
		"benchmark_name": "experiment",
		"modeling_version": 29,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"inference_mode": "train",
		"activate_dropouts": "evoformer",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "init",
		"reinit_step": "init",
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_28.3": {
		"modeling_objective": "Assessing effect of recycles. Ful model dropouts+recycles=3.",
		"benchmark_name": "experiment",
		"modeling_version": 28.3,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 3,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "init",
		"reinit_step": "init",
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_28.2": {
		"modeling_objective": f"inference_mode=train." +\
			" No MSA subampling, column masking, pose sampling. TP+FP XLs." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=prev_frame.",
		"benchmark_name": "experiment",
		"modeling_version": 28.2,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_28.1": {
		"modeling_objective": f"inference_mode=train." +\
			" No MSA subampling, column masking, pose sampling. TP+FP XLs." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=init.",
		"benchmark_name": "experiment",
		"modeling_version": 28.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_frame": "init",
		"reinit_step": "init",
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_28": {
		"modeling_objective": f"inference_mode=train." +\
			" No MSA subampling, column masking, pose sampling. TP+FP XLs." +\
			" init_coord=init;init_rep=[init,init]. reinit_frame=init.",
		"benchmark_name": "experiment",
		"modeling_version": 28,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "init",
		"init_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": { "neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
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
	"19.5": {
		"modeling_objective": f"MSA subsampling+pose sampling." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=prev_frame; reinit_step=prev_frame.",
		"benchmark_name": "experiment",
		"modeling_version": 19.5,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "init",
		"init_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": { "neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
		},
		"column_masking": {
			"enabled": False,
			"params": {"mask_frac": [0.3] }
		},
		"model_selection": {
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.75, "violation": 0.25}
				}
		}
	},
	## -------------------------------------------------------------------------------- ##
	"19.3": {
		"modeling_objective": f"MSA subsampling+pose sampling." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=prev_frame; reinit_step=prev_step.",
		"benchmark_name": "experiment",
		"modeling_version": 19.3,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "init",
		"init_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": { "neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
		},
		"column_masking": {
			"enabled": False,
			"params": {"mask_frac": [0.3] }
		},
		"model_selection": {
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.75, "violation": 0.25}
				}
		}
	},
	## -------------------------------------------------------------------------------- ##
	"18": {
		"modeling_objective": f"MSA subsampling+column masking+pose sampling." +\
			" init_coord=zero;init_rep=[zero,zero]. reinit_frame=prev_frame; reinit_step=prev_frame.",
		"benchmark_name": "experiment",
		"modeling_version": 18,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": True,
		"init_coord": "init",
		"init_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": { "neff": [25], "eff_cutoff": 0.8, "cap_msa": True}
		},
		"column_masking": {
			"enabled": True,
			"params": {"mask_frac": [0.3] }
		},
		"model_selection": {
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.75, "violation": 0.25}
				}
		}
	},

}