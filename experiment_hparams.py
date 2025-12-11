"""
Cotains the configs for an experiment.
"""

experiment_hyperparameters = {

	## -------------------------------------------------------------------------------- ##
	"experiment_41.1": {
		"modeling_objective": "Re-benchmarking MSA subsampling neff=25 with only TP+FP XLs. Pose sampling. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 41.1,
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
			"params": {"neff": [50], "eff_cutoff": 0.8, "cap_msa": False}
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
	"experiment_41": {
		"modeling_objective": "Re-benchmarking MSA subsampling neff=25 with only TP XLs. Pose sampling. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 41,
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
	"experiment_40": {
		"modeling_objective": "Sample more. num_frames=100. Pose sampling+MSA subsampling neff=100. No recycles. TP+FP XLs.",
		"benchmark_name": "experiment",
		"modeling_version": 40,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 100,
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
	"experiment_39": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100. Also template embedder. No recycles. TP+FP XLs.",
		"benchmark_name": "experiment",
		"modeling_version": 39,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_as_templates": True,
		"add_to_existing_templates": True,
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
	"experiment_38.2": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100+column masking mask_frac=0.3. No recycles. TP+FP XLs.",
		"benchmark_name": "experiment",
		"modeling_version": 38.2,
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
			"params": {"neff": [256]}
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
	"experiment_38.1": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100+Extra MSA subsampling neff=128. No recycles. TP+FP XLs.",
		"benchmark_name": "experiment",
		"modeling_version": 38.1,
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
			"enabled": True,
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
	"experiment_38": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100. No recycles. TP+FP XLs.",
		"benchmark_name": "experiment",
		"modeling_version": 38,
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
	## -------------------------------------------------------------------------------- ##
	"experiment_37.1": {
		"modeling_objective": "Testing Extra MSA subsmapling with neff=32. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 37.1,
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
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": True,
			"params": {"neff": [32]}
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
	"experiment_37": {
		"modeling_objective": "Testing Extra MSA subsmapling with neff=64. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 37,
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
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [25], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": True,
			"params": {"neff": [64]}
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
	"experiment_36": {
		"modeling_objective": "Adding pose sampled structures as template to existing templates. No MSA subsampling, column masking recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 36,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_as_templates": True,
		"add_to_existing_templates": True,
		"recycle_pose": False,
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_35": {
		"modeling_objective": "Using pose sampled structures as templates, replacing existing templates. No MSA subsampling, column masking recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 35,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_as_templates": True,
		"add_to_existing_templates": False,
		"recycle_pose": False,
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_34.1": {
		"modeling_objective": "Reusing msa and pair rep from previous frame during recycling. No MSA subsampling, column masking recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 34.1,
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
		"reinit_rep": ["prev_frame", "prev_frame"],
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_34": {
		"modeling_objective": "Reusing pair rep from previous frame during recycling. No MSA subsampling, column masking recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 34,
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
		"reinit_rep": ["init", "prev_frame"],
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_33.2": {
		"modeling_objective": "Combine Pose sampling with Dropouts. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 33.2,
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
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
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
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_33.1": {
		"modeling_objective": "Combine Pose sampling with Column masking. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 33.1,
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
	"experiment_33": {
		"modeling_objective": "Combine Pose sampling with MSA subsampling. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 33,
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
	"experiment_32.1": {
		"modeling_objective": "Assessing effect of recycles. Pose sampling only+recycles=3.",
		"benchmark_name": "experiment",
		"modeling_version": 32.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 3,
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
	"experiment_32": {
		"modeling_objective": "Assessing effect of recycles. Pose sampling only+no recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 32,
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
	"experiment_30.2": {
		"modeling_objective": "OG MSA subsampling; neff=254+extra msa subsmapling neff[16,32,64,128]. No recycles.",
		"benchmark_name": "experiment",
		"modeling_version": 30.2,
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
		"reinit_frame": "init",
		"reinit_step": "init",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": {"neff": [254], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": True,
			"params": {"neff": [16, 32, 64, 128]}
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
	"experiment_30.1": {
		"modeling_objective": "Assessing effect of recycles. MSA subsampling+recycles=3.",
		"benchmark_name": "experiment",
		"modeling_version": 30.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
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