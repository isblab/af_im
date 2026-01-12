"""
Cotains the configs for an experiment.
"""

xlmerged_hyperparameters = {

	## -------------------------------------------------------------------------------- ##
	"experiment_20.1": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=1. No extraMSA. Using pose as template. TP+FP XLs only. viol quantile=0.75.",
		"benchmark_name": "xlmerged",
		"modeling_version": 20,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"enable_relax_validate": False,
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"use_extra_msa": False,
		"use_template_embedder": True,
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_as_templates": True,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
		"select_pose": "last",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": {"neff": [1], "eff_cutoff": 0.8, "cap_msa": False}
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
					"quantiles": {"xlr": 0.9, "violation": 0.75}
				}
		}
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_20": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=1. Using pose as template. TP+FP XLs only. viol quantile=0.75.",
		"benchmark_name": "xlmerged",
		"modeling_version": 20,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
		"use_as_templates": True,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "prev_frame",
		"reinit_step": "prev_frame",
		"select_pose": "last",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": {"neff": [1], "eff_cutoff": 0.8, "cap_msa": False}
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
					"quantiles": {"xlr": 0.9, "violation": 0.75}
				}
		}
	},
	## -------------------------------------------------------------------------------- ##
	"experiment_19.2": {
		"modeling_objective": "Use pose sampled struct as template. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 19.2,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
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
	"experiment_19.1": {
		"modeling_objective": "Pose sampling with template embedder OFF. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 19.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": False,
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
	"experiment_19": {
		"modeling_objective": "Use pose sampled struct for recycling and as template. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 19,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
		"use_as_templates": True,
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
	"experiment_18": {
		"modeling_objective": "Pose sampling+Dropouts. Template embedder switched ON. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 18,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
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
	"experiment_17": {
		"modeling_objective": "Pose sampling+Column masking mask_frac=0.3. Template embedder switched ON. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 17,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
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
			"params": {"neff": [100], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [128]}
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
	"experiment_16.1": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100. Increase num_frames to 100. Template embedder switched ON. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 16.1,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 100,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
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
	"experiment_16": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=100. Template embedder switched ON. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 16,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
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
	"experiment_15": {
		"modeling_objective": "MSA subsampling neff=254 + Extra-MSA subsampling neff=[16, 32, 64, 128, 256, 512, 1024] only. No templates. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 15,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"sample_random_pose": False,
		"use_template_embedder": False,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": True,
			"type": "random",
			"params": {"neff": [254], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": True,
			"params": {"neff": [16, 32, 64, 128, 256, 512, 1024]}
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
	"experiment_14": {
		"modeling_objective": "Pose sampling only with template embedder switched ON. No recycles. TP+FP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 14,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
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
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
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
	"experiment_13": {
		"modeling_objective": "Pose sampling only with template embedder switched OFF. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 13,
		"sys_conf_suff": "",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_template_embedder": False,
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
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
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
	"experiment_12": {
		"modeling_objective": "Pose sampling+MSA subsampling neff=256. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 12,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_template_embedder": True,
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
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
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
	"experiment_11.1": {
		"modeling_objective": "Testing sampling convergence. Sample random rigid transformations for 500 steps. num_frame=2.",
		"benchmark_name": "xlmerged",
		"modeling_version": 11.1,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 2,
		"num_steps": 500,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": True,
		"use_template_embedder": True,
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
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [16]}
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
	"experiment_11": {
		"modeling_objective": "Testing sampling convergence. Run pose sampling for 500 steps. num_frame=2.",
		"benchmark_name": "xlmerged",
		"modeling_version": 11,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 2,
		"num_steps": 500,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"sample_random_pose": False,
		"use_template_embedder": True,
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
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [16]}
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
	"experiment_10": {
		"modeling_objective": "Negative control. No pose sampling or AF smapling. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 10,
		"sys_conf_suff": "",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [16]}
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
	"experiment_9.1": {
		"modeling_objective": "Using pose as templates, add to existing templates. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 9.1,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_template_embedder": True,
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
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [16]}
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
	"experiment_9": {
		"modeling_objective": "Using pose as templates, replace existing templates. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 9,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_template_embedder": True,
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
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [16]}
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
	"experiment_8.2": {
		"modeling_objective": "Random MSA subsamling neff=256 + Extra-MSA subsampling neff=[16, 32, 64, 128, 256] only. Template embedder OFF. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 8.2,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": False,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": True,
			"type": "random",
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": True,
			"params": {"neff": [16, 32, 64, 128, 256]}
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
	"experiment_8.1": {
		"modeling_objective": "OG random MSA subsamling neff=256 + Extra-MSA subsampling neff=[16, 32, 64, 128, 256] only. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 8.1,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": True,
			"type": "random",
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": True,
			"params": {"neff": [16, 32, 64, 128, 256]}
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
	"experiment_8": {
		"modeling_objective": "MSA subsamling neff=256 + Extra-MSA subsampling neff=[16, 32, 64, 128, 256] only. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 8,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": True,
			"type": "sequential",
			"params": {"neff": [256], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": True,
			"params": {"neff": [16, 32, 64, 128, 256]}
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
	"experiment_7.1": {
		"modeling_objective": "Dropouts only. Recycles=9. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 7.1,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 9,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": False,
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
	"experiment_7": {
		"modeling_objective": "Dropouts only. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 7,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "train",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": False,
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
	"experiment_6.1": {
		"modeling_objective": "Column masking mask_frac=0.3 only. Recycles=3. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 6.1,
		"sys_conf_suff": "",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 3,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [50], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [128]}
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
	"experiment_6": {
		"modeling_objective": "Column masking mask_frac=0.3 only. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 6,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
		"reinit_step": "prev_frame",
		"select_pose": "max",
		"subsampling": {
			"enabled": False,
			"type": "sequential",
			"params": {"neff": [50], "eff_cutoff": 0.8, "cap_msa": False}
		},
		"extra_msa_subsampling": {
			"enabled": False,
			"params": {"neff": [128]}
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
	"experiment_5.2": {
		"modeling_objective": "MSA subsamling neff=50 only. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 5.2,
		"sys_conf_suff": "",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"init_rep": ["zero", "zero"],
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
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
	"experiment_5.1": {
		"modeling_objective": "MSA subsamling neff=25 only. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 5.1,
		"sys_conf_suff": "",
		"device": "cuda:1",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": True,
		"use_template_embedder": True,
		"use_as_templates": False,
		"add_to_existing_templates": False,
		"recycle_pose": False,
		"init_coord": "zero",
		"reinit_rep": ["init", "init"],
		"reinit_frame": "init",
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
	"experiment_5": {
		"modeling_objective": "Pose sampling only. No recycles. TP XLs only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 5,
		"sys_conf_suff": "",
		"device": "cuda:0",
		"num_frames": 50,
		"num_steps": 20,
		"num_recycles": 1,
		"inference_mode": "eval",
		"activate_dropouts": "none",
		"skip_pose_sampling": False,
		"use_template_embedder": True,
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
		"use_template_embedder": True,
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