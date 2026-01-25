"""
Cotains the configs for an experiment.
"""

xlmerged_hyperparameters = {

	## -------------------------------------------------------------------------------- ##
	"experiment_0": {
		"modeling_objective": "Pose sampling only.",
		"benchmark_name": "xlmerged",
		"modeling_version": 0,
		"sys_conf_suff": "_tpfp",
		"device": "cuda:0",
		"enable_relax_validate": True,
		"num_frames": 50,
		"num_recycles": 1,
		"sample_random_pose": False,
		"init_coord": "init",
		"reinit_frame": "prev_frame",
		"model_selection": {
				"quant_filter": {
					"enabled": True,
					"quantiles": {"xlr": 0.9, "violation": 1.0}
				}
		}
	},

}