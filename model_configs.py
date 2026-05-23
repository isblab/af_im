"""
Contains configurations for the models to be tested.
The configs are named after Greek letters.
"""
import ml_collections as mlc

BOLTZ = mlc.ConfigDict(
{
	"alpha": {
		# Unguided
		"xl_type": None,          # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "unguided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"alpha2": {
		# Unguided with reduced step_scale
		"xl_type": None,          # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "unguided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.0   # default = 1.638
	},
	"beta1": {
		# XL max bound = 20 + 10% FP XLs
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"beta2": {
		# XL max bound = 20 + 25% FP XLs
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.25,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"beta3": {
		# XL max bound = 20 + 50% FP XLs
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"beta4": {
		# XL max bound = 20 + No FP XLs used
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"gamma": {
		# XL max bound = 30 + 10 % FP XLs
		"xl_type": "long",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# No. of diffusion sampling steps.
		"sampling_steps": 200  # default = 200
	},
	"delta1": {
		# Modifying MSA subsampling
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": True,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 30,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"delta2": {
		# Modifying MSA subsampling
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": True,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 128,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"delta3": {
		# Modifying MSA subsampling
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": True,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 256,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"epsilon1": {
		# Modifying step_scale
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.0   # default = 1.638
	},
	"epsilon2": {
		# Modifying step_scale +25% FP XLs
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.25,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.0   # default = 1.638
	},
	"epsilon3": {
		# Modifying step_scale +50% FP XLs
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.0   # default = 1.638
	},
	"epsilon4": {
		# Modifying step_scale + no FP XLs
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.0   # default = 1.638
	},
	"zeta1": {
		# Modifying recycling iterations
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 1,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"eta1": {
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 50,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 1,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.638   # default = 1.638
	},
	"theta1": {
		# Modifying recycling iterations + step_scale
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 1024,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 1,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.0   # default = 1.638
	},
	"iota1": {
		# Modifying step_scale + MSA subsampling
		"xl_type": "short",     # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# If True, use MSA subsampling.
		"subsample_msa": False,  # default = False
		# No. of MSA seq to subsample.
		"num_subsampled_msa": 30,  # default = 1024
		# No. of recycling iterations.
		"recycling_steps": 3,  # default = 3
		# No. of samples to predict.
		"diffusion_samples": 25,  # default = 25
		# No. of diffusion sampling steps.
		"sampling_steps": 200,  # default = 200
		# max no. of samples to predict in parallel.
		"max_parallel_samples": 5,  # default = 5
		# Related to the temperature at which diffusion samples the distribution.
		"step_scale": 1.0   # default = 1.638
	},

})


################################################################################
################################################################################
ALPHALINK = mlc.ConfigDict(
{
	"alpha": {},
	"beta1": {
		# XL max bound = 20 + 10% FP XLs
		"xl_type": "short",          # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # defaulrt = 20
		"num_samples": 25,  # defaulrt = 25
		"msa_neff": -1,  # defaulrt = -1
		"drop_xls": -1  # defaulrt = -1
	},
	"beta2": {
		# XL max bound = 20 + 25% FP XLs
		"xl_type": "short",          # short/long/None
		"frac_fp": 0.25,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # defaulrt = 20
		"num_samples": 25,  # defaulrt = 25
		"msa_neff": -1,  # defaulrt = -1
		"drop_xls": -1  # defaulrt = -1
	},
	"beta3": {
		# XL max bound = 20 + 50% FP XLs
		"xl_type": "short",          # short/long/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # defaulrt = 20
		"num_samples": 25,  # defaulrt = 25
		"msa_neff": -1,  # defaulrt = -1
		"drop_xls": -1  # defaulrt = -1
	},
	"delta1": {
		# Modifying MSA subsampling
		"xl_type": "short",          # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # defaulrt = 20
		"num_samples": 25,  # defaulrt = 25
		"msa_neff": 30,  # defaulrt = -1
		"drop_xls": -1  # defaulrt = -1
	},
	# "gamma": {
	# 	"xl_type": "long",      # short/long/None
	# 	"frac_fp": 0.1,   # fraction of FP XLs
	# 	"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
	# 	"pred_type": "guided",  # guided/unguided
	# 	# Use templates before this date.
	# 	"max_template_date": "2020-05-01", # default
	# 	"recycling_iters": 20,  # defaulrt = 20
	# 	"num_samples": 25,  # defaulrt = 25
	# 	"msa_neff": -1,  # defaulrt = -1
	# 	"drop_xls": -1  # defaulrt = -1
	# },
})


################################################################################
################################################################################
GRASP = mlc.ConfigDict(
{
	"alpha": {
		# Unguided
		"xl_type": None,          # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "unguided",  # guided/unguided
	},
	"beta1": {
		# XL max bound = 20 + 10% FP XLs
		"xl_type": "short",      # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
	},
	"beta2": {
		# XL max bound = 20 + 25% FP XLs
		"xl_type": "short",      # short/long/None
		"frac_fp": 0.25,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
	},
	"beta3": {
	# XL max bound = 20 + 50% FP XLs
		"xl_type": "short",      # short/long/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
	},
	"gamma": {
		# XL max bound = 30 + 10 % FP XLs
		"xl_type": "long",      # short/long/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
	}

})