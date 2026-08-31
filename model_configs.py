"""
Contains configurations for the models to be tested.
The configs are named after Greek letters.
"""
import ml_collections as mlc

BOLTZ = mlc.ConfigDict(
{
	"alpha": {
		# Unguided
		"xl_type": None,          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "unguided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
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
	"beta1": {
		# XL max bound = 20 + 10% FP XLs
		"xl_type": "short",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
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
		"xl_type": "short",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
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
	"delta1": {
		# Modifying MSA subsampling
		"xl_type": "short",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
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
	"epsilon1": {
		# Modifying step_scale
		"xl_type": "short",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
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
		"xl_type": "short",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
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
	"kappa1": {
		# Multi-state: beta1 config with only state1 XLs
		"xl_type": "S1",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
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
	"kappa2": {
		# Multi-state: beta1 config with only state2 XLs
		"xl_type": "S2",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
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
	"kappa3": {
		# Multi-state: beta1 config with only state1+2 XLs
		"xl_type": "S1_2",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
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
	"mu1": {
		# Multi-state: delta1 config with only state1 XLs
		"xl_type": "S1",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
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
	"mu2": {
		# Multi-state: delta1 config with only state2 XLs
		"xl_type": "S2",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
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
	"mu3": {
		# Multi-state: delta1 config with only state1+2 XLs
		"xl_type": "S1_2",     # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
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

})


################################################################################
################################################################################
ALPHALINK = mlc.ConfigDict(
{
	"alpha": {},
	"beta1": {
		# XL max bound = 20 + 10% FP XLs
		"xl_type": "short",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": -1,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"beta3": {
		# XL max bound = 20 + 50% FP XLs
		"xl_type": "short",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": -1,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"delta1": {
		# Modifying MSA subsampling
		"xl_type": "short",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": 30,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"kappa1": {
		# Multistate: only state1 XLs
		"xl_type": "S1",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": -1,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"kappa2": {
		# Multistate: only state2 XLs
		"xl_type": "S2",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": -1,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"kappa3": {
		# Multistate: state1+2 XLs
		"xl_type": "S1_2",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": -1,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"mu1": {
		# Multistate: only state1 XLs with MSA subsmapling
		"xl_type": "S1",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": 30,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"mu2": {
		# Multistate: only state2 XLs with MSA subsmapling
		"xl_type": "S2",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": 30,  # default = -1
		"drop_xls": -1  # default = -1
	},
	"mu3": {
		# Multistate: state1+2 XLs with MSA subsmapling
		"xl_type": "S1_2",          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",  # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # default = 20
		"num_samples": 25,  # default = 25
		"msa_neff": 30,  # default = -1
		"drop_xls": -1  # default = -1
	},

})


################################################################################
################################################################################
GRASP = mlc.ConfigDict(
{
	"alpha": {
		# Unguided
		"xl_type": None,          # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "unguided",  # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
		"num_models_multi": 5, # No. of preds per model for multimer
	},
	"beta1": {
		# XL max bound = 20 + 10% FP XLs
		"xl_type": "short",      # short/long/S1/S2/S1_2/None
		"frac_fp": 0.1,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
		"num_models_multi": 5, # No. of preds per model for multimer
	},
	"beta3": {
	# XL max bound = 20 + 50% FP XLs
		"xl_type": "short",      # short/long/S1/S2/S1_2/None
		"frac_fp": 0.5,   # fraction of FP XLs
		"no_fp_xls": False,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
		"multi_state": False,  # If True, run multi-state benchmark prediction
		"num_models_multi": 5, # No. of preds per model for multimer
	},
	"kappa1": {
		# Multistate: only state1 XLs
		"xl_type": "S1",      # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		"num_models_multi": 5, # No. of preds per model for multimer
	},
	"kappa2": {
		# Multistate: only state1 XLs
		"xl_type": "S2",      # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		"num_models_multi": 5, # No. of preds per model for multimer
	},
	"kappa3": {
		# Multistate: only state1 XLs
		"xl_type": "S1_2",      # short/long/S1/S2/S1_2/None
		"frac_fp": 0.0,   # fraction of FP XLs
		"no_fp_xls": True,  # if True, ignore using FP XLs as restraints
		"pred_type": "guided",   # guided/unguided
		"multi_state": True,  # If True, run multi-state benchmark prediction
		"num_models_multi": 5, # No. of preds per model for multimer
	},


})