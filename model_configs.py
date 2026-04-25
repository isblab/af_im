"""
Contains configurations for the models to be tested.
The configs are named after Greek letters.
"""
import ml_collections as mlc

BOLTZ = mlc.ConfigDict(
{
	"alpha": {
		"xl_type": None,          # short/long/None
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
		"sampling_steps": 200  # default = 200

	},
	"beta": {
		"xl_type": "short",     # short/long/None
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
		"sampling_steps": 200  # default = 200
	},
	"gamma": {
		"xl_type": "long",     # short/long/None
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
		"sampling_steps": 200  # default = 200
	},
})


################################################################################
################################################################################
ALPHALINK = mlc.ConfigDict(
{
	"alpha": {},
	"beta": {
		"xl_type": None,          # short/long/None
		"pred_type": "unguided",  # guided/unguided
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # defaulrt = 20
		"num_samples": 25,  # defaulrt = 25
		"msa_neff": -1,  # defaulrt = -1
		"drop_xls": -1  # defaulrt = -1
	},
	"gamma": {
		"xl_type": "long",      # short/long/None
		"pred_type": "guided",  # guided/unguided
		# Use templates before this date.
		"max_template_date": "2020-05-01", # default
		"recycling_iters": 20,  # defaulrt = 20
		"num_samples": 25,  # defaulrt = 25
		"msa_neff": -1,  # defaulrt = -1
		"drop_xls": -1  # defaulrt = -1
	},
})


################################################################################
################################################################################
GRASP = mlc.ConfigDict(
{
	"alpha": {
		"xl_type": None,          # short/long/None
		"pred_type": "unguided",  # guided/unguided
	},
	"beta": {
		"xl_type": "short",      # short/long/None
		"pred_type": "guided",   # guided/unguided
	},
	"gamma": {
		"xl_type": "long",      # short/long/None
		"pred_type": "guided",   # guided/unguided
	}
})