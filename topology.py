"""
Define the configurations for modeling the system.
Essentially, this script will create the topology file for modeling.
"""
import copy
import ml_collections as mlc


def topology_dict():
	c = copy.deepcopy( config )
	return c

config = mlc.ConfigDict(
	{
    "system": {},
	"optimizer": {
			"SGD": {
				"enabled": False,
				"lr": 1e-1,
				"momentum": 0,
				"weight_decay": 0
			},
			"Adam": {
				"enabled": True,
				"lr": 1e-3,
				"amsgrad": False,
				"weight_decay": 0
			},
			"AdamW": {
				"enabled": False,
				"lr": 1e-3,
				"amsgrad": False,
				"weight_decay": 0
			}
	},
    "loss": {
        "fape": {
            "enabled": True,
        	# For monomer.
            "backbone": {
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "weight": 1.0 # 0.5
            },
            # For multimer.
            "intra_chain_backbone": {
            	"enabled": True,
                "clamp_distance": 10.0,
                "loss_unit_distance": 10.0,
                "weight": 0.5
            },
            # For multimer.
            "interface_backbone": {
                "enabled": False,
                "clamp_distance": 30.0,
                "loss_unit_distance": 20.0,
                "weight": 0.5
			},
			# For both monomer and multimer.
            "sidechain": {
		            "clamp_distance": 10.0,
		            "length_scale": 10.0,
		            "weight": 0.5
    		},
        "eps": 1e-4,
        "weight": 1.0,
        },
        "supervised_chi": {
            "enabled": True,
            "chi_weight": 0.5,
            "angle_norm_weight": 0.01,
            "eps": 1e-8,
            "weight": 1.0,
        },
        "violation": {
            "enabled": True,
            "violation_tolerance_factor": 12.0,
            "clash_overlap_tolerance": 1.5,
            "average_clashes": True,
            "eps": 1e-8,
            "weight": 2.0
        },
        "chain_center_of_mass": {
            "enabled": True,
            "clamp_distance": -4.0,
            "weight": 0.0,
            "eps": 1e-8
        },
        "xlr": {
                "enabled": True,
                "fape_xlr": False,
                "simple_xlr": True,
                "weight": 0.5,
                "eps": 1e-8
            },
    },
    "train": {
        "max_epochs": 50
    }
}
)

