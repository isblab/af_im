"""
Initialize and return the required model.
"""
import ml_collections as mlc
from models.sm_finetuning import StructureModuleFineTuning
from models.rep_perturb import PairPerturbation, SinglePerturbation
from models.rigid_sampler import PoseSampling

def get_model( model_config: mlc.ConfigDict, system_features: mlc.ConfigDict,
						ofold_config: mlc.ConfigDict,
						mode: str, is_multimer: bool, device: str ):
	"""
	Return the required model.
	"""
	if model_config.name == "structure_module_finetuning":
		print( "Using StructureModuleFineTuning" )
		model = StructureModuleFineTuning( system_features, ofold_config, model_config,
											mode, is_multimer, device )
	elif model_config.name == "pair_perturbation":
		print( f"Using PairPerturbation with adapter = {model_config.adapter.name}" )
		model = PairPerturbation( system_features, ofold_config, model_config,
											mode, is_multimer, device )
	elif model_config.name == "single_perturbation":
		print( f"Using SinglePerturbation with adapter = {model_config.adapter.name}" )
		model = SinglePerturbation( system_features, ofold_config, model_config,
											mode, is_multimer, device )
	elif model_config.name == "rigid_transform":
		print( f"Using RigidTransformation = {model_config.name}" )
		model = PoseSampling( model_config, device )
	else:
		raise ValueError( "Incorrect model type specified..." )

	return model