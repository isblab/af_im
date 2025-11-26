"""
Funstions to create paths for the system or benchmark directories and files.
"""
import os


def get_sys_modeling_path(
	base_dir: str,
	modeling_dir_name: str, sys_name: str ) -> str:
	"""
	Return thr path to the modeling dir for given system.
	"""
	sys_path = os.path.join( 
				os.path.abspath(
					f"{base_dir}/{modeling_dir_name}/{sys_name}"
					)
		)
	return sys_path


def get_sys_data_dir_path(
	base_dir: str,
	benchmark_name: str, sys_name: str ) -> str:
	"""
	Return the path to the system data directory.
	"""
	data_dir = os.path.join(
		base_dir,
		f"{benchmark_name}_benchmark/{sys_name}/" )
	return data_dir


def get_benchmark_results_dir_path(
	base_dir: str,
	benchmark_name: str ) -> str:
	"""
	Return the path to the benchmark results dir.
	"""
	benchmark_results_dir = os.path.join(
		base_dir,
		f"{benchmark_name}_benchmark_results" )
	return benchmark_results_dir


def get_sys_modeling_version_path(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str, modeling_version: int ) -> str:
	"""
	Return the path to the system modeling version dir.
	"""
	sys_path = get_sys_modeling_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name )
	ver_path = os.path.join( sys_path,
						f"version_{modeling_version}"
						)
	return ver_path


def get_analysis_dir_path(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str, modeling_version: int ) -> str:
	"""
	Return the path to the analysis_dict file for the given system.
	"""
	ver_path = get_sys_modeling_version_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name,
		modeling_version = modeling_version )
	analysis_dir_path = os.path.join( ver_path, "analysis" )
	return analysis_dir_path


def get_sys_config_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str, sys_conf_suff: str ) -> str:
	"""
	Return the path to the config dict for the given system.
	"""
	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )

	sys_config_file = os.path.join(
		data_dir,
		f"sys_config_{sys_name}{sys_conf_suff}.json" )
	return sys_config_file


def get_xl_file_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str, sys_conf_suff: str ) -> str:
	"""
	Return the path to the XLs .csv file for the given system.
	"""
	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )

	xl_file_path = os.path.join(
		data_dir,
		f"interprotein_xls{sys_conf_suff}.csv" )
	return xl_file_path


def get_benchmark_modeling_version_path(
	base_dir: str,
	benchmark_name: str,
	modeling_version: int ) -> str:
	"""
	Return the path to the benchmark results dir.
	"""
	benchmark_results_dir = get_benchmark_results_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		modeling_version = modeling_version )

	benchmark_modeling_version_path = os.path.join(
		benchmark_results_dir,
		f"version_{modeling_version}")

	return benchmark_modeling_version_path


def get_stat_file_path(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str, modeling_version: int ) -> str:
	"""
	Return the path to the stats file for the given system.
	For all systems, stats file is named Stats.npy.
	"""
	ver_path = get_sys_modeling_version_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name,
		modeling_version = modeling_version )
	stat_file_path = os.path.join( ver_path, "Stats.npy" )
	return stat_file_path


def get_analysis_dict_path(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str, modeling_version: int ) -> str:
	"""
	Return the path to the analysis_dict file for the given system.
	"""
	analysis_dir_path = get_analysis_dir_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name,
		modeling_version = modeling_version )
	analysis_dict_file = os.path.join( analysis_dir_path, "analysis_dict.npy" )
	return analysis_dict_file


def get_unrelaxed_model_file(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str,
	modeling_version: int, model_id: int ):
	"""
	Get the file to the unrelaxed model (saved during fine-tuning).
	"""
	ver_path = get_sys_modeling_version_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name,
		modeling_version = modeling_version )
	models_dir = os.path.join( 
	ver_path, f"{sys_name}_ensemble")
	model_file = os.path.join(
		models_dir, f"model_{model_id}.pdb" )
	return model_file


def get_relaxed_model_file(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str,
	modeling_version: int, model_id: int ):
	"""
	Get the file to the relaxed model (saved during fine-tuning).
	Only good-scoring models are relaxed and saved.
	"""
	analysis_dir_path = get_analysis_dir_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name,
		modeling_version = modeling_version )
	models_dir = os.path.join( 
	analysis_dir_path, f"relaxed_models" )
	model_file = os.path.join(
		models_dir, f"model_{model_id}.pdb" )
	return model_file


def get_unrelaxed_ensemble_file(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str,
	modeling_version: int,
	struct_format: str ):
	"""
	Get the file to the unrelaxed model (saved during fine-tuning).
	"""
	ver_path = get_sys_modeling_version_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name,
		modeling_version = modeling_version )
	ensemble_file = os.path.join(
		ver_path,
		f"{sys_name}_output_models.{struct_format}" )
	return ensemble_file


def get_relaxed_ensemble_file(
	base_dir: str,
	modeling_dir_name: str,
	sys_name: str,
	modeling_version: int,
	struct_format: str ):
	"""
	Get the file to the unrelaxed model (saved during fine-tuning).
	"""
	analysis_dir_path = get_analysis_dir_path(
		base_dir = base_dir,
		modeling_dir_name = modeling_dir_name,
		sys_name = sys_name,
		modeling_version = modeling_version )
	ensemble_file = os.path.join(
		analysis_dir_path,
		f"{sys_name}_relaxed_ensemble.{struct_format}" )
	return ensemble_file


def get_native_struct_file(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the native structure file for the given system.
	"""
	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name
	)
	native_struct_file = os.path.join( data_dir, f"{sys_name}.cif" )
	return native_struct_file


def get_init_struct_file(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the initial OpenFold predicted structure for the given system.
	"""
	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name
	)
	init_struct_file = os.path.join(
		data_dir,
		f"{sys_name}_output/predictions/{sys_name}_init_pred_relaxed.cif" )
	return init_struct_file
