"""
Funstions to create paths for the system or benchmark directories and files.
Each input complex being modeled is identified by the sys_name.
This could be the PDB ID for all complexes in the benchmark.
The following dir structure is followed in this framework:
BASE_DIR/
	*_benchmark/
		For each input system, contains a dir containing the initial OpenFold
			prediction, native struct file, data file (.csv file for XLs).
	*_metadata/
		Contains the JWALK predicted XLs file, configs for creating the benchmark,
			metrics for the initial OpenFold prediction, native structures for all
			systems in the benchmark.
	# *_benchmark_analysis/
	# 	For each benchmark run, contains a dir containing the analysis results
	# 		and configs for the run.
	# *_modeling/
	# 	For each system, contains a dir per simulation (modeling_version)
	# 		containing all outputs generated during the simulation.
"""
from typing import Any
import os

# Base directory for benchmark and the aasociated content.
BASE_DIR = "./benchmark/"


def get_meta_dir_path(
	base_dir: str,
	benchmark_name: str ) -> str:
	"""
	Return the path to the metadat dir for the benchmark.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.

	Return:
	----------
	meta_dir_path: path to the metadata dir.
	"""
	meta_dir = os.path.join(
		base_dir, f"{benchmark_name}_metadata/" )
	return meta_dir


def get_sys_data_dir_path(
	base_dir: str,
	benchmark_name: str, sys_name: str ) -> str:
	"""
	Return the path to the system data directory.
		data_dir -> BASE_DIR/*_benchmark/sys_name/

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.

	Return:
	----------
	data_dir: path to the system specific data dir.
	"""
	data_dir = os.path.join(
		base_dir,
		f"{benchmark_name}_benchmark/{sys_name}/" )
	return data_dir


def get_benchmark_csv_file(
	base_dir: str,
	benchmark_name: str,
	raw_file: bool = False ) -> str:
	"""
	Return the path to the selected benchmark .csv file.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.

	Return:
	----------
	stat_file_path: path to the benchmark .csv file.
	"""
	if raw_file:
		file_name = f"{benchmark_name}_benchmark.csv"
	else:
		file_name = f"selected_{benchmark_name}_benchmark.csv"
	meta_dir = get_meta_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name )
	csv_file_path = os.path.join(
		meta_dir, file_name
	)
	return csv_file_path


def get_sys_config_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str ) -> str: #, sys_conf_suff: str ) -> str:
	"""
	Return the path to the config dict for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.

	Return:
	----------
	sys_config_file: path to the system-specific config file.
	"""
	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )

	sys_config_file = os.path.join(
		data_dir,
		f"sys_config_{sys_name}.json" )
		# f"sys_config_{sys_name}{sys_conf_suff}.json" )
	return sys_config_file


def get_xl_file_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str,
	xl_type: str
	) -> str:
	"""
	Return the path to the XLs .csv file for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
	xl_type: suffix specifying the XL type - short/long/fp.

	Return:
	----------
	xl_file_path: path to the XLs .csv file.
	"""
	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )

	xl_file_path = os.path.join(
		data_dir,
		f"interprotein_xls_{xl_type}.csv" )
	return xl_file_path

################################################################################
################################################################################
def get_openfold_output_dir_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the OpenFold output dir for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.

	Returns:
	----------
	ofold_output_dir: path to the OpenFold output dir.
	"""
	sys_data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name
	)
	ofold_output_dir = os.path.join(
		sys_data_dir,
		f"{sys_name}_output/" )
	return ofold_output_dir


def get_openfold_alignments_dir_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the OpenFold alignments dir
		for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.

	Returns:
	----------
	ofold_alignments_dir: path to the OpenFold alignments dir.
	"""
	ofold_output_dir = get_openfold_output_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	ofold_alignments_dir = os.path.join(
		ofold_output_dir,
		"alignments" )
	return ofold_alignments_dir


def get_openfold_pred_dir_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the OpenFold predictions dir
		for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.

	Returns:
	----------
	ofold_preds_dir: path to the OpenFold prediction dir.
	"""
	ofold_output_dir = get_openfold_output_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	ofold_preds_dir = os.path.join(
		ofold_output_dir,
		"predictions" )
	return ofold_preds_dir

################################################################################
################################################################################
def get_feature_dict_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the OpenFold generated feature dict
		for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.

	Returns:
	----------
	feature_dict_path: path to the OpenFold feature dict.
	"""
	ofold_preds_dir = get_openfold_pred_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	feature_dict_path = os.path.join(
		ofold_preds_dir,
		"feature_dict.pkl.gz" )
	return feature_dict_path


def get_processed_feature_dict_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the OpenFold generated processed
		feature dict for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.

	Returns:
	----------
	processed_feature_dict_path: path to the OpenFold
		processed feature dict.
	"""
	ofold_preds_dir = get_openfold_pred_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	processed_feature_dict_path = os.path.join(
		ofold_preds_dir,
		"processed_feature_dict.pkl.gz" )
	return processed_feature_dict_path

################################################################################
################################################################################
def get_sys_fasta_file_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the fasta file for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.

	Returns:
	----------
	fasta_file: fasta file path for the given system.
	"""
	data_dir = get_sys_data_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name
	)
	fasta_file = os.path.join( data_dir, f"{sys_name}.fasta" )
	return fasta_file


def get_native_struct_file(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the native structure file for the given system.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.
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

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.
	"""
	ofold_preds_dir = get_openfold_pred_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	init_struct_file = os.path.join(
		ofold_preds_dir,
		f"{sys_name}_init_pred_relaxed.cif" )
	return init_struct_file


def get_init_pred_file(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
):
	"""
	Return the path to the output dict for the initial
		OpenFold prediction for the given system stored
		as a .pkl file.

	Input:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	sys_name: we use the PDB ID as a unique identifier for a complex.
		Aka entry_id.
	"""
	ofold_preds_dir = get_openfold_pred_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	init_pred_file = os.path.join(
		ofold_preds_dir,
		f"output_dict.pkl" )
	return init_pred_file

################################################################################
################################################################################
def get_model_output_dir_path(
	base_dir: str,
	benchmark_name: str,
	model: str,
	pred_type: str,
	xl_type: str
):
	"""
	Get the output dir path for the specified model:
		AlphaLInk2, GRASP, Boltz2

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	model: identifier for the model being used: alphalInk2/grasp/boltz2
	pred_type: identifier for the type of prediction: guided/unguided.
		"" for guided prediction else "unguided".
	xl_type: identifier for tthe XL type. Could be short/long/fp.
	"""
	output_dir = os.path.join(
		base_dir,
		f"{model}/{benchmark_name}/{pred_type}/{xl_type}/"
	)
	return output_dir

#
# def get_sys_modeling_path(
# 	base_dir: str,
# 	modeling_dir_name: str, sys_name: str ) -> str:
# 	"""
# 	Return the path to the modeling dir for given system.

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	modeling_dir_name: name of the modeling dir.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.

# 	Return:
# 	----------
# 	sys_modeling_path: path to the system-specific modeling dir.
# 	"""
# 	sys_modeling_path = os.path.join( 
# 				os.path.abspath(
# 					f"{base_dir}/{modeling_dir_name}/{sys_name}"
# 					)
# 		)
# 	return sys_modeling_path


# def get_benchmark_analysis_dir_path(
# 	base_dir: str,
# 	benchmark_name: str ) -> str:
# 	"""
# 	Return the path to the benchmark analysis dir.
# 		BASE_DIR/*_benchmark_analysis/

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	benchmark_name: name of the benchmark.

# 	Return:
# 	----------
# 	benchmark_analysis_dir: path to the analysis dir for the benchmark.
# 	"""
# 	benchmark_analysis_dir = os.path.join(
# 		base_dir,
# 		f"{benchmark_name}_benchmark_analysis" )
# 	return benchmark_analysis_dir

# def get_benchmark_analysis_version_dir_path(
# 	base_dir: str,
# 	benchmark_name: str,
# 	modeling_version: Any ) -> str:
# 	"""
# 	Return the path to the benchmark analysis dir for
# 		the given modleing version.
# 		BASE_DIR/*_benchmark_analysis/version_{VERSION}/

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	benchmark_name: name of the benchmark.

# 	Return:
# 	----------
# 	benchmark_analysis_dir: path to the analysis dir for the benchmark.
# 	"""
# 	benchmark_analysis_ver_dir = os.path.join(
# 		base_dir,
# 		f"{benchmark_name}_benchmark_analysis/version_{modeling_version}" )
# 	return benchmark_analysis_ver_dir


# def get_sys_modeling_version_path(
# 	base_dir: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any, sys_name: str ) -> str:
# 	"""
# 	Return the path to the system modeling version dir.

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	benchmark_name: name of the benchmark.
# 	modeling_version: an identifier for the simulation version.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.

# 	Return:
# 	----------
# 	ver_path: path to the system-specific modleing version dir.
# 	"""
# 	sys_path = get_sys_modeling_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name )
# 	ver_path = os.path.join( sys_path,
# 						f"version_{modeling_version}"
# 						)
# 	return ver_path

# def get_analysis_dir_path(
# 	base_dir: str,
# 	sys_name: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any ) -> str:
# 	"""
# 	Return the path to the analysis_dict file for the given system.

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
# 	modeling_dir_name: name of the modeling dir.
# 	modeling_version: an identifier for the simulation version.

# 	Return:
# 	----------
# 	analysis_dir_path: path to the system-specific analysis dir for
# 		the given modleing version.
# 	"""
# 	ver_path = get_sys_modeling_version_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name,
# 		modeling_version = modeling_version )
# 	analysis_dir_path = os.path.join( ver_path, "analysis" )
# 	return analysis_dir_path

# def get_benchmark_analysis_version_path(
# 	base_dir: str,
# 	benchmark_name: str,
# 	modeling_version: Any ) -> str:
# 	"""
# 	Return the path to the benchmark analysis dir for the given modeling version.

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	benchmark_name: name of the benchmark.
# 	modeling_version: an identifier for the simulation version.

# 	Return:
# 	----------
# 	benchmark_modeling_version_path: path to the analysis dir for the
# 		benchmark for the given modeling version.
# 	"""
# 	benchmark_analysis_dir = get_benchmark_analysis_dir_path(
# 		base_dir = base_dir,
# 		benchmark_name = benchmark_name,
# 		modeling_version = modeling_version )

# 	benchmark_modeling_version_path = os.path.join(
# 		benchmark_analysis_dir,
# 		f"version_{modeling_version}")

# 	return benchmark_modeling_version_path


# def get_stat_file_path(
# 	base_dir: str,
# 	sys_name: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any ) -> str:
# 	"""
# 	Return the path to the stats file for the given system.
# 	For all systems, stats file is named Stats.npy.

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
# 	modeling_dir_name: name of the modeling dir.
# 	modeling_version: an identifier for the simulation version.

# 	Return:
# 	----------
# 	stat_file_path: path to the system-specific stats file for the
# 		given modeling version.
# 	"""
# 	ver_path = get_sys_modeling_version_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name,
# 		modeling_version = modeling_version )
# 	stat_file_path = os.path.join( ver_path, "Stats.npy" )
# 	return stat_file_path


# def get_analysis_dict_path(
# 	base_dir: str,
# 	sys_name: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any ) -> str:
# 	"""
# 	Return the path to the analysis_dict file for the given system.

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
# 	modeling_dir_name: name of the modeling dir.
# 	modeling_version: an identifier for the simulation version.

# 	Return:
# 	----------
# 	analysis_dict_file -> path to the system-specific analysis file for the
# 		given modeling version.
# 	"""
# 	analysis_dir_path = get_analysis_dir_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name,
# 		modeling_version = modeling_version )
# 	analysis_dict_file = os.path.join( analysis_dir_path, "analysis_dict.npy" )
# 	return analysis_dict_file


# def get_unrelaxed_model_file(
# 	base_dir: str,
# 	sys_name: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any,
# 	model_id: int,
# 	struct_format: str = "pdb" ):
# 	"""
# 	Get the file to the unrelaxed model (saved during fine-tuning).

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
# 	modeling_dir_name: name of the modeling dir.
# 	modeling_version: an identifier for the simulation version.
# 	model_id: an identifier for the model (predicted structure).
# 	struct_format: file format for the structure file (pdb/cif).
# 	"""
# 	ver_path = get_sys_modeling_version_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name,
# 		modeling_version = modeling_version )
# 	models_dir = os.path.join( 
# 	ver_path, f"{sys_name}_ensemble")
# 	model_file = os.path.join(
# 		models_dir, f"model_{model_id}.{struct_format}" )
# 	return model_file


# def get_relaxed_model_file(
# 	base_dir: str,
# 	sys_name: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any,
# 	model_id: int,
# 	struct_format: str = "pdb" ):
# 	"""
# 	Get the file to the relaxed model (saved during fine-tuning).
# 		Only good-scoring models are relaxed and saved.

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
# 	modeling_dir_name: name of the modeling dir.
# 	modeling_version: an identifier for the simulation version.
# 	model_id: an identifier for the model (predicted structure).
# 	struct_format: file format for the structure file (pdb/cif).
# 	"""
# 	analysis_dir_path = get_analysis_dir_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name,
# 		modeling_version = modeling_version )
# 	models_dir = os.path.join( 
# 	analysis_dir_path, f"relaxed_models" )
# 	model_file = os.path.join(
# 		models_dir, f"model_{model_id}.{struct_format}" )
# 	return model_file


# def get_unrelaxed_ensemble_file(
# 	base_dir: str,
# 	sys_name: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any,
# 	struct_format: str ):
# 	"""
# 	Get the file to the unrelaxed model (saved during fine-tuning).

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
# 	modeling_dir_name: name of the modeling dir.
# 	modeling_version: an identifier for the simulation version.
# 	struct_format: file format for the structure file (pdb/cif).
# 	"""
# 	ver_path = get_sys_modeling_version_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name,
# 		modeling_version = modeling_version )
# 	ensemble_file = os.path.join(
# 		ver_path,
# 		f"{sys_name}_output_models.{struct_format}" )
# 	return ensemble_file


# def get_relaxed_ensemble_file(
# 	base_dir: str,
# 	sys_name: str,
# 	modeling_dir_name: str,
# 	modeling_version: Any,
# 	struct_format: str ):
# 	"""
# 	Get the file to the unrelaxed model (saved during fine-tuning).

# 	Input:
# 	----------
# 	base_dir: dir to store all relevant modeling output.
# 	sys_name: name of the complex modeled. For the benchmark, it's the PDB ID.
# 	modeling_dir_name: name of the modeling dir.
# 	modeling_version: an identifier for the simulation version.
# 	struct_format: file format for the structure file (pdb/cif).
# 	"""
# 	analysis_dir_path = get_analysis_dir_path(
# 		base_dir = base_dir,
# 		modeling_dir_name = modeling_dir_name,
# 		sys_name = sys_name,
# 		modeling_version = modeling_version )
# 	ensemble_file = os.path.join(
# 		analysis_dir_path,
# 		f"{sys_name}_relaxed_ensemble.{struct_format}" )
# 	return ensemble_file

