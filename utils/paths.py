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
from typing import List, Tuple, Any
import os, glob

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
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

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
	csv_file_path: path to the benchmark .csv file.
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
	xl_type: str,
	frac_fp: str
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
		# f"interprotein_xls_{xl_type}.csv" )
		f"interprotein_xls_{xl_type}_{frac_fp}.csv" )
	return xl_file_path

################################################################################
################################################################################
def get_openfold_output_dir_path(
	base_dir: str,
	benchmark_name: str,
	sys_name: str
) -> str:
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
) -> str:
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
) -> str:
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
) -> str:
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
) -> str:
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
) -> str:
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
) -> str:
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
) -> str:
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
) -> str:
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
	config_name: str,
) -> str:
	"""
	Get the output dir path for the specified model:
		AlphaLInk2, GRASP, Boltz2

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	model: identifier for the model being used: alphalInk2/grasp/boltz2.
	config_name: identifier for the model config used for prediction.

	Returns:
	----------
	output_dir: path to the output dir for the given model's
		and xl_type.
	"""
	output_dir = os.path.join(
		base_dir,
		f"{model}/{benchmark_name}/{config_name}/"
	)
	return output_dir


def get_model_sys_output_dir_path(
	base_dir: str,
	benchmark_name: str,
	model: str,
	config_name: str,
	sys_name: str
) -> str:
	"""
	For the given system, get the path to the output dir
		for the given model and xl_type.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	model: identifier for the model being used: alphalInk2/grasp/boltz2
	config_name: identifier for the model config used for prediction.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	Returns:
	----------
	output_dir: path to the output dir for the given model's
		and xl_type.
	"""
	output_dir = get_model_output_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		model = model,
		config_name = config_name
	)
	sys_dir = os.path.join( output_dir, sys_name )
	return sys_dir

################################################################################
################################################################################
def return_model_sys_file(
	model: str,
	base_dir: str,
	benchmark_name: str,
	config_name: str,
	sys_name: str
) -> Tuple:
	"""
	Return the system-specific predicted output files for the given model.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	model: identifier for the model being used: alphalInk2/grasp/boltz2
	config_name: identifier for the model config used for prediction.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	Returns:
	----------
	Tuple of predicted output file paths for the specifed model and config.
	"""
	if model == "alphalink2":
		return get_alphalink2_sys_files(
			base_dir = base_dir,
			benchmark_name = benchmark_name,
			config_name = config_name,
			sys_name = sys_name
		)
	elif model == "grasp":
		return get_grasp_sys_files(
			base_dir = base_dir,
			benchmark_name = benchmark_name,
			config_name = config_name,
			sys_name = sys_name
		)
	elif model == "boltz2":
		return get_boltz2_sys_files(
			base_dir = base_dir,
			benchmark_name = benchmark_name,
			config_name = config_name,
			sys_name = sys_name
		)
	else:
		raise ValueError( f"incorrect model: {model} specified. " +
			"Supported: alphalink2/grasp/boltz2..."
		)


def get_alphalink2_sys_files(
	base_dir: str,
	benchmark_name: str,
	config_name: str,
	sys_name: str
) -> Tuple[List[str], List[str]]:
	"""
	For the given system, get the path to the AlphaLink2
		predicted structure file.
	AlphaLink2 5 trained models to predict 5 models each.
		We return a list of file paths for all predicted models.
	File name format: AlphaLink2_5182{MODEL CKPT}_{CONFIDENCE_SCORE}.pdb
	We ignore the file for the best predicted structure.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	config_name: identifier for the model config used for prediction.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	Returns:
	----------
	struct_file_list: list of paths to all AlphaLink2
		predicted structure files.
	output_file_list: list of paths to the AlphaLink2
		output .pkl.gz file.
	"""
	sys_dir = get_model_sys_output_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		model = "alphalink2",
		config_name = config_name,
		sys_name = sys_name
	)
	struct_file_list = []
	output_file_list = []
	for file in glob.glob( f"{sys_dir}/AlphaLink2_*.pdb" ):
		if "best.pdb" in file:
			continue
		else:
			struct_file_list.append( file )

	for file in glob.glob( f"{sys_dir}/AlphaLink2_*.pkl.gz" ):
		output_file_list.append( file )

	return struct_file_list, output_file_list


def get_grasp_sys_files(
	base_dir: str,
	benchmark_name: str,
	config_name: str,
	sys_name: str
) -> Tuple[List[str], List[str]]:
	"""
	For the given system, get the path to the GRASP
		predicted structure file.
	For all trained models, GRASP writes a structure file
		for each restraint filtering iteration.
	We select the final struture file.
	File name forat:
		unrelaxed_model_1_multimer_v3_v11_{MODEL_CKPT}_pred_{NUM_PRED}_final.pdb
	No output files are written on disk.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	config_name: identifier for the model config used for prediction.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	Returns:
	----------
	struct_file_list: list of paths to all AlphaLink2
		predicted structure and output files.
	output_file_list: empty list.
	"""
	sys_dir = get_model_sys_output_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		model = "grasp",
		config_name = config_name,
		sys_name = sys_name
	)
	struct_file_list = []
	output_file_list = []
	for file in glob.glob(
		f"{sys_dir}/unrelaxed_model_1_multimer_v3_v11_*_final.pdb" ):
		struct_file_list.append( file )

	return struct_file_list, output_file_list


def get_boltz2_sys_files(
	base_dir: str,
	benchmark_name: str,
	config_name: str,
	sys_name: str
) -> Tuple[List[str], List[str], List[str]]:
	"""
	For the given system, get the path to the Boltz2
		predicted structure file.
	Boltz2 predicts the specified no. of diffusion samples
		(structure) 0-indexed.
		Format: {SYS_NAME}_restraint_model_{DIFFUSION_SAMPLE}.cif
	It also provides:
		pTM, ipTm in a confidence .json file.
		Format: confidence_{SYS_NAME}_restraint_model_{DIFFUSION_SAMPLE}.json
		PAE in a .npy file
		Format: pae_{SYS_NAME}_restraint_model_{DIFFUSION_SAMPLE}.npz

	This function returns all three types of files.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.
	config_name: identifier for the model config used for prediction.
	sys_name: name of the complex modeled. For the benchmark,
		it's the PDB ID.

	Returns:
	----------
	struct_file_list: list of paths to all AlphaLink2
		predicted structure and output files.
	conf_file_list: list of paths to the confidence .json file.
	pae_file_list: list of paths to the .npy file containing
		the PAE/
	"""
	sys_dir = get_model_sys_output_dir_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		model = "boltz2",
		config_name = config_name,
		sys_name = sys_name
	)
	struct_file_list = []
	conf_file_list = []
	pae_file_list = []
	boltz_pred_sub_dir = f"boltz_results_{sys_name}_restraint/predictions/{sys_name}_restraint/"
	for file in glob.glob( f"{sys_dir}/{boltz_pred_sub_dir}*.cif" ):
		struct_file_list.append( file )
	for file in glob.glob( f"{sys_dir}/{boltz_pred_sub_dir}confidence_*.json" ):
		conf_file_list.append( file )
	for file in glob.glob( f"{sys_dir}/{boltz_pred_sub_dir}pae_*.npz" ):
		pae_file_list.append( file )

	return struct_file_list, conf_file_list, pae_file_list

################################################################################
################################################################################
def get_benchmark_analysis_dir_path(
	base_dir: str,
	benchmark_name: str,
):
	"""
	Get the path to the analysis dir for the given benchmark.

	Inputs:
	----------
	base_dir: dir to store all relevant modeling output.
	benchmark_name: name of the benchmark.

	Returns:
	----------
	analysis_dir: path to the analysis dir for the specified benchmark.
	"""
	analysis_dir = os.path.join(
		base_dir, f"{benchmark_name}_analysis/"
	)
	return analysis_dir


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

