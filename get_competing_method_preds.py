"""
Wrapper for obtaining GRASP and AlphaLink2 predictions for the benchmark.
Duplicating some function here from utils.utils, so as to not have more
	extra dependency while running GRASP and AlphaLink2.
The following dir structure is followed:
	BASE_DIR
		Model dir
			Dataset specific dir
				Prediction type (guided/unguided) specific dir
					XL type (short/long/fp) specific dir
						System specific dir
"""
from typing import List, Dict, Any
import os, subprocess, time, gzip, argparse, json, shutil, yaml
import pickle as pkl
import numpy as np
import pandas as pd

from config import db_dir, get_config_dict
from model_configs import BOLTZ, GRASP, ALPHALINK
from utils.mappings import (
	get_entity_chain_mapping,
	yield_restraints
)
from utils.utils import get_gpu_mem_mb
from utils.pdb_utils import get_chain_id
from utils.paths import (
	BASE_DIR,
	get_benchmark_csv_file,
	get_sys_data_dir_path,
	get_xl_file_path,
	get_model_output_dir_path
)


class CompetingMethodsRunner():
	"""
	A wrapper for running Boltz2, GRASP and AlphaLink2 on the benchmark.
	"""
	def __init__(
		self,
		model:str,
		benchmark_name: str,
		config_name: str,
		device: str
		):
		self.config_dict = get_config_dict()
		self.base_dir = os.path.join(
			os.path.abspath( self.config_dict.models.base_dir )
			)
		self.benchmark_name = benchmark_name
		if self.benchmark_name == "multistate":
			# Contains monomeric proteins.
			self.skip_intra_xls = False
		else:
			self.skip_intra_xls = True

		self.model = model # grasp/alphalink2
		self.config_name = config_name
		self.device = device

		self.model_config = {}
		# XL max bound (Ca-Ca).
		self.xl_max_bound = None

		self.inputs = {}


	def forward( self ):
		"""
		Initialize the XL max bound based on the XL type.
		Initialize the model configs.
		Create the required file paths and the
			dir structure.
		Initialize the logs dict.
		Create an input dict that contains the following
			for each system:
			Syste-specific dir path
			Path to the input fasta file.
			Path to the input restraints file.
			Path to the feature dict.
		Create the input files required for the specifie model.
		Run the model prediction.
		"""
		self.init_model_configs()
		self.set_xl_max_bound()

		# False discovery rate for AlphaLink2/GRASP.
		if self.model_config.no_fp_xls:
			if self.model == "grasp":
				# GRASP uses a default FDR of 0.05
				# 	For few XLs this can run into error.
				self.fdr = 0.05
			else:
				self.fdr = 0.0
		else:
			self.fdr = self.config_dict.benchmark.jwalk.frac_tp_fp[1]

		if self.model_config.pred_type == "guided":
			prefix = "Running restraint guided"
		elif self.model_config.pred_type == "unguided":
			prefix = "Running unguided"
			if self.model == "alphalink2":
				raise ValueError( "AlphaLink2 does not allow running unguided prediction..." )
		else:
			raise ValueError( "Incorrect pred_type: " +
				f"{self.model_config.pred_type} specified..."
			)
		print(
			f"{prefix} prediction for {self.model} " +
			f"with xl_type = {self.model_config['xl_type']}..."
			)
		print( "-"*80, "\n" )

		self.create_dir_structure()
		self.create_required_file()
		self.init_logs()
		self.load_benchmark()
		self.systems_to_model()

		self.create_system_specifc_inputs()

		self.create_inputs_for_benchmark()
		self.run_model_for_benchmark()


	def init_logs( self ):
		"""
		Initialize or load the logs dict.
		"""
		self.logs_file = os.path.join( self.output_dir, f"Logs_{self.model}.json" )
		if os.path.exists( self.logs_file ):
			with open( self.logs_file, "r" ) as f:
				self.logs = json.load( f )
		else:
			self.logs = {k:{} for k in ["completed", "time", "memory"]}


	def systems_to_model( self ):
		"""
		Initialize the systems (complexes) to be modeled.
		"""
		# if self.model_config.multi_state:
		# 	self.sys_to_model = ["1sc1", "8g0p", "8sjj"]
		# else:
		self.sys_to_model = self.benchmark["PDB ID"]


	def set_xl_max_bound( self ):
		"""
		Set the XL max bound according to the XL type used.
		"""
		if self.model_config.xl_type is None:
			self.xl_max_bound = 0.0
		elif self.model_config.xl_type == "short":
			self.xl_max_bound = self.config_dict.benchmark.jwalk.short_linker
		elif self.model_config.xl_type == "long":
			self.xl_max_bound = self.config_dict.benchmark.jwalk.long_linker
		elif self.model_config.xl_type in ["S1", "S2", "S1_2"]:
			self.xl_max_bound = self.config_dict.benchmark.jwalk.short_linker
		else:
			raise ValueError( f"Invalid XL type: " +
				f"{self.model_config['xl_type']} specified..."
			)


	def init_model_configs( self ):
		"""
		Initialize the model configs based pn the specified model.
		"""
		if self.model == "alphalink2":
			self.model_config = ALPHALINK
		elif self.model == "grasp":
			self.model_config = GRASP
		elif self.model == "boltz2":
			self.model_config = BOLTZ
		else:
			raise ValueError( f"Invalid model: {self.model} specified..." )

		if self.config_name not in self.model_config:
			raise ValueError( f"The specified model_config: {self.model_config} does not exist. " +
				"Check out model_configs.py for existing configs or define a new one..."
			)
		else:
			self.model_config = self.model_config[self.config_name]

	################################################################################
	################################################################################
	def create_required_file( self ):
		"""
		Create the required file and dir paths.
		"""
		# GRASP ----------
		# Path to the cloned GRASP directory.
		self.grasp_dir = "/home/kartik/Documents/IMP_Rewired/GRASP-JAX"
		# GRASP inference script.
		self.grasp_script = os.path.join( self.grasp_dir, "run_grasp.py" )

		# AlphaLink2 ----------
		# Path to the cloned GRASP directory.
		self.alphalink2_dir = f"/home/kartik/Documents/IMP_Rewired/AlphaLink2/"
		# AlphaLink2 inference script.
		self.alphalink2_script = os.path.join( self.alphalink2_dir, "run_alphalink.sh" )
		# AlphaLink2 parameters.
		self.alphalink2_params = os.path.join(
			self.alphalink2_dir,
			"params/AlphaLink-Multimer_SDA_v2.pt"
			)
		# Databases for running AlphaLink2.
		self.alphafold_dbs_dir = db_dir


	def load_benchmark( self ):
		"""
		Load the benchmark from disk.
		"""
		# Load the benchmark.
		benchmark_file = get_benchmark_csv_file(
			self.base_dir,
			self.benchmark_name,
			raw_file = False
			)
		self.benchmark = pd.read_csv( benchmark_file )

	################################################################################
	def create_dir_structure( self ):
		"""
		Dir to store GRASP/AlphaLink2/Boltz2 preds for the benchmark.
		The following dir structure is followed:
			BASE_DIR
				Model dir
					Dataset specific dir
						Config name

						# Prediction type (guided/unguided) specific dir
						# 	XL type (short/long/fp) specific dir
		System specific dir will be created later.
		"""
		self.output_dir = get_model_output_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			model = self.model,
			config_name = self.config_name
		)
		os.makedirs( self.output_dir, exist_ok = True )

	################################################################################
	################################################################################
	def create_system_specifc_inputs( self ):
		"""
		Create the following for all systems in the benchmark:
			System-specific dir to store the input
				files and predictions.
			Path for the FASTA file.
				Required for Alphaink2/GRASP only.
			path for the restraint file.
			Path to the precomputed feature dict.
				Required for GRASP only.
		"""
		# Dict to store input file paths for GRASP.
		self.inputs = {k:{} for k in [
			"sys_dir_path", "fasta_file", "restraints_file", "feat_dict_file"]}
		self.inputs["sys_dir_path"] = {}

		for sys_name in self.sys_to_model:
			data_dir = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name )

			sys_dir_path = os.path.join( self.output_dir, sys_name )

			self.inputs["sys_dir_path"][sys_name] = os.path.abspath( sys_dir_path )

			fasta_file = os.path.join( sys_dir_path, f"{sys_name}.fasta" )
			self.inputs["fasta_file"][sys_name] = os.path.abspath( fasta_file )

			# ext = "txt" if self.model == "grasp" else "csv"
			if self.model == "grasp":
				ext = "txt"
			elif self.model == "alphalink2":
				ext = "csv"
			else:
				# For Boltz-2
				ext = "yml"
			restraints_file = os.path.join( sys_dir_path, f"{sys_name}_restraint.{ext}" )
			self.inputs["restraints_file"][sys_name] = os.path.abspath( restraints_file )

			feat_dict_file = os.path.join( data_dir, f"{sys_name}_output", "predictions/feature_dict.pkl.gz" )
			self.inputs["feat_dict_file"][sys_name] = os.path.abspath( feat_dict_file )

			# Create the system dir for GRASP.
			os.makedirs( sys_dir_path, exist_ok = True )

	################################################################################
	################################################################################
	def create_fasta_file( self, sys_name: str ):
		"""
		Create a .fasta file containg the sequences for each chain to be modeled.
		Use the sequences from the sys_config dict.
		Fasta header -> {sys_name}_{entity_id}_{chain_id}
		This is required as input for AlphaLink2 and GRASP.
		"""
		fasta_file = self.inputs["fasta_file"][sys_name]
		w = open( fasta_file, "w" )

		entity_chain_map = get_entity_chain_mapping(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		for entity_id in entity_chain_map:
			seq = entity_chain_map[entity_id]["seq"]
			for chain_id in entity_chain_map[entity_id]["chains"]:
				if self.model == "alphalink2":
					chain_id = get_chain_id( chain_id - 1 )
				fasta_header = f"{sys_name}_{entity_id}_{chain_id}"
				w.writelines( f">{fasta_header}\n{seq}\n" )
		w.close()

	################################################################################
	################################################################################
	def create_restraints_file_grasp( self, sys_name: str ):
		"""
		Create a .txt file containing the Xl residues.
		Format:
			residue1, residue2, cutoff, FDR
				residue -> {CHAIN_ID}-{RES_POS}-{RES}
					CHAIN_ID -> numeric chain ID.
					RES_POS -> residue position.
					RES -> 1-letter aminao acid symbol.
		We assign every XL the same FDR.
		The XL file for our benchmark contain residues numbering
			based on the PDB file (seq_id).

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		"""
		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		# xl_file = os.path.join( data_dir, f"interprotein_xls{self.sys_conf_suff}.csv" )
		xl_type = "short" if self.model_config.xl_type == None else self.model_config.xl_type
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = xl_type,
			frac_fp = self.model_config.frac_fp
		)

		entity_chain_map = get_entity_chain_mapping(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		# GRASP expects Cb-Cb distances.
		# We convert Ca-Ca distances usingL Cb-Cb = max_bound - 2*Ca-Ca
		# 	Assuming Ca-Cb bond length is ~1.5 angstorm.
		restraints_included = []
		cb_max_bound = self.xl_max_bound - 2*1.5
		w = open( self.inputs["restraints_file"][sys_name], "w" )
		for row in yield_restraints(
			sys_name = sys_name,
			xl_file = xl_file,
			entity_chain_map = entity_chain_map,
			numeric_chain_ids = True,
			skip_intra_xls = self.skip_intra_xls
			):
			( entity_id1, entity_id2, chain_id1,
				chain_id2, res1, res2, label ) = row

			seq1 = entity_chain_map[entity_id1]["seq"]
			seq2 = entity_chain_map[entity_id2]["seq"]

			aa1 = seq1[res1-1]
			aa2 = seq1[res2-1]
			residue1 = f"{chain_id1}-{res1}-{aa1}"
			residue2 = f"{chain_id2}-{res2}-{aa2}"

			if self.model_config.no_fp_xls and label == 0:
				# Do not add FP XLs as restraints.
				print( "Skipping FP XLs..." )
			else:
				# ignore suplicate restraints.
				# 	GRASP is agnostic to A-B and B-A restraint.
				if f"{residue1},{residue2}" in restraints_included or f"{residue2},{residue1}" in restraints_included:
					continue
				restraints_included.append( f"{residue1},{residue2}" )

				w.writelines( f"{residue1}, {residue2}, {cb_max_bound}, {self.fdr}\n" )
		w.close()


	def create_inputs_for_grasp( self, sys_name: str ):
		"""
		To run GRASP with precomputed alignments, we need the following:
			FASTA file.
			Feature dict.
			Restraints file.
		To run unguided prediction, the restraint file must be
			specified as None.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		"""
		self.create_fasta_file( sys_name = sys_name )

		self.create_restraints_file_grasp( sys_name = sys_name )


	def run_grasp_per_system( self,
		sys_name: str, gpu_id: int
		) -> subprocess.Popen:
		"""
		Run GRASP prediction for the given system with the default settings.
		Here I assume that the feature_dict already exist.
		GRASP requires the feature dict as a .pkl file.
			We have stored the feature dict as a .pkl.gz file.
			Need to unzip and modify the path.
			Delete the .pkl file.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		gpu_id: iIndex of the GPU as recognized by `nvidia-smi` (after any
			CUDA_VISIBLE_DEVICES remapping).

		Returns:
		----------
		proc: Handle to the launched Boltz2 process.
			The caller is responsible for monitoring completion,
			capturing logs, and handling failures.
		"""
		feat_file = f"{self.inputs['feat_dict_file'][sys_name]}"
		pkl_feat_file = feat_file.removesuffix( ".gz" )
		with gzip.open( feat_file ) as f_in:
			with open( pkl_feat_file, "wb" ) as f_out:
				shutil.copyfileobj( f_in, f_out )

		if self.model_config.pred_type == "unguided":
			restraints_file = self.inputs['restraints_file'][sys_name]
		else:
			restraints_file = None
		env = os.environ.copy()
		# This remaps the device numbering.
		env["CUDA_VISIBLE_DEVICES"] = str( gpu_id )
		# So the device must be changed to cuda:0.
		device = "cuda:0"
		cmd = [
			"python", f"{self.grasp_script}",
			"--feature_pickle", f"{pkl_feat_file}",
			"--fasta_path", f"{self.inputs['fasta_file'][sys_name]}",
			"--data_dir", f"{self.grasp_dir}",
			"--output_dir", f"{self.inputs['sys_dir_path'][sys_name]}",
			"--restraints_file", f"{restraints_file}",
			"--iter_num", "5",
		]
		# subprocess.cal doe snot allow conrol over the process, so using Popen.
		proc = subprocess.Popen( cmd, env = env )
		return proc

	################################################################################
	################################################################################
	def reuse_msa_for_alphalink2( self,
		sys_name: str,
		entity_chain_map: Dict[int, Dict],
		data_dir: str
		):
		"""
		We reuse the precomputed MSAs for running AlphaLink2.
		Following files are needded for each chain:
			bfd_uniclust_hits.a3m
			mgnify_hits.sto
			pdb_hits.sto (hmm_search.sto in OpenFold)
			uniprot_hits.sto
			uniref90_hits.sto

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		entity_chain_map: dict containing a mapping between all
			the corresponding chains along with metadata, including
			entity sequence and residues positions.
		"""
		for entity_id in entity_chain_map:
			for chain_id in entity_chain_map[entity_id]["chains"]:
				chain_id = get_chain_id( chain_id - 1 )
				# AlphaLink2 alignment dir for the given chain.
				tgt_alignment_dir = os.path.join( self.inputs["sys_dir_path"][sys_name], f"{chain_id}" )
				os.makedirs( tgt_alignment_dir, exist_ok = True )
				# OpenFold slignment dir for the given chain.
				src_alignment_dir = os.path.join( data_dir, f"{sys_name}_output/" f"alignments/{sys_name}_{entity_id}_{chain_id}" )

				for file_name in ["mgnify_hits.sto", "uniprot_hits.sto", "uniref90_hits.sto"]:
					file_path = os.path.join( src_alignment_dir, file_name )
					cmd = ["cp", f"{file_path}", f"{tgt_alignment_dir}"]
					subprocess.call( cmd )

				src_file_path = os.path.join( src_alignment_dir, "bfd_unirefclust_hits.a3m" )
				tgt_file_path = os.path.join( tgt_alignment_dir, "bfd_uniclust_hits.a3m" )
				cmd = ["cp", f"{src_file_path}", f"{tgt_file_path}"]
				subprocess.call( cmd )

				src_file_path = os.path.join( src_alignment_dir, "hmm_output.sto" )
				tgt_file_path = os.path.join( tgt_alignment_dir, "pdb_hits.sto" )
				cmd = ["cp", f"{src_file_path}", f"{tgt_file_path}"]
				subprocess.call( cmd )


	def create_chain_mapping_for_alphalink2( self,
		sys_name: str,
		entity_chain_map: Dict[int, Dict]
		):
		"""
		Create a chains.txt file containing the chain IDs.
			e.g. A B
		Create a .json file that maps each chain ID to the
			corresponding FASTA header and sequence.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		entity_chain_map: dict containing a mapping between all
			the corresponding chains along with metadata, including
			entity sequence and residues positions.
		data_dir: path to the system data directory.
		"""
		chains_file = os.path.join( self.inputs["sys_dir_path"][sys_name], "chains.txt" )
		chain_mapping_file = os.path.join( self.inputs["sys_dir_path"][sys_name], "chain_id_map.json" )
		chains = []
		chain_mapping = {}
		for entity_id in entity_chain_map:
			seq = entity_chain_map[entity_id]["seq"]
			for chain_id in entity_chain_map[entity_id]["chains"]:
				# Get alphabetical chain ID.
				chain_id = get_chain_id( chain_id - 1 )
				chains.append( chain_id )
				fasta_header = f"{sys_name}_{entity_id}_{chain_id}"
				chain_mapping[chain_id] = {
					"descriptions": [fasta_header],
					"sequence": seq
				}

		with open( chains_file, "w" ) as w:
			w.writelines( " ".join( chains ) )
		with open( chain_mapping_file, "w" ) as w:
			json.dump( chain_mapping, w, indent = 4 )


	def split_feature_dict( self,
		sys_name: str,
		entity_chain_map: Dict[int, Dict] ):
		"""
		For using precomputed feature dicts, AlphaLink2
			expects a feature_dict per chain.
			e.g. A.feature_dict.pkl.gz
		Split the precomputed feature_dict by chain.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		entity_chain_map: dict containing a mapping between all
			the corresponding chains along with metadata, including
			entity sequence and residues positions.
		"""
		with gzip.open( self.inputs["feat_dict_file"][sys_name], "rb" ) as f:
			feature_dict = pkl.load( f )
		# f = open( self.inputs["feat_dict_file"][sys_name], "rb" )
		# feature_dict = pkl.load( f )
		# f.close()
		for entity_id in entity_chain_map:
			for chain_id in entity_chain_map[entity_id]["chains"]:
				# Get the alphabetical chain ID.
				chain = get_chain_id( chain_id - 1 )

				# asym_id is 1-indexed numeric chain ID.
				chain_mask = np.where( feature_dict["asym_id"] == chain_id )

				feat_chain = {}
				for k in feature_dict:
					if k in ["seq_length", ]:
						# The precomputed feature-dict contains seq_length for the system not individual chain.
						feat_chain[k] = len( chain_mask[0] )
					elif k in ["num_alignments", "num_templates", "assembly_num_chains"]:
						# These are just integer values.
						feat_chain[k] = feature_dict[k]
					elif "template" in k or "msa" in k or k in ["bert_mask", "deletion_matrix"]:
						# For these the tensor dhape is [S, N, ...]; where S is the no. of sequences.
						feat_chain[k] = feature_dict[k][:, chain_mask]
						# print( k, "  ", feature_dict[k].shape, "  ", feat_chain[k].shape )
					elif k == "cluster_bias_mask":
						# Singleton tensor.
						feat_chain[k] = feature_dict[k]
					else:
						feat_chain[k] = feature_dict[k][chain_mask]
				chain_feat_file = os.path.join(
					self.inputs["sys_dir_path"][sys_name], f"{chain}.feature.pkl.gz" )
				with gzip.open( chain_feat_file, "wb" ) as w:
					pkl.dump( feat_chain, w, protocol = pkl.HIGHEST_PROTOCOL )


	def create_alphalink2_restraints_file( self,
		sys_name: str,
		entity_chain_map: Dict[int, Dict] ):
		"""
		Create a .csv file containing the XL residues.
		Format:
			residue1,chain1,residue2,chain2
		We assign every XL the same FDR.
		Alphabetical chain IDs are needed.
		The XL file for our benchmark contain residues numbering
			based on the PDB file (seq_id).
		AlphaLink2 expects Ca-Ca crosslinks.
		When running unguided prediction, an empty restraint file is created.
		AlphaLink2 does not allow running unguided prediction.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		entity_chain_map: dict containing a mapping between all
			the corresponding chains along with metadata, including
			entity sequence and residues positions.
		"""
		# xl_file = os.path.join( data_dir, f"interprotein_xls{self.sys_conf_suff}.csv" )
		xl_type = "short" if self.model_config.xl_type == None else self.model_config.xl_type
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = xl_type,
			frac_fp = self.model_config.frac_fp
		)

		restraints_included = []
		w = open( self.inputs["restraints_file"][sys_name], "w" )
		if self.model_config.pred_type == "guided":
			for row in yield_restraints(
				sys_name = sys_name,
				xl_file = xl_file,
				entity_chain_map = entity_chain_map,
				numeric_chain_ids = False,
				skip_intra_xls = self.skip_intra_xls
				):
				( entity_id1, entity_id2, chain_id1,
					chain_id2, res1, res2, label ) = row

				if self.model_config.no_fp_xls and label == 0:
					# Do not add FP XLs as restraints.
					print( "Skipping FP XLs..." )
				else:
					# ignore duplicate restraints: AB and BA.
					restraint = f"{res1},{chain_id1},{res2},{chain_id2}"
					restraint_inv = f"{res2},{chain_id2},{res1},{chain_id1}"
					if restraint in restraints_included or restraint_inv in restraints_included:
						continue
					restraints_included.append( restraint )
					w.writelines( f"{res1},{chain_id1},{res2},{chain_id2},{self.fdr}\n" )
		if self.model_config.pred_type == "unguided":
			w.writelines( "" )

		w.close()


	def create_inputs_for_alphalink2( self, sys_name: str ):
		"""
		To run AlphaLink2 with precomputed alignments, we need the following:
			Per chain alignments.
		Stil, AlphaLink2 needs to recreate its own feature_dict.
		Restraints file.
		A .txt file containing the chain IDs.
		A .json file containing mapping betwen the
			chain IDs and the FASTA header and sequence.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		"""
		if os.path.exists( self.inputs["restraints_file"][sys_name] ):
			print( "Inputs already created..." )
		else:
			data_dir = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name )

			entity_chain_map = get_entity_chain_mapping(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name )

			self.create_fasta_file( sys_name = sys_name )

			self.reuse_msa_for_alphalink2(
				sys_name = sys_name,
				entity_chain_map = entity_chain_map,
				data_dir = data_dir
			)

			self.create_chain_mapping_for_alphalink2(
				sys_name = sys_name,
				entity_chain_map = entity_chain_map
			)

			self.create_alphalink2_restraints_file(
				sys_name = sys_name,
				entity_chain_map = entity_chain_map
				# data_dir = data_dir
			)


	def run_alphalink2_per_system( self,
		sys_name: str, gpu_id: int
		) -> subprocess.Popen:
		"""
		Run AlphaLink2 prediction for the given system with the default settings.
		Here I assume that the feature_dict already exist.
		We use the default setting specified for AlphaLink2.
		Added two arguments to run_alphalink2.sh to specify
			the device and the XL max bound.
		To run unguided prediction, the restraint file must be "".

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		gpu_id: iIndex of the GPU as recognized by `nvidia-smi` (after any
			CUDA_VISIBLE_DEVICES remapping).

		Returns:
		----------
		proc: Handle to the launched Boltz2 process.
			The caller is responsible for monitoring completion,
			capturing logs, and handling failures.
		"""
		env = os.environ.copy()
		# This remaps the device numbering.
		env["CUDA_VISIBLE_DEVICES"] = str( gpu_id )
		# So the device must be changed to cuda:0.
		device = "cuda:0"

		if self.model_config.pred_type == "guided":
			restraints_file = self.inputs['restraints_file'][sys_name]
		elif self.model_config.pred_type == "unguided":
			restraints_file = ""
		cmd = [
			"bash", f"{self.alphalink2_script}",
			f"{self.inputs['fasta_file'][sys_name]}",
			f"{restraints_file}",
			f"{self.inputs['sys_dir_path'][sys_name]}",  # output dir.
			f"{self.alphalink2_params}",  # path to the model params.
			f"{self.alphafold_dbs_dir}",
			f"{self.model_config.max_template_date}",
			f"{self.model_config.recycling_iters}",   # Max recycling iterations.
			f"{self.model_config.num_samples}",   # No. of samples to generate.
			f"{self.model_config.msa_neff}",   # MSA neff.
			f"{self.model_config.drop_xls}",   # Mask crosslinked residues in MSA or not.
			f"{device}",
			f"{self.xl_max_bound}"
		]
		# subprocess.call() doe snot allow conrol over the process, so using Popen.
		proc = subprocess.Popen( cmd, env = env )
		return proc

	################################################################################
	################################################################################
	def create_inputs_for_boltz2( self, sys_name: str ):
		"""
		Boltz2 accepts input in a .yaml file.
		Format can be found here: https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
		For prediction we need:
			protein seq and chain IDs.
			Pre-computed MSA path.
			XL restraint.
		To run unguided prediction, the yaml file must not have any
			constraints specified.

		Note:
		At this stage we assume that the MSA filesexists.
		For homomeric entities, a single MSA is reused for all chains.
		Boltz2 requires alphabetical chain IDs.
		XLs are modeled as contacts with a max-bound potential (contact_potential).
		For unguided prediction, no constraints are specified.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		"""
		xl_type = "short" if self.model_config.xl_type == None else self.model_config.xl_type
		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = xl_type,
			frac_fp = self.model_config.frac_fp
		)

		entity_chain_map = get_entity_chain_mapping(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		alignment_dir = os.path.join( data_dir, f"{sys_name}_output/" f"alignments/" )

		boltz_input = {"version": 1}
		boltz_input.update( {k:[] for k in ["sequences", 'constraints']} )

		for entity_id in entity_chain_map:
			chains = entity_chain_map[entity_id]["chains"]
			seq = entity_chain_map[entity_id]["seq"]
			# Convert numeric chain IDs to alphabets.
			chain_ids = [get_chain_id( idx-1 ) for idx in chains]
			# For homomers the MSA remains the same.
			msa_file = os.path.join( alignment_dir, f"{sys_name}_{entity_id}_{chain_ids[0]}/bfd_unirefclust_hits.a3m" )
			protein = {
				"protein": {
					"id": chain_ids,
					"sequence": seq,
					"msa": msa_file
				}
			}
			boltz_input["sequences"].append( protein )

		if self.model_config.pred_type == "guided":
			xl_file = get_xl_file_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name,
				xl_type = xl_type,
				frac_fp = self.model_config.frac_fp
			)

			# Will add all Xl restraints as contacts for conditioning Boltz-2.
			for row in yield_restraints(
				sys_name = sys_name,
				xl_file = xl_file,
				entity_chain_map = entity_chain_map,
				numeric_chain_ids = False,
				skip_intra_xls = self.skip_intra_xls
				):
				( entity_id1, entity_id2, chain_id1,
					chain_id2, res1, res2, label ) = row

				if self.model_config.no_fp_xls and label == 0:
					# Do not add FP XLs as restraints.
					print( "Skipping FP XLs..." )
				else:
					contact = {
						"contact": {
							"token1": [chain_id1, int( res1 )-1],
							"token2": [chain_id2, int( res2 )-1],
							"max_distance": self.xl_max_bound
						}
					}
					boltz_input["constraints"].append( contact )
		elif self.model_config.pred_type == "unguided":
			boltz_input.pop( "constraints" )

		# Save as a yaml file.
		with open( self.inputs["restraints_file"][sys_name], "w" ) as w:
			yaml.safe_dump( boltz_input, w, sort_keys = False )


	def run_boltz2_per_system( self, sys_name: str, gpu_id: int ) -> subprocess.Popen:
		"""
		Run Boltz2 prediction for the given system with the default settings.
		This is the easiet among the three to run.
		To run unguided prediction, the yaml file must not have any
			constraints specified.

		Inputs:
		----------
		sys_name: name of the complex modeled. For the benchmark,
			it's the PDB ID.
		gpu_id: iIndex of the GPU as recognized by `nvidia-smi` (after any
			CUDA_VISIBLE_DEVICES remapping).

		Returns:
		----------
		proc: Handle to the launched Boltz2 process.
			The caller is responsible for monitoring completion,
			capturing logs, and handling failures.
		"""
		env = os.environ.copy()
		# This remaps the device numbering.
		env["CUDA_VISIBLE_DEVICES"] = str( gpu_id )
		# So the device must be changed to cuda:0.
		device = "cuda:0"
		cmd = [
			"boltz",
			"predict",
			f"{self.inputs['restraints_file'][sys_name]}",
			"--out_dir", f"{self.inputs['sys_dir_path'][sys_name]}",  # output dir.
			"--num_subsampled_msa", f"{self.model_config.num_subsampled_msa}",
			"--recycling_steps", f"{self.model_config.recycling_steps}",
			"--sampling_steps", f"{self.model_config.sampling_steps}",
			"--diffusion_samples", f"{self.model_config.diffusion_samples}",
			"--max_parallel_samples", f"{self.model_config.max_parallel_samples}",
			"--step_scale", f"{self.model_config.step_scale}"
			# "--use_msa_server"  # We use pre-computed alignments.
		]
		if self.model_config.subsample_msa:
			cmd.append( "--subsample_msa" )
		# subprocess.call() doe snot allow conrol over the process, so using Popen.
		proc = subprocess.Popen( cmd, env = env )
		return proc

	################################################################################
	################################################################################
	def create_inputs_for_benchmark( self ):
		"""
		Create input files for running GRASP/AlphaLink2 on the benchmark.
		"""
		for sys_name in self.sys_to_model:
			# if sys_name in self.logs["completed"]:
			# 	print( f"Already completed for {sys_name}" )
			# 	continue
			# else:
			print( f"Creating inputs for {sys_name} to run {self.model}" )
			if self.model == "grasp":
				self.create_inputs_for_grasp( sys_name = sys_name )
			elif self.model == "alphalink2":
				self.create_inputs_for_alphalink2( sys_name = sys_name )
			elif self.model == "boltz2":
				self.create_inputs_for_boltz2( sys_name = sys_name )
			else:
				raise ValueError( f"Incorrect model: {self.model} specified. " +
					"Supported alphalink2/grasp/boltz2..."
				)


	def run_model_for_benchmark( self ):
		"""
		Run prediction on the benchmark for the specified modle.
		Keep track of the time taken and GPY memory used.
		If prediction already completed, do not run again.
		For GRASP, one needs to remove the .pkl feature dict file
			that was created for running GRASP.
		"""
		base = os.path.abspath( os.getcwd() )
		gpu_id = int( self.device.split( ":" )[1] )
		for idx, sys_name in enumerate( self.sys_to_model ):
			# if sys_name in self.logs["completed"] and sys_name not in ["6iww", "7agf"]:
			if sys_name in self.logs["completed"]:
				print( f"Already completed for {sys_name}" )
				continue
			else:
				ts = time.perf_counter()
				peak_mem = 0
				print( "\n" )
				print( "-"*80 + f"\nRunning {self.model} for: {idx}. {sys_name}\n" + "-"*80 )
				if self.model == "grasp":
					proc = self.run_grasp_per_system( sys_name = sys_name, gpu_id = gpu_id )
				elif self.model == "alphalink2":
					if sys_name == "5xct":
						continue
					# AlphaLik2 must be run from the AlphaLinki2 dir.
					os.chdir( self.alphalink2_dir )
					proc = self.run_alphalink2_per_system( sys_name = sys_name, gpu_id = gpu_id )
					os.chdir( base )
				else:
					proc = self.run_boltz2_per_system( sys_name = sys_name, gpu_id = gpu_id )

				# Keep track of memory usage.
				while proc.poll() is None:
					mem = get_gpu_mem_mb( gpu_id = gpu_id )
					peak_mem = max( peak_mem, mem )
					time.sleep( 0.1 )

				te = time.perf_counter()

				if proc.returncode != 0:
					print( f"Failed to run {self.model} for {sys_name}..." )
					exit()
				# Remove the feature-dict .pkl file created for GRASP.
				if self.model == "grasp":
					feat_file = f"{self.inputs['feat_dict_file'][sys_name]}"
					pkl_feat_file = feat_file.removesuffix( ".gz" )
					os.remove( pkl_feat_file )

				time_taken = te-ts
				self.logs["completed"][sys_name] = None
				self.logs["time"][sys_name] = time_taken
				self.logs["memory"][sys_name] = peak_mem

				with open( self.logs_file, "w" ) as w:
					json.dump( self.logs, w, indent = 4 )

			print( f"Time taken = {self.logs['time'][sys_name]/60} minutes" )
			print( f"Peak memory = {self.logs['memory'][sys_name]/1024} GB" )


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description = "Obtain predictions from GRASP, AlphaLink2 for the benchmark."
	)
	parser.add_argument(
		"-m", "--model",
		type = str, required = True,
		help = "Specify the model to use: grasp/alphalink2/boltz2." )
	parser.add_argument(
		"-c", "--config_name",
		type = str, required = True,
		help = "Specify the model config to be used. See model_configs.py." )
	parser.add_argument(
		"-b", "--benchmark",
		type = str, required = True,
		help = "Specify the benchmark: crosslink/multistate." )
	parser.add_argument(
		"-d", "--device",
		type = str, required = True,
		help = "device to be used (cpu/cuda:0/cuda:1)..." )
	args = parser.parse_args()

	CompetingMethodsRunner(
		model = args.model,
		benchmark_name = args.benchmark,
		config_name = args.config_name,
		device = args.device
		).forward()
