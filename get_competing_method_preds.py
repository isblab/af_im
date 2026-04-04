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

from config import get_config_dict
from utils.utils import read_pkl_gz
from utils.paths import (
	BASE_DIR,
	get_benchmark_csv_file,
	get_sys_config_path,
	get_sys_data_dir_path,
	get_xl_file_path,
	get_model_output_dir_path
)


def get_gpu_mem_mb( gpu_id: int ):
	"""
	Obtain the GPU memory used for the given device using nvidia-smi.
	"""
	out = subprocess.check_output(
		[
			"nvidia-smi",
			f"--id={gpu_id}",
			"--query-gpu=memory.used",
			"--format=csv,noheader,nounits"
		],
		encoding = "utf-8"
	)
	return int( out.strip() )


def get_chain_id( idx: int ) -> str:
	"""
	Get a chain ID based on an index.
	"""
	alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

	if idx < len( alphabet ):
		chain_id = alphabet[idx]

	else:
		raise ValueError( "Too many chains..." )
	return chain_id


def get_entities_in_system(
	base_dir: str,
	benchmark_name: str,
	sys_name: str ) -> List[Dict[str, Any]]:
	"""
	Parse the sys_config file and return the List of entities in the system.
	"""
	sys_config_path = get_sys_config_path(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	with open( sys_config_path, "r" ) as f:
		sys_conf = json.load( f )
	entities = sys_conf["entity"]
	return entities


def get_entity_chain_mapping(
	base_dir: str,
	benchmark_name: str,
	sys_name: str ) -> Dict[int, Dict]:
	"""
	Map all entities to the corresponding chains.
	For entity it contains:
		sequence to be modeled.
		a numeric chain_id starting from 1.
		residues to be modeled.
	"""
	entities = get_entities_in_system(
		base_dir = base_dir,
		benchmark_name = benchmark_name,
		sys_name = sys_name )
	# GRASP expects numeric chain IDs.
	chain_id = 1
	entity_chain_map = {}
	for entity_id, entity in enumerate( entities, start = 1 ):
		entity_chain_map[entity_id] = {"seq": "", "chains": [], "residues": []}
		for cp in range( entity["copy_num"] ):
			entity_chain_map[entity_id]["seq"] = seq = entity["sequence"]
			entity_chain_map[entity_id]["chains"].append( chain_id )
			entity_chain_map[entity_id]["residues"] = np.arange( entity["start"], entity["end"] + 1 )
			chain_id += 1
	return entity_chain_map


class CompetingMethodsRunner():
	"""
	A wrapper for running GRASP and AlphaLink2 on the benchmark.
	"""
	def __init__(
		self,
		model:str,
		xl_type: str,
		device: str
		):
		self.config_dict = get_config_dict()
		self.base_dir = os.path.join(
			os.path.abspath( self.config_dict.models.base_dir )
			)
		self.benchmark_name = self.config_dict.benchmark.globals.benchmark_name

		self.model = model # grasp/alphalink2
		self.xl_type = xl_type
		self.device = device

		self.model_config = {}
		# XL max bound (Ca-Ca).
		self.xl_max_bound = None
		# False discovery rate for GRASP.
		self.fdr = 0.1

		self.inputs = {}


	def forward( self ):
		"""
		Start by creating the required file paths
			and the dir structure.
		Initialize the logs dict.
		Initialize the XL amx bound based on the XL type.
		Initialize the model configs.
		Create an input dict that contains the following
			for each system:
			Syste-specific dir path
			Path to the input fasta file.
			Path to the input restraints file.
			Path to the feature dict.
		Create the input files required for the specifie model.
		Run the model prediction.
		"""
		self.set_xl_max_bound()
		self.init_model_configs()

		if self.model_config.guided_pred:
			print( f"Running restraint guided prediction for {self.model} with xl_type = {self.xl_type}..." )
			print( "-"*80, "\n" )
		else:
			print( f"Running unguided prediction for {self.model} with xl_type = {self.xl_type}..." )
			print( "-"*80, "\n" )

		self.create_dir_structure()
		self.create_required_file()
		self.init_logs()

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


	def set_xl_max_bound( self ):
		"""
		Set the XL amx bound according to the XL type used.
		"""
		if self.xl_type == "short":
			self.xl_max_bound = self.config_dict.benchmark.jwalk.short_linker
		elif self.xl_type == "long":
			self.xl_max_bound = self.config_dict.benchmark.jwalk.long_linker
		else:
			raise ValueError( f"Invalid XL type: {self.xl_type} specified..." )


	def init_model_configs( self ):
		"""
		Initialize the model configs based pn the specified model.
		"""
		if self.model == "alphalink2":
			self.model_config = self.config_dict.models.alphalink2
		elif self.model == "grasp":
			self.model_config = self.config_dict.models.grasp
		elif self.model == "boltz2":
			self.model_config = self.config_dict.models.boltz2
		else:
			raise ValueError( f"Invalid model: {self.model} specified..." )

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
		self.alphalink2_params = os.path.join( self.alphalink2_dir, "params/AlphaLink-Multimer_SDA_v2.pt" )
		# Databases for running AlphaLink2.
		self.alphafold_dbs_dir = "/data/alpha-fold-db/"

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
						Prediction type (guided/unguided) specific dir
							XL type (short/long/fp) specific dir
		System specific dir will be created later.
		"""
		pred_type = "" if self.model_config.guided_pred else "unguided"

		self.output_dir = get_model_output_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			model = self.model,
			pred_type = pred_type,
			xl_type = self.xl_type
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

		for sys_name in self.benchmark["PDB ID"]:
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
	def yield_restraints( self,
		xl_file: str,
		entity_chain_map: Dict[int, Dict],
		numeric_chain_ids: bool ):
		"""
		A generator that yields restrained residue pairs.
		Accounts for ambiguity.
		"""
		xl_df = pd.read_csv( xl_file )

		for row in xl_df.iterrows():
			p1, p2 = row[1]["prot1"], row[1]["prot2"]
			r1, r2, label = row[1]["res1"], row[1]["res2"], row[1]["label"]
			r1, r2 = int( r1 ), int( r2 )

			entity_id1 = int( p1.split( "_" )[1] )
			entity_id2 = int( p2.split( "_" )[1] )

			# Get the residue indices.
			try:
				r1_idx = np.where( entity_chain_map[entity_id1]["residues"] == r1 )[0][0]
			except:
				continue

			try:
				r2_idx = np.where( entity_chain_map[entity_id2]["residues"] == r2 )[0][0]
			except:
				continue

			# For ambiguous XLs, we consider all combinations.
			for chain_id1 in entity_chain_map[entity_id1]["chains"]:
				if not numeric_chain_ids:
					chain_id1 = get_chain_id( chain_id1 - 1  ) # 0-indexed.
				for chain_id2 in entity_chain_map[entity_id2]["chains"]:
					if not numeric_chain_ids:
						chain_id2 = get_chain_id( chain_id2 - 1  ) # 0-indexed.

					yield entity_id1, entity_id2, chain_id1, chain_id2, r1_idx, r2_idx

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
			For GRASP we need to map this to the numbering based on the
				input sequence, essentially the residue index + 1.
		"""
		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		# xl_file = os.path.join( data_dir, f"interprotein_xls{self.sys_conf_suff}.csv" )
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = self.xl_type
		)
		# xl_df = pd.read_csv( xl_file )

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
		for row in self.yield_restraints( xl_file, entity_chain_map, numeric_chain_ids = True ):
			( entity_id1, entity_id2, chain_id1,
				chain_id2, r1_idx, r2_idx ) = row

			seq1 = entity_chain_map[entity_id1]["seq"]
			seq2 = entity_chain_map[entity_id2]["seq"]

			# For ambiguous XLs, we consider all combinations.
			if seq1[r1_idx] != "K":
				raise ValueError( f"{sys_name}: Entity: {entity_id1}; " +
					f"Chain: {chain_id1}; residue {r1_idx+1} is not a Lys..." )
			residue1 = f"{chain_id1}-{r1_idx+1}-{seq1[r1_idx]}"

			if seq2[r2_idx] != "K":
				raise ValueError( f"{sys_name}: Entity: {entity_id2}; " +
					f"Chain: {chain_id2}; residue {r2_idx+1} is not a Lys..." )
			residue2 = f"{chain_id2}-{r2_idx+1}-{seq2[r2_idx]}"

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
		"""
		self.create_fasta_file( sys_name = sys_name )

		self.create_restraints_file_grasp( sys_name = sys_name )


	def run_grasp_per_system( self, sys_name: str, gpu_id: int ):
		"""
		Run GRASP prediction for the given system with the default settings.
		Here I assume that the feature_dict already exist.
		GRASP requires the feature dict as a .pkl file.
			We have stored the feature dict as a .pkl.gz file.
			Need to unzip and modify the path.
			Delete the .pkl file.
		"""
		feat_file = f"{self.inputs['feat_dict_file'][sys_name]}"
		pkl_feat_file = feat_file.removesuffix( ".gz" )
		with gzip.open( feat_file ) as f_in:
			with open( pkl_feat_file, "wb" ) as f_out:
				shutil.copyfileobj( f_in, f_out )

		if self.model_config.guided_pred:
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
		data_dir: str,
		sys_name: str,
		entity_chain_map: Dict[int, Dict]
		):
		"""
		We reuse the precomputed MSAs for running AlphaLink2.
		Following files are needded for each chain:
			bfd_uniclust_hits.a3m
			mgnify_hits.sto
			pdb_hits.sto (hmm_search.sto in OpenFold)
			uniprot_hits.sto
			uniref90_hits.sto
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


	def create_chain_mapping_for_alphalink2(
		self,
		sys_name: str,
		entity_chain_map: Dict[int, Dict]
		):
		"""
		Create a chains.txt file containing the chain IDs.
			e.g. A B
		Create a .json file that maps each chain ID to the
			corresponding FASTA header and sequence.
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
		"""
		# f = open( self.inputs["feat_dict_file"][sys_name], "rb" )
		# feature_dict = pkl.load( f )
		# f.close()
		feature_dict = read_pkl_gz( self.inputs["feat_dict_file"][sys_name] )
		for entity_id in entity_chain_map:
			for chain_id in entity_chain_map[entity_id]["chains"]:
				# Get the alphabetical chain ID.
				chain = get_chain_id( chain_id - 1 )

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
		entity_chain_map: Dict[int, Dict],
		data_dir: str  ):
		"""
		Create a .csv file containing the Xl residues.
		Format:
			residue1,chain1,residue2,chain2
		We assign every XL the same FDR.
		Alphabetical chain IDs are needed.
		The XL file for our benchmark contain residues numbering
			based on the PDB file (seq_id).
			For AlphaLink2 we need to map this to the numbering based on the
				input sequence, essentially the residue index + 1.
		AlphaLink2 expects Ca-Ca crosslinks.
		To run unguided prediction, an empty restraint file must be used.
		"""
		# xl_file = os.path.join( data_dir, f"interprotein_xls{self.sys_conf_suff}.csv" )
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = self.xl_type
		)

		restraints_included = []
		w = open( self.inputs["restraints_file"][sys_name], "w" )
		if self.model_config.guided_pred:
			for row in self.yield_restraints( xl_file, entity_chain_map, numeric_chain_ids = False ):
				( entity_id1, entity_id2, chain_id1,
					chain_id2, r1_idx, r2_idx ) = row

				# ignore duplicate restraints: AB and BA.
				restraint = f"{r1_idx+1},{chain_id1},{r2_idx+1},{chain_id2}"
				restraint_inv = f"{r2_idx+1},{chain_id2},{r1_idx+1},{chain_id1}"
				if restraint in restraints_included or restraint_inv in restraints_included:
					continue
				restraints_included.append( restraint )
				w.writelines( f"{r1_idx+1},{chain_id1},{r2_idx+1},{chain_id2},{self.fdr}\n" )
		else:
			w.writelines( ",,,," )

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
				data_dir = data_dir,
				sys_name = sys_name,
				entity_chain_map = entity_chain_map
			)

			self.create_chain_mapping_for_alphalink2(
				sys_name = sys_name,
				entity_chain_map = entity_chain_map
			)

			self.create_alphalink2_restraints_file(
				sys_name = sys_name,
				entity_chain_map = entity_chain_map,
				data_dir = data_dir
			)


	def run_alphalink2_per_system( self, sys_name: str, gpu_id: int ):
		"""
		Run AlphaLink2 prediction for the given system with the default settings.
		Here I assume that the feature_dict already exist.
		We use the default setting specified for AlphaLink2.
		Added two arguments to run_alphalink2.sh to specify
			the device and the XL max bound.
		To run unguided prediction, an empty restraint file must be used.
		"""
		env = os.environ.copy()
		# This remaps the device numbering.
		env["CUDA_VISIBLE_DEVICES"] = str( gpu_id )
		# So the device must be changed to cuda:0.
		device = "cuda:0"
		# if "cuda" in self.device:
		# 	device = int( self.device.split( ":" )[-1] )
		# else:
		# 	device = self.device

		cmd = [
			"bash", f"{self.alphalink2_script}",
			f"{self.inputs['fasta_file'][sys_name]}",
			f"{self.inputs['restraints_file'][sys_name]}",
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
		We can reuse the pre-computed MSAs.
		To run unguided prediction, the yaml file must not have any
			constraints specified.
		"""
		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )
		xl_file = get_xl_file_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name,
			xl_type = self.xl_type
		)

		# xl_file = os.path.join( data_dir, f"interprotein_xls{self.sys_conf_suff}.csv" )
		# xl_df = pd.read_csv( xl_file )

		entity_chain_map = get_entity_chain_mapping(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		alignment_dir = os.path.join( data_dir, f"{sys_name}_output/" f"alignments/" )

		boltz_input = {"version": 1}
		boltz_input.update( {k:[] for k in ["sequences", 'constraints']} )
		# if not self.boltz_unguided:
		# 	boltz_input.pop( "constraints" )

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

		if self.model_config.guided_pred:
			xl_file = get_xl_file_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name,
				xl_type = self.xl_type
			)

			# Will add all Xl restraints as contacts for conditioning Boltz-2.
			for row in self.yield_restraints( xl_file, entity_chain_map, numeric_chain_ids = False ):
				( entity_id1, entity_id2, chain_id1,
					chain_id2, r1_idx, r2_idx ) = row

				contact = {
					"contact": {
						"token1": [chain_id1, int( r1_idx+1 )],
						"token2": [chain_id2, int( r2_idx+1 )],
						"max_distance": self.xl_max_bound
					}
				}
				boltz_input["constraints"].append( contact )
		else:
			boltz_input.pop( "constraints" )

		# Save as a yaml file.
		with open( self.inputs["restraints_file"][sys_name], "w" ) as w:
			yaml.safe_dump( boltz_input, w, sort_keys = False )


	def run_boltz2_per_system( self, sys_name: str, gpu_id: int ):
		"""
		Run Boltz2 prediction for the given system with the default settings.
		This is the easiet among the three to run.
		To run unguided prediction, the yaml file must not have any
			constraints specified.
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
			"--recycling_steps", f"{self.model_config.recycling_steps}",
			"--sampling_steps", f"{self.model_config.sampling_steps}",
			"--diffusion_samples", f"{self.model_config.diffusion_samples}",
			# "--use_msa_server"  # We use pre-computed alignments.
		]
		# subprocess.call() doe snot allow conrol over the process, so using Popen.
		proc = subprocess.Popen( cmd, env = env )
		return proc

	################################################################################
	################################################################################
	def create_inputs_for_benchmark( self ):
		"""
		Create input files for running GRASP/AlphaLink2 on the benchmark.
		"""
		for sys_name in self.benchmark["PDB ID"]:
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
		for idx, sys_name in enumerate( self.benchmark["PDB ID"] ):
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
					os.remove( pkl )

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
		"-x", "--xl_type",
		type = str, required = True,
		help = "cross-link type to be used: short/long/fp..." )
	parser.add_argument(
		"-d", "--device",
		type = str, required = True,
		help = "device to be used (cpu/cuda:0/cuda:1)..." )
	args = parser.parse_args()

	CompetingMethodsRunner(
		model = args.model,
		xl_type = args.xl_type,
		device = args.device
		).forward()
