"""
This module uses the OpenFold data pipeline to create MSAs
	and input features.
"""
from typing import List, Tuple, Dict, Any
import os, pathlib, shutil, time, re, argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
import tqdm

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
# from openfold.utils.script_utils import parse_fasta

from config import get_config_dict
from utils.utils import (
	seed_worker,
	log_error,
	open_file_handler,
	read_json,
	write_json,
	write_pkl_gz,
	parse_nested_dict
)
from utils.pdb_utils import get_chain_id
from utils.paths import (
	get_sys_data_dir_path,
	get_meta_dir_path,
	get_sys_config_path,
	get_openfold_alignments_dir_path,
	get_openfold_output_dir_path,
	get_openfold_pred_dir_path,
	get_sys_fasta_file_path,
	get_feature_dict_path,
	get_processed_feature_dict_path
)
from utils.paths import get_benchmark_csv_file

def list_files_with_extensions( dir, extensions ):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]

def parse_fasta(data):
	"""
	Taken from openfold/utils/script_utils/
	"""
	data = re.sub('>$', '', data, flags=re.M)
	lines = [
		l.replace('\n', '')
		for prot in data.split('>') for l in prot.strip().split('\n', 1)
	][1:]
	tags, seqs = lines[::2], lines[1::2]

	tags = [re.split('\W| \|', t)[0] for t in tags]

	return tags, seqs



class MsaPipeline():
	"""
	Use OpenFold data pipeline for creating the MSAs and input features.
	"""
	def __init__(
		self,
		benchmark_name: str,
		is_multimer: bool
		):
		self.config_dict = get_config_dict( is_multimer = is_multimer )
		self.base_dir = os.path.join(
			os.path.abspath( self.config_dict.benchmark.globals.base_dir )
			)
		self.benchmark_name = benchmark_name
		self.msa_configs = self.config_dict.msa

		self.is_multimer = is_multimer
		self.db_dir = self.msa_configs.db_dir
		self.config_preset = self.msa_configs.config_preset
		self.db_preset = self.msa_configs.db_preset
		self.tool_base = self.msa_configs.tool_base
		self.openfold_checkpoint_path = self.msa_configs.model_checkpoint
		self.jax_params_path = self.msa_configs.jax_params_path
		self.save_feature_dicts = self.msa_configs.save_feature_dicts

		self.cpu_cores = self.msa_configs.cpu_cores
		self.max_template_date = self.msa_configs.max_template_date
		self.databases_n_tools = self.msa_configs.databases_n_tools

		seed_worker( seed = self.config_dict.prng_seed )

		# Will be created downstream.
		self.feature_dict = {}
		self.processed_feature_dict = {}
		self.benchmark = pd.DataFrame()
		# Initialize a dict to store file paths.
		self.file_paths = {}
		# Logs
		self.logs = {}


	def forward( self ):
		"""
		"""
		s = time.perf_counter()
		self.init_logs()
		print( "Loading the benchmark..." )
		self.load_benchmark()
		# Initialize the dict containing the file paths.
		self.create_file_paths_for_msa_creation()

		self.init_ofold_config()
		self.init_feature_processor()
		print( "Initialized the feature processor..." )
		self.run_msa_creation_in_parallel()
		# Save the selected complexes on disk.
		self.save_selected_complexes()
		e = time.perf_counter()
		self.logs["total_time"] = e-s

		write_json( self.logs, self.logs_file )
		print( "May the Force be with you..." )

	################################################################################
	################################################################################
	def init_logs( self ):
		"""
		Log time taken per system and errors.
		Also create the logs file path.
		"""
		meta_dir = get_meta_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name
		)
		self.logs_file = os.path.join( meta_dir, "Logs_msa.json" )

		if os.path.exists( self.logs_file ):
			self.logs = read_json( self.logs_file )
		else:
			self.logs = {
				"time": {},
				"errored": [],
				"selected": []
			}


	def load_benchmark( self ):
		"""
		Load the benchmark .csv file.
		"""
		benchmark_file = get_benchmark_csv_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			raw_file = True
		)
		self.benchmark = pd.read_csv( benchmark_file )
		# self.benchmark = self.benchmark.iloc[:2]
		# print( self.benchmark.columns )


	def init_feature_processor( self ):
		self.feature_processor = feature_pipeline.FeaturePipeline( self.ofold_config.data )


	def init_ofold_config( self ):
		self.ofold_config = model_config(
			self.config_preset,
			long_sequence_inference = False,
			use_deepspeed_evoformer_attention = False,
			)


	def create_file_paths_for_msa_creation( self ):
		"""
		Create the following file paths for each system:
			alignments dir
				{sys_dir}/{sys_name}_outputs/alignments/
			fasta file path.
			feature_dict path.
			processed feature_dict path.
		Create the OpenFold output dir, alignemnt dir, and
			predictions dir.
		"""
		for sys_name in self.benchmark["PDB ID"]:
			ofold_output_dir = get_openfold_output_dir_path(
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					sys_name = sys_name )
			alignment_dir = get_openfold_alignments_dir_path(
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					sys_name = sys_name )
			pred_dir = get_openfold_pred_dir_path(
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					sys_name = sys_name )
			fasta_file_path = get_sys_fasta_file_path(
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					sys_name = sys_name )
			feature_dict_path = get_feature_dict_path(
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					sys_name = sys_name )
			processed_feature_dict_path = get_processed_feature_dict_path(
					base_dir = self.base_dir,
					benchmark_name = self.benchmark_name,
					sys_name = sys_name )

			self.file_paths[sys_name] = {
				"ofold_output_dir": ofold_output_dir,
				"alignment_dir": alignment_dir,
				# "preds_dir": pred_dir,
				"fasta_file": fasta_file_path,
				"feature_dict": feature_dict_path,
				"processed_feature_dict": processed_feature_dict_path,
			}
			for dir_ in [ofold_output_dir, alignment_dir, pred_dir]:
				os.makedirs( dir_, exist_ok = True )

	################################################################################
	################################################################################
	def run_msa_creation_in_parallel( self ):
		"""
		Parallelize MSA creation for the benchmark.
		Distributes per-system MSA creation across multiple cores
			and aggregates success/failures.
		imap_unordered provides lazy task scheduing and is faster.
		Terminate all workers  on keyboardInterrupt.
		For other errors just log the error and let the workers run.
		"""
		pool = Pool( self.cpu_cores )
		try:
			print( "Creating alignments and input features..." )
			iterator = pool.imap_unordered(
				self.create_msa_per_system,
				self.benchmark["PDB ID"]
			)

			for result in tqdm.tqdm( iterator, total = len( self.benchmark["PDB ID"] ) ):
				sys_name, errored, time_taken = result
				if errored:
					if sys_name not in self.logs["errored"]:
						self.logs["errored"].append( sys_name )
				else:
					if sys_name not in self.logs["selected"]:
						self.logs["selected"].append( sys_name )
				if sys_name not in self.logs["time"]:
					self.logs["time"][sys_name] = time_taken
				write_json( self.logs, self.logs_file )
			pool.close()
			pool.join()
		except KeyboardInterrupt:
			print("\nKeyboardInterrupt detected. Terminating workers...")
			pool.terminate()
			pool.join()
			raise


	def create_msa_per_system( self, sys_name: str ) -> Tuple[str, bool, float]:
		"""
		Use OpenFold MSA creation data pipeline for the given system.
			Create the fasta file for the system.
			Run the OpenFold data pipeline.
			Save the feature dicts on disk.
		For KeyboardInterrupt, re-raise exception and allow
			shuting down the process.
		For other errors, log the error to a system-specific file and
			notify the upstream process of th error without raising.

		Input:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.

		Returns:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		errored: True if an error occurs, else false.
		"""
		errored = False
		time_taken = 0.0
		try:
			t_s = time.perf_counter()
			self.create_sys_fasta_file( sys_name = sys_name )
			# Get the input features (See AF2 supplementary).
			feature_dict, processed_feature_dict, tag = self.prepare_input( sys_name = sys_name )

			processed_feature_dict = parse_nested_dict(
				processed_feature_dict, "detach" )

			write_pkl_gz(
				data = feature_dict,
				file_path = self.file_paths[sys_name]["feature_dict"] )
			write_pkl_gz(
				data = processed_feature_dict,
				file_path = self.file_paths[sys_name]["processed_feature_dict"] )
			t_e = time.perf_counter()
			time_taken = t_e - t_s
		except KeyboardInterrupt as e:
			print( f"Keyboard Interrupt while processing {sys_name}..." )
			raise
		except:
			errored = True
			sys_dir = get_sys_data_dir_path(
				base_dir = self.base_dir,
				benchmark_name = self.benchmark_name,
				sys_name = sys_name )
			error_file = os.path.join( sys_dir, f"error_msa_creation.txt" )
			log_error( error_file = error_file )
			print( f"An error occured during MSA creation for {sys_name}. " +
					f"Check error log in {error_file}...\n" )
		return sys_name, errored, time_taken

	################################################################################
	################################################################################
	def create_sys_fasta_file( self, sys_name: str ):
		"""
		Create a fasta file containing the sequences to be modeled for
			the given system.
			Skip if fasta file already exists.

		Input:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		"""
		fasta_file = self.file_paths[sys_name]["fasta_file"]
		# if os.path.exists( fasta_file ):
		# 	pass
		# else:
		sys_conf_file = get_sys_config_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		sys_conf_dict = read_json( file_path = sys_conf_file )

		w = open_file_handler( file_path = fasta_file, mode = "w" )
		chain = 0
		# All entities in the system.
		for entity in sys_conf_dict["entity"]:
			entity_id = entity["entity_id"]
			seq = entity["sequence"]
			# All chains in an entity.
			for cp in range( entity["copy_num"] ):
				chain_id = get_chain_id( chain )
				header = f"{sys_name}_{entity_id}_{chain_id}"
				w.writelines( f">{header}\n{seq}\n" )
				chain += 1
		w.close()

	################################################################################
	def alignments_exist( self,
		local_alignment_dir: str
	) -> bool:
		"""
		Check if all the required alignment files exist or not.
		For monomers,
			bfd_unirefclust_hits.a3m
		For multimers,
			bfd_unirefclust_hits.a3m
			hmm_output.sto
			mgnify_hits.sto
			uniprot_hits.sto
			uniref90_hits.sto

		Inputs:
		----------
		local_alignment_dir: dir to store the MSA files and
			template-search results.

		Returns:
		----------
		aln_exist: True if all required fils exists.
		"""
		monomer_files = ["bfd_unirefclust_hits.a3m"]
		multimer_files = [
			"bfd_unirefclust_hits.a3m", "hmm_output.sto",
			"mgnify_hits.sto", "uniprot_hits.sto", "uniref90_hits.sto"
		]
		exists = []
		if self.is_multimer:
			for file_name in multimer_files:
				file_path = os.path.join(
					local_alignment_dir, file_name )
				exists.append( os.path.exists( file_path ) )
		else:
			for file_name in monomer_files:
				file_path = os.path.join(
					local_alignment_dir, file_name )
				exists.append( os.path.exists( file_path ) )
		aln_exist = all( exists )
		return aln_exist

	################################################################################
	def get_alignment( self,
		tmp_fasta_path: str,
		local_alignment_dir: str
	):
		"""
		Generate the sequence alignments and template-search results for the
			given fasta input.
		Template search is performed using,
			HHSearch for monomer.
			Hmmsearch for multimer.
		Perform alignment for the query sequence.
		Uses the openFold AlignmentRunner for creating the MSAs.
		Currently fixed the no. of CPU cores to be used by JaxkHMER to 1.

		Inputs:
		----------
		tmp_fasta_path: absolute path to the temp fasta file for the given
			query sequence.
		local_alignment_dir: dir to store the MSA files and
			template-search results.
		"""
		if self.is_multimer:
			template_searcher = hmmsearch.Hmmsearch(
				binary_path = self.databases_n_tools.hmmsearch_binary_path,
				hmmbuild_binary_path = self.databases_n_tools.hmmbuild_binary_path,
				database_path = self.databases_n_tools.pdb_seqres_database_path,
			)
		else:
			template_searcher = hhsearch.HHSearch(
				binary_path = self.databases_n_tools.hhsearch_binary_path,
				databases = [self.databases_n_tools.pdb70_database_path],
			)

		alignment_runner = data_pipeline.AlignmentRunner(
			jackhmmer_binary_path = self.databases_n_tools.jackhmmer_binary_path,
			hhblits_binary_path = self.databases_n_tools.hhblits_binary_path,
			uniref90_database_path = self.databases_n_tools.uniref90_database_path,
			mgnify_database_path = self.databases_n_tools.mgnify_database_path,
			bfd_database_path = self.databases_n_tools.bfd_database_path,
			uniref30_database_path = self.databases_n_tools.uniref30_database_path,
			uniclust30_database_path = self.databases_n_tools.uniclust30_database_path,
			uniprot_database_path = self.databases_n_tools.uniprot_database_path,
			template_searcher = template_searcher,
			use_small_bfd = self.databases_n_tools.bfd_database_path is None,
			no_cpus = 1
		)

		alignment_runner.run(
			tmp_fasta_path, local_alignment_dir
		)


	def precompute_alignments( self,
		sys_name: str,
		tags: List[str],
		seqs: List[str] ):
		"""
		Taken from run_pretrained_openfold.py.
		Precompute and cache alignments for all input sequences.
		OpenFold inference runs alignment for all copies in case of a homomeric input.
			We reuse the alignment for homomers.
		For each ( tag, sequence ) pair,
			Write the sequence to a temporary fasta file.
			Create a dir to store alignments for each sequence.
			Create alignments anew if not already existing.
				Reuse alignments for homomeric sequences.
			Remove the temporary fasta file.

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		tags: identifier for the chain.
		seqs: sequence for the corresponding tag.
		"""
		ofold_output_dir = self.file_paths[sys_name]["ofold_output_dir"]
		# Keep track of the tag and the associated seq.
		tmp_dict = {}
		for tag, seq in zip( tags, seqs ):
			# print( f"Initiating alignment creation for {tag}..." )
			local_alignment_dir = os.path.join(
				self.file_paths[sys_name]["alignment_dir"],
				tag )

			# Skip alignments already exist for a chain.
			if self.alignments_exist( local_alignment_dir = local_alignment_dir ):
				print( f"Using precomputed alignments for {tag} at {local_alignment_dir}..." )
				continue

			tmp_fasta_path = os.path.join(
				ofold_output_dir, f"tmp_{os.getpid()}.fasta" )
			w = open_file_handler( tmp_fasta_path, "w" )
			w.write( f">{tag}\n{seq}" )
			w.close()

			# if args.use_precomputed_alignments is None:
			# if not os.path.exists( local_alignment_dir ):
			print( f"Generating alignments for {tag}..." )
			os.makedirs( local_alignment_dir, exist_ok = True )
			# For identical sequences reuse the alignments.
			if seq in tmp_dict:
				prev_local_alignment_dir = tmp_dict[seq]["local_alignment_dir"]
				src = pathlib.Path( prev_local_alignment_dir )
				dst = pathlib.Path( local_alignment_dir )
				for p in src.iterdir():
					shutil.copy( str( p ), str( dst ) )
				print( f"Reusing alignments for {tag} from {tmp_dict[seq]['tag']}..." )
			else:
				self.get_alignment(
					tmp_fasta_path = tmp_fasta_path,
					local_alignment_dir = local_alignment_dir
				)
				tmp_dict[seq] = {
					"tag": tag,
					"local_alignment_dir": local_alignment_dir
				}
			# else:
			# 	print( f"Using precomputed alignments for {tag} at {local_alignment_dir}..." )

			# Remove temporary FASTA file
			os.remove( tmp_fasta_path )


	def generate_feature_dict( self,
		sys_name: str,
		tags,
		seqs,
		data_processor
		) -> Dict[str, np.ndarray]:
		"""
		Taken from run_pretrained_openfold.py.
		Create input features from sequence and alignments.
		Writes the sequence to a temporary FASTA, deleted later, file and uses
			the OpenFold data pipeline to construct the input features.

		Inputs:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		tags: identifier for the chain.
		seqs: sequence for the corresponding tag.
		data_processor: wrapper over the OpenFold DataPipeline and
			DataPipelineMultimer that creates input features from the
			sequence and alignments.

		Returns:
		----------
		feature_dict: input features required for structure prediction.
			See contents in self.prepare_input().
		"""
		ofold_output_dir = self.file_paths[sys_name]["ofold_output_dir"]
		alignment_dir = self.file_paths[sys_name]["alignment_dir"]
		tmp_fasta_path = os.path.join( ofold_output_dir, f"tmp_{os.getpid()}.fasta" )

		with open(tmp_fasta_path, "w") as fp:
			fp.write(
				'\n'.join([f">{tag}\n{seq}" for tag, seq in zip( tags, seqs)] )
			)
		feature_dict = data_processor.process_fasta(
			fasta_path = tmp_fasta_path, alignment_dir = alignment_dir,
		)

		# Remove temporary FASTA file
		os.remove( tmp_fasta_path )

		return feature_dict

	################################################################################
	def prepare_input( self, sys_name: str ):
		"""
		Taken from run_pretrained_openfold.py -> main().
		Create input features for running AF2.
		feature_dict contains
			aatype, residue_index, seq_length, msa, num_alignments,
			template_aatype, template_all_atom_mask, template_all_atom_positions,
			asym_id, sym_id, entity_id,
			deletion_matrix, deletion_mean,
			all_atom_mask, all_atom_positions,
			assembly_num_chains, entity_mask, num_templates,
			cluster_bias_mask, bert_mask, seq_mask, msa_mas
		processed_feature_dict contains
			aatype, residue_index, seq_length, msa, num_alignments,
			template_aatype, template_all_atom_mask, template_all_atom_positions,
			asym_id, sym_id, entity_id,
			deletion_matrix, seq_mask, msa_mask, msa_profile, target_feat,
			atom14_atom_exists, residx_atom14_to_atom37, residx_atom37_to_atom14, atom37_atom_exists,
			extra_msa, extra_deletion_matrix, extra_msa_mask, bert_mask, true_msa,
			cluster_profile, cluster_deletion_mean, msa_feat, use_clamped_fape
		The shapes of all features can be found in openfold config.py.

		Input:
		----------
		sys_name: we use the PDB ID as a unique identifier for a complex.
			Aka entry_id.
		"""
		if self.is_multimer:
			template_featurizer = templates.HmmsearchHitFeaturizer(
				mmcif_dir = self.databases_n_tools.template_mmcif_dir,
				max_template_date = self.max_template_date,
				max_hits = self.ofold_config.data.predict.max_templates,
				kalign_binary_path = self.databases_n_tools.kalign_binary_path,
				# Path to a file with a mapping from PDB IDs to their release dates.
				#	 Thanks to this we don't have to redundantly parse mmCIF files to get that information.
				release_dates_path = None,
				# contains a mapping from obsolete PDB IDs to the PDB IDs of their replacements.
				obsolete_pdbs_path = None
			)
		else:
			template_featurizer = templates.HhsearchHitFeaturizer(
				mmcif_dir = self.databases_n_tools.template_mmcif_dir,
				max_template_date = self.max_template_date,
				max_hits = self.ofold_config.data.predict.max_templates,
				kalign_binary_path=self.databases_n_tools.kalign_binary_path,
				release_dates_path = None,
				obsolete_pdbs_path = None
			)
		data_processor = data_pipeline.DataPipeline(
			template_featurizer = template_featurizer,
		)
		if self.is_multimer:
			# For multimer.
			data_processor = data_pipeline.DataPipelineMultimer(
				monomer_data_pipeline = data_processor,
			)

		# The system data dir contains the fasta file.
		data_dir = get_sys_data_dir_path(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			sys_name = sys_name )

		tag_list = []
		seq_list = []
		for fasta_file in list_files_with_extensions( data_dir, (".fasta", ".fa") ):
			# Gather input sequences
			fasta_path = os.path.join ( data_dir, fasta_file )
			with open( fasta_path, "r" ) as fp:
				data = fp.read()

			tags, seqs = parse_fasta( data )

			if not self.is_multimer and len( tags ) != 1:
				print(
					f"{fasta_path} contains more than one sequence but " +
					f"multimer mode is not enabled. Skipping..."
				)
				continue

			tag = '-'.join(tags)

			tag_list.append( ( tag, tags ) )
			seq_list.append( seqs )

		seq_sort_fn = lambda target: sum( [len( s ) for s in target[1]] )
		sorted_targets = sorted( zip( tag_list, seq_list ), key = seq_sort_fn )

		if len( sorted_targets ) == 0:
			print( sorted_targets )
			raise ValueError( f"No fasta files found for {sys_name}..." )
		elif len( sorted_targets ) > 1:
			print( sorted_targets )
			raise ValueError( f"Multiple fasta files ({len( sorted_targets )}) detected for {sys_name}..." )
		( tag, tags ), seqs = sorted_targets[0]

		# Precompute aignments.
		self.precompute_alignments( sys_name = sys_name, tags = tags, seqs = seqs )

		feature_dict = self.generate_feature_dict(
			sys_name,
			tags,
			seqs,
			data_processor
		)

		processed_feature_dict = self.feature_processor.process_features(
			feature_dict, mode = "predict", is_multimer = self.is_multimer
		)

		return feature_dict, processed_feature_dict, tag

	################################################################################
	################################################################################
	def save_selected_complexes( self ):
		"""
		Subset the benchmark and svae the selected complexes on disk.
		"""
		drop_rows = []
		benchmark_file = get_benchmark_csv_file(
			base_dir = self.base_dir,
			benchmark_name = self.benchmark_name,
			raw_file = False
		)
		for idx in self.benchmark.index:
			pdb_id = self.benchmark.loc[idx, "PDB ID"]
			if pdb_id not in self.logs["selected"]:
				drop_rows.append( idx )
		subset_benchmark = self.benchmark.drop( drop_rows )
		subset_benchmark.to_csv( benchmark_file, index = False )


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description = "Create MSA using the OpenFold pipeline."
	)
	parser.add_argument(
		"-b", "--benchmark",
		type = str, required = True,
		help = "Specify the benchmark: crosslink/multistate." )
	parser.add_argument(
		"-p", "--pred_type",
		required = False, action = "store_true",
		default = True,
		help = "If specified, consider multimer prediction, else monomer." )
	args = parser.parse_args()

	MsaPipeline(
		benchmark_name = args.benchmark,
		is_multimer = args.pred_type
	).forward()

