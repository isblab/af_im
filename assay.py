import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from multiprocessing import Pool
import os
import glob

from Bio.PDB import PDBParser, PDBIO, Structure, Model

from utils import run_subprocess
from pdb_utils import Parser

from typing import Dict, Iterator

from create_plots import create_plot_from_dict


class Assay():
	def __init__( self, sys_name: str, base_dir, cores: int, prec: int, 
						model_ids: np.array,
						ensmeble_dir: str, output_dir: str ):
		self.sys_name = "2ayo"
		self.base_dir = base_dir
		self.cores = 5
		self.chain_break_threshold = 4.0
		self.prec = 4
		self.selection_statistic = "mean"
		self.model_ids = model_ids
		self.ensmeble_dir = ensmeble_dir
		
		self.output_dir = output_dir
		self.analysis_dir = os.path.join( self.output_dir, "analysis" )
		
		# self.parser = Parser( self.ensmeble_file )

		self.full_val_dict = {}
		self.full_val_dict_file = os.path.join( self.analysis_dir, "full_validation_dict.npy" )
		self.full_val_plot_file = os.path.join( self.analysis_dir, "full_validation_plots.png" )
		self.full_val_csv = os.path.join( self.analysis_dir, "full_validation_report.csv" )

		# self.selected_val_dict = {}
		# self.selected_val_dict_file = os.path.join( self.analysis_dir, "selected_validation_dict.npy" )
		# self.selected_val_plot_file = os.path.join( self.analysis_dir, "selected_validation_plots.png" )
		# self.selected_val_csv = os.path.join( self.analysis_dir, "selected_validation_report.csv" )
		# self.selected_avg_csv_file = os.path.join( self.analysis_dir, "selected_val_avg.csv" )



	def forward( self ):
		# idx = 0
		# chain_breaks_dict = {}
		# for coords_dict in self.parser.get_coordinates():
		# 	chain_breaks_dict[idx] = self.get_chain_breaks( coords_dict )
		# 	idx += 1

		# self.quantify_chain_breaks( chain_breaks_dict )
		self.create_required_dir()
		if not os.path.exists( self.full_val_dict_file ):
			self.parallelize_validation_pipeline()
			np.save( self.full_val_dict_file, self.full_val_dict )
		else:
			self.full_val_dict = np.load( self.full_val_dict_file, allow_pickle = True ).item()


		self.write_full_val_dict()

		for stat in ["mean", "median"]:
			self.selection_statistic = stat
			self.selected_val_dict = {}
			self.selected_val_dict_file = os.path.join( self.analysis_dir, f"selected_validation_dict_{stat}.npy" )
			self.selected_val_plot_file = os.path.join( self.analysis_dir, f"selected_validation_plots_{stat}.png" )
			self.selected_val_csv = os.path.join( self.analysis_dir, f"selected_validation_report_{stat}.csv" )
			self.selected_avg_csv_file = os.path.join( self.analysis_dir, f"selected_val_avg_{stat}.csv" )

			self.filter_models()
			self.write_selected_val_dict()

			# self.plot()

			# self.write_to_csv()

		self.remove_tmp_dir()



	def create_required_dir( self ):
		"""
		Create the following directories:
			analysis
				tmp
		Temporary directory is used for storing the intermediate files 
			from running Molprobity validation.
		"""
		if not os.path.exists( self.analysis_dir ):
			os.makedirs( self.analysis_dir )

		self.tmp_dir = os.path.join( self.analysis_dir, "tmp" )
		if not os.path.exists( self.tmp_dir ):
			os.makedirs( self.tmp_dir )



	def remove_tmp_dir( self ):
		cmd = ["rm", "-r", f"{self.tmp_dir}"]
		run_subprocess( cmd )



	# def write_model_to_pdb( self, model_id: str, model: Model, model_file: str ):
	# 	"""
	# 	Write a model to PDB file.
	# 	Required to run Molprobity on all models separately.
	# 	"""
	# 	io = PDBIO()

	# 	struct = Structure.Structure( f"Model_{model_id}" )
	# 	struct.add( model )

	# 	io.set_structure( struct )
	# 	io.save( model_file )


	def split_ensemble( self, model_id: str, model: Model, model_file: str ):
		"""
		Split the multi-model PDB into separate PDB files.
		This would avoid loading the structure on disk.
		"""

		io = PDBIO()

		struct = Structure.Structure( f"Model_{model_id}" )
		struct.add( model )

		io.set_structure( struct )
		io.save( model_file )



	def validation_pipeline( self, model_id ):
		"""
		Compute the no. of chain breaks.
		Perform Molprobity validation.
		"""
		# model_id, model = list( entry )
		# model_file = os.path.join( self.tmp_dir, f"{self.sys_name}_{model_id}.pdb" )
		model_file = os.path.join( self.ensmeble_dir, f"model_{model_id}.pdb" )
		molprob_output_dir = os.path.join( self.tmp_dir, f"molprob_{self.sys_name}_{model_id}" )

		# self.write_model_to_pdb( model_id, model, model_file )

		self.run_molprobity( model_file, molprob_output_dir )
		summary_dict = {}
		summary_dict[model_id] = self.get_molprobity_validation_summary( molprob_output_dir )
		# summary_dict = self.get_molprobity_validation_summary( molprob_output_dir )
		# summary_dict.update( {"model_id": model_id} )

		files_to_remove = glob.glob( f"{molprob_output_dir}*" )
		cmd = ["rm", "-r"] + files_to_remove
		run_subprocess( cmd )

		return summary_dict



	def parallelize_validation_pipeline( self ):
		"""
		Parse all models in parallel and perform all the required validations.
		"""
		# structure = self.parser.structure
		# model_ids = self.parser.get_model_ids()
		# entry_list = list( zip( model_ids, structure ) )

		with Pool( self.cores ) as p:
			for result in p.imap_unordered( self.validation_pipeline, self.model_ids ):
			# for result in p.imap_unordered( self.validation_pipeline, entry_list ):
				summary_dict = result

				self.full_val_dict.update( summary_dict )
				# for k, v in summary_dict.items():
				# 	self.full_val_dict.setdefault( k, [] ).append( v )



	def get_molprobity_validation_summary( self, output_dir: str ):
		"""
		Molprobity can provide detailed information about:
			Ramachandran outliers
						favored
			Rotamer outliers
			C-beta deviations
			Clashscore
			RMS(bonds)
			RMS(angles)
			MolProbity score
			Resolution
			R-work
			R-free
			Refinement program
		All this can be obtained from the *.out file generated by molprobity.
		"""
		with open( f"{output_dir}.out", "r" ) as f:
			summary = f.readlines()

		summary_dict = {}
		for i in range( len( summary ) ):
			line = summary[i]
			if "Summary" in line:
				for j in range( i+1, len( summary ) ):
					l = summary[j]

					split_line = l.strip().split( " = " )
					if len( split_line ) < 2:
						continue

					key, value = split_line 
					key = "".join( key.split( "  " ) )
					value = "".join( value.split( "  " ) )

					# Remove spaces from the end.
					key = key[:-1] if key[-1] == " " else key
					# Remove '%' symbols.
					value = value.replace( "%", "" )
					
					if value.isalpha():
						summary_dict[key] = value
					else:
						summary_dict[key] = float( value )

		return summary_dict



	def run_molprobity( self, model_file: str, output_dir: str ) -> None:
		"""
		Run Molprobity validation using Phenix.
		requires a PDB file with just 1 model.
		Generates the following files:
			{prefix}_coot.py  {prefix}.out  {prefix}.pkl  {prefix}.txt

		output_dir --> must be in the format "/path/prefix".
			where path is the output directory path and prefix is the 
				name for the output files generated.
		"""
		molprob_cmd = [
						"phenix.molprobity",
						f"{model_file}",
						f"output.prefix={output_dir}"
						]

		run_subprocess( molprob_cmd )



	def filter_models( self ):
		"""
		Here we use the Molprobity score as a criterion for assessing model quality.
		Select the good scoring models.
		"""
		selected_models = self.get_good_models()

		keys = ["model_id"] + list( self.full_val_dict[0].keys() )
		for model_id in self.full_val_dict.keys():
			if model_id in selected_models:
				# self.selected_val_dict["model_id"].append( model_id )

				for k in keys:
					if k == "model_id":
						v = model_id
					else:
						v = self.full_val_dict[model_id][k]
					self.selected_val_dict.setdefault( k, [] ).append( v )

		# for model_id in self.full_val_dict["model_id"]:
		# 	if model_id in selected_models:
				# for k, v in self.full_val_dict.items():
				# 	self.selected_val_dict.setdefault( k, [] ).append( v )



	# def cluster( self ):
	# 	"""
	# 	Use HDBSCAN for clustering models based on the Molprobity validation metrics.
	# 	"""




	def get_good_models( self ):
		"""
		To filter out bad models we consider the median Molprobity score.
			Median because mean is sensitive to outliers.
		We remove models with a Molprobity score higher than 1*MAD (median absolute deviation).
		"""
		molprob_scores = [self.full_val_dict[k]["MolProbity score"] for k in self.full_val_dict.keys()]

		multiplier = 1
		if self.selection_statistic == "mean":
			mean_score = np.mean( molprob_scores )
			# Compute the standard deviation.
			sd = np.std( molprob_scores )
			
			the_score = mean_score
			dev = sd
		
		elif self.selection_statistic == "median":
			median_score = np.median( molprob_scores )
			# Compute the abolsute deviations from the median.
			abs_dev = np.abs( molprob_scores - median_score )
			mad = np.median( abs_dev )
			
			the_score = median_score
			dev = mad

		selected_models = []

		# for i in range( len( molprob_scores ) ):
			# model_id = self.full_val_dict["model_id"][i]
		for model_id in self.full_val_dict.keys():
			score = self.full_val_dict[model_id]["MolProbity score"]
			# Calculate the lower and upper median bounds.
			lb = the_score - multiplier*dev
			ub = the_score + multiplier*dev

			if score >= lb and score <= ub:
				selected_models.append( model_id )

		return selected_models



	def write_full_val_dict( self ):
		"""
		Save the full validation dict on disk.
		Save as a .csv file.
		Plot the metrics.
		"""
		tmp_dict = {}
		# Need the dict to be storing lists.
		keys = ["model_id"] + list( self.full_val_dict[0].keys() )
		for model_id in self.full_val_dict.keys():
			for k in keys:
				if k == "model_id":
					v = model_id
				else:
					v = self.full_val_dict[model_id][k]
				tmp_dict.setdefault( k, [] ).append( v )
	
		df = pd.DataFrame( tmp_dict )
		# for k, v in tmp_dict.items():
		# 	df[k] = v
		df.to_csv( self.full_val_csv, index = False )

		_ = tmp_dict.pop( "model_id", None )
		create_plot_from_dict( tmp_dict, self.full_val_plot_file )


	def write_selected_val_dict( self ):
		"""
		Save the selected validation dict on disk.
		Save as a .csv file.
		Plot the metrics.
		"""		
		df = pd.DataFrame( self.selected_val_dict )

		df.to_csv( self.selected_val_csv, index = False )

		df = pd.DataFrame()
		for k, v in self.selected_val_dict.items():
			if k != "model_id":
				df[k] = [round( np.mean( v ), self.prec )]
		df.to_csv( self.selected_avg_csv_file, index = False )

		tmp_dict = self.selected_val_dict.copy()
		_ = tmp_dict.pop( "model_id", None )
		create_plot_from_dict( tmp_dict, self.selected_val_plot_file )



	# def write_to_csv( self ):
	# 	"""
	# 	Write validation dicts to csv file.
	# 	"""
	# 	tmp_dict = {}
	# 	# Need the dict to be storing lists.
	# 	keys = ["model_id"] + list( self.full_val_dict[0].keys() )
	# 	for model_id in self.full_val_dict.keys():
	# 		for k in keys:
	# 			if k == "model_id":
	# 				v = model_id
	# 			else:
	# 				v = self.full_val_dict[model_id][k]
	# 			tmp_dict.setdefault( k, [] ).append( v )
	
	# 	df = pd.DataFrame( tmp_dict )
	# 	# for k, v in tmp_dict.items():
	# 	# 	df[k] = v
	# 	df.to_csv( self.full_val_csv, index = False )

	# 	df = pd.DataFrame( self.selected_val_dict )

	# 	df.to_csv( self.selected_val_csv, index = False )

	# 	df = pd.DataFrame()
	# 	for k, v in self.selected_val_dict.items():
	# 		if k != "model_id":
	# 			df[k] = [round( np.mean( v ), self.prec )]
	# 	df.to_csv( self.selected_avg_csv_file, index = False )



	# def plot( self ):
	# 	tmp_dict = {}
	# 	# Need the dict to be storing lists.
	# 	keys = list( self.full_val_dict[0].keys() )
	# 	for model_id in self.full_val_dict.keys():
	# 		for k in keys:
	# 				v = self.full_val_dict[model_id][k]
	# 				tmp_dict.setdefault( k, [] ).append( v )
	# 	# tmp_dict = self.full_val_dict.copy()
	# 	# _ = tmp_dict.pop( "model_id", None )
	# 	create_plot_from_dict( tmp_dict, self.full_val_plot_file )

	# 	tmp_dict = self.selected_val_dict.copy()
	# 	_ = tmp_dict.pop( "model_id", None )
	# 	create_plot_from_dict( tmp_dict, self.selected_val_plot_file )




	# def get_distance_matrix( self, coords1: np.array, coords2: np.array ):
	# 	"""
	# 	Calculate Ca distance matrix.
	# 	"""
	# 	return distance_matrix( coords1, coords2 )



	# def get_chain_breaks( self, coords_dict: Dict[str, np.array] ):
	# 	"""
	# 	Given the coordinates for all chains in a model, calculate the Ca-Ca chain breaks.
	# 	A chain break is identified if Ca-Ca distance between consecutive residues is >5 Angstrom.
	# 	"""
	# 	chain_breaks_dict = {}
	# 	for chain in coords_dict.keys():
	# 		if chain not in chain_breaks_dict.keys():
	# 			chain_breaks_dict[chain] = {"breaks": [], # np.array( coords.shape[0] )
	# 										"total_bonds": 0}
	# 		coords = coords_dict[chain]

	# 		dist_mat = self.get_distance_matrix( coords, coords )

	# 		for i in range( coords.shape[0] - 1 ):
	# 			j = i + 1
	# 			if dist_mat[i, j] > self.chain_break_threshold:
	# 				chain_breaks_dict[chain]["breaks"].append( ( i, j, round( dist_mat[i, j], self.prec ) ) )
	# 			chain_breaks_dict[chain]["total_bonds"] += 1

	# 	return chain_breaks_dict




	# def quantify_chain_breaks( self, chain_breaks_dict ):
	# 	"""
	# 	Calculate global and samplewise average for the fraction of chain breaks.
	# 	"""
	# 	global_avg = 0
	# 	global_breaks = 0
	# 	total_bonds = 0
	# 	samplewise_avg = np.array( [] )

	# 	for model in chain_breaks_dict.keys():
	# 		samplewise_breaks = 0
	# 		for chain in chain_breaks_dict[model]:
	# 			num_breaks = len( chain_breaks_dict[model][chain]["breaks"] )
	# 			samplewise_breaks += num_breaks
	# 			global_breaks += num_breaks
	# 			total_bonds += chain_breaks_dict[model][chain]["total_bonds"]
	# 		samplewise_avg = np.append( samplewise_avg, samplewise_breaks )

	# 	print( global_breaks, "  ", total_bonds )
	# 	global_avg = round( global_breaks/total_bonds, self.prec )
	# 	samplewise_avg = round( np.mean( samplewise_avg ), self.prec )

	# 	print( global_avg, "  ", samplewise_avg )



if __name__ == "__main__":
	Assay().forward()

