"""
Contains tools to aid in analysis.
"""
from typing import List
import warnings, os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

from utils.utils import run_subprocess, open_file_handler

################################################################################
# ----------------------------------> EMAN2 <------------------------------- --#
################################################################################
def eman2_pdb2density(
	modeL_file: str,
	density_file: str,
	apix: float,
	res: float,
	box: List[float] ):
	"""
	Using EMAN2 for converting a structure to density map.
	EMAN2 is install in a separate conda environment.
		Run density map creation.
			e2pdb2mrc.py input_file output_file apix res
			apix -> angstrom per voxel.
			res -> required resolution of the density map.
				Note: res >= apix
	There is a potential (but harmless) bug in e2pdb2mrc.py script.
		When given an input XYZ coordinates of the bounding box,
			it selects the max of the XYZ coordinates.
	"""
	if res < apix:
		raise ValueError( f"res ({res}) <= apix ({apix}). " +
					"Generally res should be 2x apix or more" )

	# If the density file exists, EMAN2 appends to the header.
	if os.path.exists( density_file ):
		cmd = ["rm", f"{density_file}"]
		run_subprocess( cmd )

	box = list( map( str, box ) )
	cmd = [
		"conda",
		"run", "-n", "eman2",
		"e2pdb2mrc.py",
		f"{modeL_file}", f"{density_file}",
		"--apix", f"{apix}",
		"--res", f"{res}",
		"--box", f"{','.join( box )}",
		"--quiet"
		]

	run_subprocess( cmd )

################################################################################
# -------------------------------> MDAnalysis <------------------------------- #
################################################################################
def load_ensemble( struct_file: str ) -> Universe:
	"""
	Load the ensemble file on memory.

	Input:
	----------
	struct_file -> path to the structure file (may contain one or more  structures).
	"""
	u = mda.Universe( struct_file )
	return u


def get_selection( u: Universe ) -> AtomGroup:
	"""
	Select all CA-atoms from the universe.
	"""
	ca_atoms = u.select_atoms( "protein and name CA" )

	return ca_atoms


def get_residue_ids( ensemble_file ) -> np.ndarray:
	"""
	Given the selected CA-atoms, return the residue IDs.
	"""
	u = load_ensemble( ensemble_file = ensemble_file )
	ca_atoms = get_selection( u = u )
	return ca_atoms.resids


def align_models( u: Universe, ref: Universe, ref_frame = 0 ) -> Universe:
	"""
	Align all models to a given model.
	"""
	aligner = align.AlignTraj( u, ref,
			select = "protein and name CA",
			ref_frame = ref_frame,
			in_memory = True ).run()
	return u


def create_average_model( u: Universe ) -> Universe:
	"""
	Compute an average model, given all the models in the ensemble.
	"""
	average = align.AverageStructure( u, u,
									select = "protein and name CA",
									ref_frame = 0 ).run()
	avg_model = average.results.universe
	return avg_model


def compute_rmsf( ca_atoms: AtomGroup ) -> np.ndarray:
	"""
	Compute the RMSF using the MDAnalysis package.
	rmsf -> [R]; where R is the no. of residues in a model.
	"""
	R = rms.RMSF( ca_atoms )
	R.run()
	rmsf = R.rmsf
	return rmsf


def compute_rmsd( to_align: Universe, ref: Universe, ref_frame: int = 0 ) -> np.ndarray:
	"""
	Compute the RMSd for all models wrt the first model
		using the MDAnalysis package.
	Returns a row for each timestep.
	rmsd -> [N,3]; where N is the total no. of models.
	For each model it gives - [frame no., timestep, rmsd for selection].
	Input can be a universe or AtomGroup.
	"""
	R = rms.RMSD( to_align, ref, select = "protein and name CA", ref_frame = ref_frame )
	R.run()
	# rmsd = R.rmsd
	rmsd = R.results.rmsd
	return rmsd


def compute_rmsf_wrt_avg_model( ensemble_file: str ) -> np.ndarray:
	"""
	Source: https://userguide.mdanalysis.org/stable/examples/analysis/alignment_and_rms/rmsf.html
	Given the ensemble_file,
		Load the models (Universe).
		Get an average model.
		Align all models wrt the average model.
		Select CA-atoms from the universe.
		Compute RMSF.
	"""
	warnings.filterwarnings( "ignore" ) 
	u = load_ensemble( ensemble_file = ensemble_file )
	avg_model = create_average_model( u = u )
	# This aligns the u inplace. So need to reload before computing RMSF.
	aligned_u = align_models(
		u = load_ensemble( ensemble_file = ensemble_file ),
		ref = avg_model, ref_frame = 0 )
	ca_atoms = get_selection( u = aligned_u )
	rmsf = compute_rmsf( ca_atoms = ca_atoms )

	return rmsf


def compute_rmsf_wrt_first_model( ensemble_file: str ) -> np.ndarray:
	"""
	Given the ensemble_file,
		Load the models (Universe).
		Align all models wrt the first model.
		Select CA-atoms from the universe.
		Compute RMSF.
	"""
	warnings.filterwarnings( "ignore" ) 
	u = load_ensemble( ensemble_file = ensemble_file )
	# This aligns the u inplace. So need to reload before computing RMSF.
	aligned_u = align_models(
		u = load_ensemble( ensemble_file = ensemble_file ),
		ref = u, ref_frame = 0 )
	ca_atoms = get_selection( u = aligned_u )
	rmsf = compute_rmsf( ca_atoms = ca_atoms )

	return rmsf


def compute_rmsd_post_align( ensemble_file: str ) -> np.ndarray:
	"""
	Source: https://userguide.mdanalysis.org/stable/examples/analysis/alignment_and_rms/rmsd.html
	Given the ensemble_file,
		Load the models (Universe).
		Align all models to the first model.
		Select CA-atoms from the universe.
		Compute RMSD wrt first model.
	RMSD returned is in Angstorm.
	"""
	warnings.filterwarnings( "ignore" )
	u = load_ensemble( ensemble_file = ensemble_file )
	rmsd = compute_rmsd( to_align = u, ref = u )

	return rmsd

################################################################################
# ----------------------------------> DockQ <--------------------------------- #
################################################################################
def dockq( native_file: str, model_file: str ) -> float:
	"""
	Given the path to the native and the predicted structure (model),
		compute the DockQ metric.
	Ensure that the chain IDs are the same in the native
		and the predicted structure.
	"""
	model = load_PDB( model_file )
	native = load_PDB( native_file )
	dockq_result = run_on_all_native_interfaces( model, native )
	dockq = dockq_result[1]

	return dockq


def dockq_cli( native_file: str, model_file: str, out_file: str ) -> float:
	"""
	Use the DockQ CLI for computing the DockQ between the native and model structures.
	This computes the optimal alignment by itself and does not require remapping the chains
		in the input structures.
	We just read the Total DockQ line. Example,
		Total DockQ over 3 native interfaces: 0.653 with BAC:ABC model:native mapping

	Inputs:
	----------
	native_file: path to the native structure file.
	model_file: path to the model structure file.
	out_file: path to the file for redirecting the DockQ output.
	"""
	cmd = [
		"DockQ",
		f"{model_file}",
		f"{native_file}",
		"--short"
	]
	run_subprocess( command = cmd, stdout_file = out_file )

	f = open_file_handler( out_file, "r" )
	for line in f.readlines():
		if "Total DockQ over " in line:
			dockq = float( line.strip().split( ": " )[1][:3] )
	f.close()

	cmd = ["rm", f"{out_file}"]
	run_subprocess( cmd )

	return dockq

################################################################################
# ---------------------------------> USalign <-------------------------------- #
################################################################################
def usalign(
	usalign_script: str,
	model_id1: int,
	model1_file: str,
	model2_file: str,
	model_id2: int,
	tmp_dir: str,
	mol: str = "prot",
	mm: int = 1,
	ter: int = 1 ):
	"""
	Use USalign for computing the TM-score
		and RMSD for the given models.
	Assuming model_id1 to be the reference.

	mol --> molecule type [auto, prot, RNA].
	mm --> multimeric laignment option.
		0: alignment of two monomeric structures.
		1: alignment of two multi-chain oligomeric structures.
		2: alignment of individual chains to an oligomeric structure.
		Look at USalign -h option for more details.
	ter --> #chains to align.
		0: align all chains from all models.
		1: align all chains of the first model.
		2: only align the first chain.

	USalign model1.pdb model2.pdb -ter 0 -mm 1 -mol prot
	"""
	# model1 = get_struct_file( model_id = model_id1 )
	# model2 = get_struct_file( model_id = model_id2 )

	stdout_file = os.path.join( tmp_dir, f"model_{model_id1}_{model_id2}.txt" )
	stderr_file = os.path.join( tmp_dir, f"error_{model_id1}_{model_id2}.txt" )

	cmd = [f"./{usalign_script}", 
			f"{model1_file}",
			f"{model2_file}",
			"-mol", f"{mol}",
			"-mm", f"{mm}",
			"-ter", f"{ter}"]

	run_subprocess(
		command = cmd,
		stdout_file = stdout_file,
		stderr_file = stderr_file
	)
	return stdout_file


def get_alignment_score( stdout_file: str ):
	"""
	Return the TM-score and RMSD.
	Read the MMalign/USalign output stored in a txt file.
		Line14: Aligned length= 572, RMSD=   0.76, Seq_ID=n_identical/n_aligned= 1.000
		Line15: TM-score= 0.XXXXX (if normalized by length of Chain_1, i.e., LN=XX, d0=X.XX)
		Line16: TM-score= 0.XXXXX (if normalized by length of Chain_2, i.e., LN=XXX, d0=X.XX)
	"""
	with open( stdout_file, "r" ) as f:
		output = f.readlines()

	rmsd_line = output[14]
	tm_line = output[15]

	rmsd = float( rmsd_line.split( "," )[1].split( "RMSD=" )[1] )
	tm = float( tm_line.split( " " )[1] )
	return rmsd, tm
