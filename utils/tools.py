"""
Contains tools to aid in analysis.
"""
import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup


################################################################################
# ------------------------------> RMSD and RMSF <----------------------------- #
################################################################################
def load_ensemble( ensemble_file: str ):
	"""
	Load the ensemble file on memory.
	"""
	u = mda.Universe( ensemble_file )
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