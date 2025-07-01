"""
Contains functions to perform AMBER relaxation.
Taken from OpenFold.
"""
from typing import Optional
import ml_collections as mlc
from openfold.np import protein
from openfold.np.relax import relax


def amber_relaxation(
				unrelaxed_protein: protein.Protein,
				model_num: int,
				amber_config: mlc.ConfigDict,
				relaxed_output_path: str,
				device: Optional[str] = "cpu",
				cif_output: optional[bool] = False ):
	"""
	Perform AMBER relaxation for the predicted structure.
	# Taken from openfold.utils.script_utils.py.
	"""
	# Not making too many changes.
	model_device = device
	cif_output = False
	config = self.ofold_config
	
	amber_relaxer = relax.AmberRelaxation(
		use_gpu=(model_device != "cpu"),
		**config.relax,
	)

	t = time.perf_counter()
	visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
	if "cuda" in model_device:
		device_no = model_device.split(":")[-1]
		os.environ["CUDA_VISIBLE_DEVICES"] = device_no
	# the struct_str will contain either a PDB-format or a ModelCIF format string
	struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein, cif_output=cif_output)
	os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
	relaxation_time = time.perf_counter() - t

	# logger.info(f"Relaxation time: {relaxation_time}")
	# update_timings({"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json"))

	# Save the relaxed PDB.
	suffix = "_relaxed.pdb"
	if cif_output:
		suffix = "_relaxed.cif"

	with open(relaxed_output_path, 'w') as fp:
		fp.write(struct_str)

	print( f"Relaxed output written to {relaxed_output_path}..." )
	# logger.info(f"Relaxed output written to {relaxed_output_path}...")
