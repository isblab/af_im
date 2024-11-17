import subprocess
import os
import ml_collections as mlc

from utils import download_pdb, run_subprocess


def get_benchmark():
	"""
	Returns a list containing all benchmark systems.
	"""
	benchmark = mlc.ConfigDict( 
		{
		"System1": {
			"pdb_id": "2ayo",
			"chains": [
				{
				"uni_id": "",
				"copy_num": 1,
				"start": ,
				"end": 
				},
				{
				"uni_id": "",
				"copy_num": 1,
				"start": ,
				"end": 
				}
			]
		},
		}
	 )

	return benchmark



def initialize_benchmark_dir():
	"""
	For all systems in the benchmark, do:
		Create a base_dir named "benchmark/".
		Create a directoty for each system in the benchmark.
		Download the .pdb file for each system.
		Fetch the UniProt sequence for all chains in the system.
		Map PDB to UniProt with SIFTS.
			We consider the UniProt sequence in the PDB as the system to be modeled.
		Create a dir containing FASTA file for the system to be modeled.
	"""
	benchmark_dir = "./benchmark/"
	if not os.path.exists( benchmark_dir ):
		os.makedirs( benchmark_dir )

	benchmark = get_benchmark()

	for system in benchmark:
		sys_name = system.pdb_id

		sys_dir = f"{benchmark_dir}{sys_name}/"
		if not os.path.exists( sys_dir ):
			os.makedirs( sys_dir )

		print( f"Downloading .pdb file for {sys_name}..." )
		_ = download_pdb( pdb_id = sys_name, ext = "pdb", max_trials = 10, wait_time = 10, return_id = False )
		if os.path.exists( f"./{sys_name}.pdb" ):
			cmd = ["mv", f"./{sys_name}.pdb", f"{sys_dir}{sys_name}.pdb"]
			run_subprocess( cmd )
		else:
			raise Exception( f"Couldn't download .pdb file for {sys_name}..." )


if __name__ == "__main__":
	initialize_benchmark_dir()
