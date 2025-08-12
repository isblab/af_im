# IMP DL


## Installation
### OpenFold
Unzip the openfold.tar.gz file. This has been cloned from the pl_upgrades branch in openfold git repo (for Cuda12) (https://github.com/aqlaboratory/openfold.git).  
Minor modifications have been in script_utils/prep_output().  
We are using the pl_upgrades branch from the openfold git repo.  
For installation, see instructions in OpenFold gdoc.  

### JWalk
Clone the git repo from https://github.com/Topf-Lab/Jwalk.git.  
It's implemented in Python2, so I converted the code to Python3 using python2to3.com server.  
For installation, run  
```
python setup.py install
```

### imp_dl
Add the imp_dl repository path to the ~/.bash_profile and run:  
```
source ~/.bash_profile
```

## Benchmark
### Simulated data benchmark
Currently using the PDB benchmark from AFUnmasked and SAbDab datset for simulated data.  
```
cd ./data/
```
Simulated datset creation occurs in multiple stages:  
1. Metadata collection  
```
python prep_data2.py
```
This downloads the structure for all complexes (.pdb abd .cif), followed by parsing the CIF file to obtain the sequence, residue numbers (seq_id).  
Further it runs JWalk to obtain crosslinks for all complexes.  
It creates the following 2 directories: `{benchmark name}_benchmark/` and `{benchmark name}_metadata/`.  

2. Creating input files for modeling

Run modeling for 100 epochs for the benchmark PDB IDs and select those for which the iitial OpenFold prediction does not satisfy the data.  
Run the follwing script to get the selected benchmark PDBs:  
```
python create_benchmark2.py
```
For all complexes selected in step 1, it creates a directory within `{benchmark name}_benchmark/` containing the structure file (.cif mostly), data file (.csv filr for crosslinks), and a JSON dict containing configs for modeling.  

**Note:** Specify the benchmark name and the dataset configs for step 1 and 2 in the constructors of the respective scripts.  

3. Assessing data satisfaction for initial structure  
```
cd ../
python eye_drop.py
```
This script runs the modeling to obtain the initial predicted structures and data satisfaction at epoch 0.  


### Real data benchmark
Experimental cross-links along with the AF2-multimer predicted structures taken from the [Integrative_docking_benchmark] (https://github.com/isblab/Integrative_docking_benchmark.git) repo.  
To create input file for modeling, run the following script:
```
cd ./data/
python prep_oreilly_complexes.py
```
This script will create the input files and the directory structure as for the simulated benchmark. Once completed run the following script to get initial data satisfcation,  
```
cd ../
python eye_drop.py
```


## Modeling
To run the modeling, make the required changes to the parameters specified in `topology.py` file.  
For now, change the system name at the bottom of the `openfold_wrapper.py` script.  
Run the following script to start the modeling,
```
python openfold_wrapper.py
```
