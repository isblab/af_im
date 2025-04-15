# IMP DL


## Installation
### OpenFold
Unzip the openfold.zip file. This has been cloned from the pl_upgrades branch in openfold git repo (for Cuda12) (https://github.com/aqlaboratory/openfold.git).  
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
Run the following script to create the required input files for the benchmark dataset.  
```
python prep_data.py
```
Run modeling for 100 epochs for the benchmark PDB IDs and select those for which the iitial OpenFold prediction does not satisfy the data.  
Run the follwing script to get the selected benchmark PDBs:  
```
python eye_drop.py
```

## Modeling
To run the modeling, make the required changes to the parameters specified in `topology.py` file.  
For now, change the system name at the botton of the `openfold_wrapper.py` script.  
Run the following script to start the modeling,
```
python openfold_wrapper.py
```