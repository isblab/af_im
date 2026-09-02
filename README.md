
# AlphaFold-based Integrative Modeling
We evaluate existing AlphaFold-based, including AlphaLink2, Boltz2, and GRASP for their applicability towards integrative modeling.

![main_fig]()

## Publication and Data
* Kartik Majila, Shruthi Viswanath. **Evaluation of methods for AlphaFold-based integrative modeling.** (2026) [bioRxiv]().
* Data is deposited in [Zenodo]()

## Installation 

### Dependencies

* See `environment.yml` for the dependencies.  
* Ensure Mamba is installed on the system.
```
mamba env create -n af_im -f environment.yml
```

### OpenFold 
We used the `pl_upgrades` branch in [openfold git repo (for Cuda12)](https://github.com/aqlaboratory/openfold.git). As in `openfold` directory. 

Download the databases for AlphaFold/OpenFold. See instructions for downloading the databases on the OpenFold repository.

Add the path to the `af_im` repository to bash_profile and run,

```
source ~/.bash_profile
```

Update the following paths in the config.py file: `db_dir, tool_base, openfold_params_dir`  

### JWalk
Clone the git repo from [here](https://github.com/Topf-Lab/Jwalk.git).  
We converted the Python2 implementation to Python3 using python2to3.com server.  
For installation, run  
```
python setup.py install
```

### AlphaLink2
Clone and install AlphaLink2 as described [here](https://github.com/Rappsilber-Laboratory/AlphaLink2.git).

### Boltz2
Clone and install Boltz2 as described [here](https://github.com/jwohlwend/boltz.git).

### GRASP
Clone and install GRASP as described [here](https://github.com/aqlaboratory/openfold.git).

Post installation, activate the respective models environment and install the following packages: 

```
pip install pandas, typing-extensions, ml_collections
```


## Benchmark creation
### Multimeric benchmark
```
mkdir benchmark
cd ./data/
```
Simulated datset creation occurs in multiple stages:  
1. Metadata collection  
```
python prep_data2.py
```
This downloads the required metadata for all complexes (.pdb abd .cif), followed by parsing the CIF file to obtain the sequence, residue numbers (seq_id).  
Further it runs JWalk to obtain crosslinks for all complexes.  
It creates the following 2 directories: `{benchmark name}_benchmark/` and `{benchmark name}_metadata/`.  

2. Creating input files for modeling
```
python create_benchmark2.py
```
For all complexes selected in step 1, it creates a directory within `{benchmark name}_benchmark/` containing the structure file (.cif or .pdb), data file (.csv file for crosslinks), and a JSON dict containing configs for modeling.  


3. Creating MSAs
```
python create_msas.py -b BENCHMARK_NAME -p
```
`BENCHMRK_NAME` could be either of crosslink or multistate. Use `-p` for multimers.  
This script runs the OpenFold MSA creation pipeline for obtaining the MSA required for structure prediction.  


### Multi-state proteins
For obtaining the input files for the multi-state proteins run the following script:
```
cd ../
python multi_state2.py
```
This script will create the input files for the multi-state proteins in the same format as for the crosslink benchmark. The directory structure is the same as above.  


## Predictions
Activate the respective models environment and then run the following command for obtaining predictions,
```
python get_competing_method_preds.py -m MODEL -c CONFIG_NAME -b BENCHMARK_NAME -d DEVICE
```

|  Flags |                                     Description                                                                          |
| ------ | ------------------------------------------------------------------------------------------------------------------------ |
|  -m    | Model to use for prediction: alphalink2/boltz2/grasp                                                                     |
|  -c    | Configs to use for obtaining predictions from a model. See model_configs.py for the available configs                    |
|  -b    | Name of the benchmark to obtain predictions for: crosslink/multistate                                                    |
|  -d    | Device to run predictions on: cpu/cuda:0/cuda:1                                                                          |

For the results shown in the paper we used the following configs: alpha, beta1, beta3, kappa1, kappa2, kappa3.


## Analysis
For the crosslink benchmark use the following command to run the analysis,
```
python analysis.py -m MODEL -b crosslink
```
For multistate benchmark run the following command,
```
python analysis.py -m MODEL -b multistate -ms
```

`MODEL` is the same as defined above.  
This script runs the analysis for all specified configs for a given model in the `model_configs.py`.

To obtain the plots shown in the paper, run the following command,
```
python paper_figures.py
```


## Information
__Author(s):__ Kartik Majila, Shruthi Viswanath

__Date__: MM DD, 2026

__License:__ GPL v3
This work is licensed under the terms of the GNU General Public License,
 Version 3, as published by the Free Software Foundation on 29 June 2007.

__Testable:__ Yes

__Parallelizeable:__ Yes

__Publications:__  Majila K., Viswanath S. Evaluation of methods for AlphaFold-based integrative modeling. bioRxiv  (2026), [DOI]().

