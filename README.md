# tfit-benchmark
Benchmarking tools for coregulation prediction


## Installation 

### 1. Clone the repo

```bash
git clone https://github.com/shubhvjain/tfit-benchmark.git
cd tfit-benchmark
```

### 2. Install  dependencies 

#### Container 

[Docker](https://www.docker.com/) or [Apptainer](https://apptainer.org/) 

#### Python dependencies 


## Setting up the environment

Before running an experiment, it is important to make sure you are using the correct environment because different types of containers are run in different environment. 

This is controlled by the `tfit_mode` variable. We provide 2 modes:
- `local`: use this mode to run on a system where docker is available. This uses the `.env.local` file 
- `hpc`: use this mode to run on a system where apptainer is available. Note that this is specifically configured to run on the [HPC](https://hpc.fau.de/) resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU). 
  - The `.env.hpc` can be easily modified to use on other systems. 

A typical env file looks like:
```bash

```


To load an env file, run:

```bash

```

### Create containers 


## Setup datasets

```bash
make 
```


Ready to run experiments!

## Running an experiment 


### 1. Load env file 

### 2. Generate an experiment file 


### 3. Run Pipeline 

### 4. Generate results 



# Folder structure 

```
tfit-exp/
├── containers/          # Dockerfiles + .def files
├── scripts/             # all Python/R code
├── experiments/         # experiment config JSONs
├── analyses/            # analysis config JSONs
├── outputs/             # committed — result CSVs, status.db per exp/tool/dataset
├── results/             # committed — final figures, tables for paper
├── notebooks/
├── datasets.json
├── Snakefile
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```
