
# Deep Learning Earth System Modelling 

This repository uses publically available datasets to create models of the Earth system. Our focus is on modular and customizable models that allow rapid development and experimentation. Compared to other popular models, ours are small. They can train in a few days on small ML clusters. 

## Getting set up

### Connecting to our compute
Those in active collaboration with our group at the University of Washington are welcome to use our machines for analysis and training. If you do not plan to use our resources but would like to play around with our code, move onto instructions for setting up the virtual environment. We have 5 nodes dedicated to developing and testing deep learning earth system models.

| Node Name  | CPU | Logical Cores | RAM | Swap | GPU | Intended Use |
|------------|-----|---------------|-----|------|-----|--------------|
| quicksilver| AMD EPYC 7742 64-Core Processor | 128 | 503Gi | 238Gi | 4x Nvidia A100 80Gi | training
| mercury    | Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz | 64 | 251Gi | 476Gi | 4x Nvidia A100 80Gi, 2x Nvidia Tesla V100 32Gi | training |
| gold       | Intel(R) Xeon(R) W-2123 CPU @ 3.60GHz | 8 | 251Gi | 59Gi | 2x Nvidia Titan RTX 80Gi | training, forecasting |
| rhodium    | Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz | 32 | 376Gi | 355Gi | None | analysis, data processing |
| brass      | Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz | 64 | 187Gi | 178Gi | None | analysis, data processing


1. Once you have an account, you can access our nodes via ssh. The command can be structured as follows: 
```bash
ssh <your-username>@<name-of-node>.atmos.washington.edu
```

### Creating the Virtual Environment
1. Download an intialize Anaconda. Our code has been developed using [conda 4.13.0](https://anaconda.org/anaconda/conda/files?version=4.13.0)
2. Clone **zephyr**. The repository includes a requiremnents file that enumerates all necessary packages. Checkout dlesm, the actively developed branch of zephyr.
```bash
git checkout dlesm
```
3. Create conda environment. Once conda has been initialized, make zephyr/ your working directory and use the following command to create the proper conda environment:  
```bash
conda env create -f environments/zephyr-1.1.yml
```

#### Running Training
1. ssh to our virtual machine via the above.
2. `cd <path/to/training/script> `
3. run_training.sh
	4. explain the params needed to pass in, overrides etc here

#### Running Evaluation 
1. fill out similar steps 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Folder structure
< call out any important files and/or folders inside each, i.e. the most up to date model >
* data_processing: 
	* remap:
	* utils:
* environments:
* evaluation: 
	* configs:
* scripts:
* training: 
	* configs:
	* dlwp:
		* data: contains `data_loading.py` script for loading data and coupling ocean and atmosphere data `couplers.py`.
		* model: contains `models` and `modules`
		* trainer: contains optimizer


#### Dataset

Our data pipelines mostly live on Rhodium at </ insert/ path /here >
See the <a href="https://github.com/nathanielcresswellclay/zephyr/tree/main/data_processing"> data processing</a> folder for an overview of our data scripts. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Models

See the <a href="https://github.com/nathanielcresswellclay/zephyr/tree/dlesm/training/dlwp/model/models">models</a> folder for an overview of our models.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact
Have questions? Contact us at <preferred method of contact, probably your email?>

