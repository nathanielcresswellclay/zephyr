
# Atmospheric modeling using data-driven methods. Evolved from jweyn/DELUSioN.

<short blurb here about the project statement, associated lab, authors and important contributors> 

A link to our most recent paper can be found [here](link%20goes%20here) 

## Getting set up
### Package dependencies
1. Python. 
You should be running Python 3.8 <are other versions ok?> 
2. < other dependencies here >.

#### Connecting to our compute
We have several machines and mostly use `rhodium` for our data pipelines, training, and evaluation of models. This machine has 4 GPUs...
<more description of our nodes how to ssh to it, what compute each one has - I think 4 GPU right?>
1. If you haven't already, ask for an account from David Warren <dwarren@uw.edu> to access Rhodium, Mercury and QuickSilver.
2. Test your access by running the following command
	 `ssh <your-username>@<machine_name>.atmos.washington.edu` (or setup ssh key?)
	 Our machines names include: `brass`, `rhodium`, `gold`, `quicksilver`, and `mercury`.  Some of our members also have access to a Navy computer which is used to run training, called `<name>`
3. Include a brief description of where to find key files ie. data and models.

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

