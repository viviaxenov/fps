# Running on `escher-05` 
New node, Intel CPUs + 4 Nvidia H200 GPUs. 
Launching the code via the `HTCondor` queue system.
## Installation
Installation has to be done in one of the other nodes.
For me, installation from `gate` crashes (presumably due to the lack of memort), so I had to do from `escher-02`.


Create conda environment
```bash
conda create -n fps_cuda13 -c conda-forge --override-channels python=3.12
```
Optionally: create an environment with Intel Python distribution ([details here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-python-download.html?install-type=conda&python-conda=python-3_12&operatingsystem-conda=linux&packagetype-conda=idp-allcomponents))
```bash
conda create -n fps_cuda13 intelpython3_full python=3.12 -c https://software.repos.intel.com/python/conda -c conda-forge --override-channels
```
Activate the environment:
```bash
conda activate fps_cuda13
```
Pull the code from GitHub and `cd` to the project directory
```bash
git clone https://github.com/viviaxenov/fps.git
cd ./fps
```
Install the code with dependencies
```bash
unset LD_LIBRARY_PATH
pip install ".[cuda13,examples]"
```
## Logging in
The code is run on `escher-05`, but submitting to the queue is done from `bernoulli`.
Additional info [here](https://lab.wias-berlin.de/prieto/htcondor_wias_manual/-/tree/main?ref_type=heads)
```bash
ssh USERNAME@gate.wias-berlin.de
ssh bernoulli.wias-berlin.de
```

## Configuring the executable and the launch file
In this case, the GPU selection and memory allocation is handled by the queue system.

First, one has to update the slot name in `launch.sub`. 
On `bernoulli`, run 

```bash
condor_status
```
and choose the appropriate slot name.
Replace the placeholder `$SLOT_NAME` in `launch.sub` with this name.

We then need to let the job know how to activate the `conda` environment.
```bash
conda env list | grep fps_cuda13
```
In the right column of the output you find the path to the environment.
Go to the `job.sh` file and replace the path in the following line
```bash
# Replace $PATH by own environment path
conda activate $PATH
```

## Submitting a job

Make sure that in `launch.sub` you have the correct executable name
```
executable = job.sh
```

Then submit the job with HTCondor
```bash
condor_submit launch.sub
```
