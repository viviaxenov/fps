# Running on `escher-02` 
Old node, AMD CPUs + 8 Nvidia A100 GPUs.
## Installation
Create conda environment
```bash
conda create -n fps_cuda12 -c conda-forge --override-channels python=3.12
conda activate fps_cuda12
```
Pull the code from GitHub and `cd` to the project directory
```bash
git clone https://github.com/viviaxenov/fps.git
```
Install the code with dependencies
```bash
unset LD_LIBRARY_PATH
pip install ".[cuda12,examples]"
```

## Running
Before running, make sure that GPU resources are available. 
It could be also useful to disable JAX'es default behaviour to immediately allocate a large portion of GPU's memory.

Analyze GPU utilization
```bash
nvidia-smi
```
and choose the id of the least busy GPU.

### Python script
Append the following lines to the code example you wish to run.
Here `0` has to be replaced with the id of the selected GPU
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
Override the default preallocation behavior (choose one):
```python
# disable preallocation behavior
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# alternatively, preallocate only a small fraction 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1" 
```
Finally, run the script
```bash
python test_distributions.py
```

### bash script
Alternatively, edit the values of the environment variables and script name in the executable file `job.sh`.

Before the first run, give the file permission to execute:
```bash
chmod +x job.sh
```
Then
```bash
./job.sh
```
