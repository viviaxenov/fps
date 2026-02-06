#!/bin/bash

# Configure GPU and memory utilization
# by setting environment variables
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=7
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export SCIPY_ARRAY_API=1
# Run the targeted python script
python /Home/optimier/berkowsky/Documents/fps/examples/datasets/10D_rosenbrock/SVGD_StepS_Search.py

