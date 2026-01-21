#!/bin/bash

# Configure GPU and memory utilization
# by setting environment variables
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
# Run the targeted python script
python ../../test_distributions.py

