#!/bin/bash

# Configure GPU and memory utilization
# by setting environment variables
unset LD_LIBRARY_PATH
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=5
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
# Run the targeted python script
python ./gridsearch.py
