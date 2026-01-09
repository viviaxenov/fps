#!/bin/bash

# Configure GPU and memory utilization
# by setting environment variables
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
# Run the targeted python script
python ../../BayesNN.py

