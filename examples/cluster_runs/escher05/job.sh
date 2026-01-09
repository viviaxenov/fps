#!/bin/bash

# Find conda when using escher-05:
source /opt/intel/oneapi/intelpython/python3.12/etc/profile.d/conda.sh

# Replace $PATH by own environment path
conda activate $PATH

# Run the targeted python script
python test_distributions.py

