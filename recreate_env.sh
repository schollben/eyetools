#!/bin/bash

ENV_NAME="eyetools"

echo "Removing existing Conda environment (if any)..."
conda deactivate
conda env remove -n $ENV_NAME --yes

echo "Creating new environment from env.yml..."
conda env create -f env.yml

echo "Done. To activate it now, run:"
echo "    conda activate $ENV_NAME"