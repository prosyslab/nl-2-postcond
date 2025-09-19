#!/bin/bash

PYTHONPATH=$HOME/anaconda3/envs/nl2postcond/bin/python

if [ -z "$PYTHONPATH" ]; then
    echo "Please create a conda environment with the name nl2postcond"
    exit 1
fi

exp_configs=(
    "generateLLMSamplesSimple.yaml"
    "generateLLMSamplesBase.yaml"
    "generateLLMSamplesSimpleRefCode.yaml"
    "generateLLMSamplesBaseRefCode.yaml"
)

for exp_config in ${exp_configs[@]}; do
    $PYTHONPATH llm_sample_generator.py experiment=${exp_config}
done