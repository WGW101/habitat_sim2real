#!/bin/bash

source /home/gbono/.conda/conda_init.bash
conda activate habitat
python ../habitat-lab/habitat_baselines/run.py \
	--run-type train \
	--exp-config ./configs/locobot_ppo_citi_sim.yaml
