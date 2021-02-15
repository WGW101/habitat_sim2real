#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

source /home/gbono/.conda/conda_init.bash
conda activate habitat
python ../facebookresearch/habitat-lab/habitat_baselines/run.py \
	--exp-config ./configs/locobot_ppo_citi_sim.yaml \
	--run-type train
