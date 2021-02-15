#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

source /home/gbono/.conda/conda_init.bash
conda activate habitat
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    ../facebookresearch/habitat-lab/habitat_baselines/run.py \
    --exp-config configs/locobot_ddppo_citi_sim.yaml \
    --run-type train
