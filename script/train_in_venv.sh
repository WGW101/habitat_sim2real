#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

source /home/gbono/facebookresearch/habitat-lab/.env/bin/activate
python script/run_baseline.py -c "configs/locobot_ppo_citi_sim.yaml" -r "train"
