#!/bin/bash

singularity exec --nv /home/gbono/container_images/habitat_v016_cuda.sif \
	python3 script/run_baseline.py -c "configs/locobot_ppo_citi_sim.yaml" -r "train"
