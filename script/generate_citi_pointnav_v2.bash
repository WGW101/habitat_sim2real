python script/generate_multigoal_dataset.py -c configs/locobot_pointnav_citi_sim.yaml \
    -n 20000 -l 1 --min-dist 1 --max-dist 6 --if-exist override
python script/generate_multigoal_dataset.py -c configs/locobot_pointnav_citi_sim.yaml \
    -n 15000 -l 1 --min-dist 6 --max-dist 9 --if-exist append
python script/generate_multigoal_dataset.py -c configs/locobot_pointnav_citi_sim.yaml \
    -n 15000 -l 1 --min-dist 9 --max-dist 50 --if-exist append
