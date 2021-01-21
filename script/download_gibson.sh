curl "https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip" -o "gibson_habitat_full.zip" \
	&& unzip "gibson_habitat_full.zip" -d "data/scene_datasets/"
curl "https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip" -o "gibson_habitat_challenge" \
	&& unzip "gibson_habitat_challenge.zip" -d "data/scene_datasets/"

curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip" -o "pointnav_gibson_v1.zip" \
	&& unzip "pointnav_gibson_v1.zip" -d "data/datasets/pointnav/gibson/v1/"
curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip" -o "pointnav_gibson_v2.zip" \
	&& unzip "pointnav_gibson_v2.zip" -d "data/datasets/pointnav/gibson/v2/"
