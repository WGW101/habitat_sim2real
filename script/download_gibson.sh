curl "https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip" -o "gibson_habitat_trainval.zip" \
	&& mkdir -p "data/scene_datasets/" \
	&& unzip -o "gibson_habitat_full.zip" -d "data/scene_datasets/"
curl "https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat.zip" -o "gibson_habitat.zip" \
    && mkdir -p "data/scene_datasets/" \
	&& unzip -o "gibson_habitat_challenge.zip" -d "data/scene_datasets/"

curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip" -o "pointnav_gibson_v1.zip" \
    && mkdir -p "data/datasets/pointnav/gibson/v1" \
	&& unzip -o "pointnav_gibson_v1.zip" -d "data/datasets/pointnav/gibson/v1/"
curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip" -o "pointnav_gibson_v2.zip" \
    && mkdir -p "data/datasets/pointnav/gibson/v2" \
	&& unzip -o "pointnav_gibson_v2.zip" -d "data/datasets/pointnav/gibson/v2/"
