curl "http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip" -o "mp3d_habitat.zip" \
	&& unzip "mp3d_habitat.zip" -d "data/scene_datasets/"

curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip" -o "pointnav_mp3d_v1.zip" \
	&& unzip "pointnav_mp3d_v1.zip" -d "data/datasets/pointnav/mp3d/v1"
curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip" -o "objectnav_mp3d_v1.zip" \
	&& unzip "objectnav_mp3d_v1" -d "data/datasets/objectnav/mp3d/v1"
curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/eqa/mp3d/v1/eqa_mp3d_v1.zip" -o "eqa_mp3d_v1.zip" \
	&& unzip "eqa_mp3d_v1.zip" -d "data/datasets/eqa/mp3d/v1"
curl "https://dl.fbaipublicfiles.com/habitat/data/datasets/vln/mp3d/r2r/v1/vln_r2r_mp3d_v1.zip" -o "vln_r2r_mp3d_v1.zip" \
	&& unzip "vln_r2r_mp3d_v1.zip" -d "data/datasets/vln/mp3d/r2r/v1"
