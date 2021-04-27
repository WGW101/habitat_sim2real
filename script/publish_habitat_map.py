import numpy

import rospy
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid

import habitat
from habitat_sim2real import BaseSimulatorViewer


class MapPublisher(BaseSimulatorViewer):
    def __init__(self, sim_cfg):
        sim_cfg.defrost()
        sim_cfg.AGENT_0.RADIUS=0
        sim_cfg.freeze()
        super().__init__(sim_cfg, "MapPublisher", draw_origin=True)

        rospy.init_node("habitat_map_publisher")
        occ_grid_msg = OccupancyGrid()
        occ_grid_msg.header.frame_id = "map"
        occ_grid_msg.info.map_load_time = rospy.Time.now()
        occ_grid_msg.info.resolution = self.map_resolution.mean()
        occ_grid_msg.info.width = self.map_size[0]
        occ_grid_msg.info.height = self.map_size[1]
        occ_grid_msg.info.origin.position.x = self.map_origin[1]
        occ_grid_msg.info.origin.position.y = self.map_origin[0]

        map_pix_to_occ_prob = numpy.array([-1, 0, 100], dtype=numpy.int8)
        occ_grid_msg.data = map_pix_to_occ_prob[self.raw_map.T].flatten().tolist()
        pub = rospy.Publisher("map", OccupancyGrid, queue_size=1, latch=True)
        pub.publish(occ_grid_msg)

        self.publish_static_tf()

        self.tf_broadcast = TransformBroadcaster()
        self.publish_tf()

    def publish_static_tf(self):
        static_tf_broadcast = StaticTransformBroadcaster()

        map_tf = TransformStamped()
        map_tf.header.stamp = rospy.Time.now()
        map_tf.header.frame_id = "map"
        map_tf.child_frame_id = "habitat_map"
        map_tf.transform.rotation.x = 0.5
        map_tf.transform.rotation.y = 0.5
        map_tf.transform.rotation.z = 0.5
        map_tf.transform.rotation.w = 0.5

        base_tf = TransformStamped()
        base_tf.header.stamp = rospy.Time.now()
        base_tf.header.frame_id = "habitat_base_footprint"
        base_tf.child_frame_id = "base_footprint"
        base_tf.transform.rotation.x = -0.5
        base_tf.transform.rotation.y = 0.5
        base_tf.transform.rotation.z = 0.5
        base_tf.transform.rotation.w = 0.5

        static_tf_broadcast.sendTransform([map_tf, base_tf])

    def publish_tf(self):
        s = self.sim.get_agent_state()
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "habitat_map"
        t.child_frame_id = "habitat_base_footprint"
        t.transform.translation.x = s.position[0]
        t.transform.translation.z = s.position[2]
        t.transform.rotation.y = s.rotation.y
        t.transform.rotation.w = s.rotation.w
        self.tf_broadcast.sendTransform(t)

    def update(self):
        super().update()
        self.publish_tf()


if __name__ == "__main__":
    cfg = habitat.get_config("configs/locobot_pointnav_citi_sim.yaml").SIMULATOR
    node = MapPublisher(cfg)
    node.run()
