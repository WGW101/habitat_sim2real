import os

import rospy
from sensor_msgs.msg import Image
import message_filters
import cv_bridge
import cv2
import tf2_ros
import tf_conversions
import numpy

MIN_DEPTH = 0.155
MAX_DEPTH = 10.0
MIN_MOVE_DIST = 0.2
MIN_MOVE_ANG = 10
OUT_DIR = "out/traj_cap/real"



class ROSBagExporter:
    def __init__(self):
        rospy.init_node("rosbag_stamped_image_extractor")
        self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        self.img_sync = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 10)
        self.img_sync.registerCallback(self.on_images)
        self.bridge = cv_bridge.CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listerner = tf2_ros.TransformListener(self.tf_buffer)
        self.prv_pose = None

        os.makedirs(OUT_DIR, exist_ok=True)
        self.list_f = open(os.path.join(OUT_DIR, "chrono_list.txt"), 'w')

    def on_images(self, color_img_msg, depth_img_msg):
        try:
            raw_color = self.bridge.imgmsg_to_cv2(color_img_msg, "passthrough")
            raw_depth = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except cv_bridge.CvBridgeError:
            return
        pose = self.get_robot_pose()
        if pose is not None and self.has_moved(pose):
            str_pose = "x{:.3f}_y{:.3f}_z{:.3f}_r{:.0f}".format(*pose)
            depth = raw_depth.astype(numpy.float32) / 1000
            depth = numpy.clip(depth, MIN_DEPTH, MAX_DEPTH)
            depth = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            converted = (depth * 255).astype(numpy.uint8)
            self.save(str_pose, raw_color[:, :, ::-1], converted)
            self.prv_pose = pose

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", rospy.Time(0))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = trans.transform.translation
        q = trans.transform.rotation
        _, _, th = tf_conversions.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
        return numpy.array([t.x, t.y, t.z, numpy.degrees(th)])

    def has_moved(self, new_pose):
        if self.prv_pose is None:
            return True
        d_th = new_pose[3] - self.prv_pose[3]
        if d_th < -180:
            d_th += 180
        if d_th > 180:
            d_th -= 180
        return numpy.sqrt(numpy.sum((new_pose[:3] - self.prv_pose[:3])**2)) >= MIN_MOVE_DIST \
                or abs(d_th) >= MIN_MOVE_ANG

    def run(self):
        rospy.spin()
        self.list_f.close()

    def save(self, filebase, rgb, depth):
        print("Saving '{}'".format(filebase))
        path = os.path.join(OUT_DIR, filebase + "_rgb.jpeg")
        self.list_f.write(path + '\n')
        cv2.imwrite(path, rgb)
        path = os.path.join(OUT_DIR, filebase + "_depth.jpeg")
        cv2.imwrite(path, depth)


if __name__ == "__main__":
    node = ROSBagExporter()
    node.run()
