import os
import threading
import queue

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

        os.makedirs("out/real", exist_ok=True)
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self.save_worker, daemon=True)
        self.save_thread.start()

    def on_images(self, color_img_msg, depth_img_msg):
        try:
            raw_color = self.bridge.imgmsg_to_cv2(color_img_msg, "passthrough")
            raw_depth = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except cv_bridge.CvBridgeError:
            return
        pose = self.get_robot_pose()
        if pose is not None and self.has_moved(pose):
            str_pose = "x{:.3f}_y{:.3f}_z{:.3f}_r{:.0f}".format(*pose)
            self.save_queue.put(("out/real/" + str_pose + "_rgb.jpeg", raw_color[:, :, ::-1]))
            depth = raw_depth.astype(numpy.float32) / 1000
            depth = numpy.clip(depth, MIN_DEPTH, MAX_DEPTH)
            depth = (depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            converted = (depth * 255).astype(numpy.uint8)
            self.save_queue.put(("out/real/" + str_pose + "_depth.jpeg", converted))
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
        return numpy.sqrt(numpy.sum((new_pose[:3] - self.prv_pose[:3])**2)) >= 0.2 \
                or abs(d_th) >= 10

    def run(self):
        rospy.spin()
        self.save_queue.join()

    def save_worker(self):
        while True:
            filename, image = self.save_queue.get()
            print("Saving '{}' (queue_size: {})".format(filename, queue.qsize()))
            cv2.imwrite(filename, image)
            self.save_queue.task_done()


if __name__ == "__main__":
    node = ROSBagExporter()
    node.run()
