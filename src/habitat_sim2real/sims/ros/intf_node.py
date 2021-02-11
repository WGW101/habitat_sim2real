import threading
import math
import numpy

import rospy
import message_filters
import cv_bridge
import tf2_ros
import tf_conversions
import actionlib

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from dynamixel_workbench_msgs.msg import DynamixelStateList
from dynamixel_workbench_msgs.srv import DynamixelCommand
from kobuki_msgs.msg import BumperEvent


class HabitatInterfaceROSNode:
    def __init__(self, cfg):
        self.cfg = cfg
        timeout = rospy.Duration(cfg.CONNECTION_TIMEOUT)

        try:
            rospy.get_published_topics() # Raises ConnectionRefusedError if master is offline
        except ConnectionRefusedError:
            raise RuntimeError("Unable to connect to ROS master.")
        rospy.init_node(cfg.NODE_NAME)

        self.color_sub = message_filters.Subscriber(cfg.COLOR_IMAGE_TOPIC, Image)
        self.depth_sub = message_filters.Subscriber(cfg.DEPTH_IMAGE_TOPIC, Image)
        self.img_sync = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub],
                                                         cfg.IMAGE_SYNC_QUEUE_SIZE)
        self.img_sync.registerCallback(self.on_img)
        self.bridge = cv_bridge.CvBridge()
        self.raw_images_buffer = None
        self.img_buffer_lock = threading.Lock()
        self.has_first_images = threading.Event()

        self.map_sub = rospy.Subscriber(cfg.MAP_TOPIC, OccupancyGrid, self.on_map)
        self.map_buffer = None
        self.map_buffer_lock = threading.Lock()
        self.has_first_map = threading.Event()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listerner = tf2_ros.TransformListener(self.tf_buffer)

        self.move_base_client = actionlib.SimpleActionClient(cfg.MOVE_BASE_ACTION_SERVER,
                                                             MoveBaseAction)
        if not self.move_base_client.wait_for_server(timeout):
            raise RuntimeError("Unable to connect to move_base action server.")

        try:
            rospy.wait_for_service(cfg.DYNAMIXEL_SERVICE, timeout)
        except rospy.ROSException:
            raise RuntimeError("Unable to connect to dynamixel service.")
        self.dynamixel_cmd_proxy = rospy.ServiceProxy(cfg.DYNAMIXEL_SERVICE, DynamixelCommand)
        self.dynamixel_sub = rospy.Subscriber(cfg.DYNAMIXEL_STATE_TOPIC,
                                              DynamixelStateList,
                                              self.on_dynamixel_state)
        self.tilt_target_value = None
        self.tilt_target_event = threading.Event()
        self.tilt_reached_event = threading.Event()

        try:
            rospy.wait_for_service(cfg.MOVE_BASE_PLAN_SERVICE, timeout)
        except rospy.ROSException:
            raise RuntimeError("Unable to connect to get_plan service.")
        self.get_plan_proxy = rospy.ServiceProxy(cfg.MOVE_BASE_PLAN_SERVICE, GetPlan)

        self.episode_goal_pub = rospy.Publisher("habitat_episode_goal", PoseStamped,
                                                queue_size=1)

        self.bump_sub = rospy.Subscriber(cfg.BUMPER_TOPIC, BumperEvent, self.on_bump)
        self.collided_lock = threading.Lock()
        self.collided = False

    def on_img(self, color_img_msg, depth_img_msg):
        try:
            raw_color = self.bridge.imgmsg_to_cv2(color_img_msg, "passthrough")
            raw_depth = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except cv_bridge.CvBridgeError:
            return
        with self.img_buffer_lock:
            self.raw_images_buffer = (raw_color, raw_depth)
        self.has_first_images.set()

    def get_raw_images(self):
        self.has_first_images.wait()
        with self.img_buffer_lock:
            return self.raw_images_buffer

    def on_map(self, occ_grid_msg):
        grid = numpy.array(occ_grid_msg.data).reshape(occ_grid_msg.info.height,
                                                      occ_grid_msg.info.width)
        cell_size = occ_grid_msg.info.resolution
        origin_pos = (occ_grid_msg.info.origin.position.x,
                      occ_grid_msg.info.origin.position.y,
                      occ_grid_msg.info.origin.position.z)
        origin_rot = (occ_grid_msg.info.origin.orientation.x,
                      occ_grid_msg.info.origin.orientation.y,
                      occ_grid_msg.info.origin.orientation.y,
                      occ_grid_msg.info.origin.orientation.w)
        with self.map_buffer_lock:
            self.map_buffer = (grid, cell_size, origin_pos, origin_rot)
        self.has_first_map.set()

    def get_map(self):
        self.has_first_map.wait()
        with self.map_buffer_lock:
            return self.map_buffer

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.cfg.TF_REF_FRAME,
                                                    self.cfg.TF_ROBOT_FRAME,
                                                    rospy.Time(0))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        p = (trans.transform.translation.x,
             trans.transform.translation.y,
             trans.transform.translation.z)
        q = (trans.transform.rotation.x, trans.transform.rotation.y,
             trans.transform.rotation.z, trans.transform.rotation.w)
        return p, q

    def get_distance(self, source, target):
        start = PoseStamped()
        start.header.stamp = rospy.Time.now()
        start.header.frame_id = self.cfg.TF_REF_FRAME
        start.pose.position.x = source[0]
        start.pose.position.y = source[1]
        start.pose.position.z = source[2]
        start.pose.orientation.w = 1.0

        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.cfg.TF_REF_FRAME
        goal.pose.position.x = target[0]
        goal.pose.position.y = target[1]
        goal.pose.position.z = target[2]
        goal.pose.orientation.w = 1.0

        res = self.get_plan_proxy(start, goal, self.cfg.MOVE_BASE_PLAN_TOL)
        if res.plan.poses:
            dist = 0
            prv_x = res.plan.poses[0].pose.position.x
            prv_y = res.plan.poses[0].pose.position.y
            for pose in res.plan.poses[1:]:
                x = pose.pose.position.x
                y = pose.pose.position.y
                dist += math.sqrt((x - prv_x)**2 + (y - prv_y)**2)
                prv_x, prv_y = x, y
            return dist
        else:
            return None

    def set_camera_tilt(self, tilt):
        self.tilt_reached_event.clear()
        self.tilt_target_value = int(2048 + 638 * tilt)
        res = self.dynamixel_cmd_proxy("", self.cfg.DYNAMIXEL_TILT_ID,
                                       "Goal_Position", self.tilt_target_value)
        if res.comm_result:
            self.tilt_target_event.set()
            if self.tilt_reached_event.wait(timeout=self.cfg.DYNAMIXEL_TIMEOUT):
                self.tilt_target_event.clear()
                return True
        return False

    def on_dynamixel_state(self, dynamixel_msg):
        self.tilt_target_event.wait()
        for state in dynamixel_msg.dynamixel_state:
            if state.id == self.cfg.DYNAMIXEL_TILT_ID:
                err = abs(state.present_position - self.tilt_target_value)
                if err <= self.cfg.DYNAMIXEL_TILT_TOL:
                    self.tilt_reached_event.set()
                    break

    def move_to_relative(self, x, y, theta):
        return self._move_to(x, y, theta, self.cfg.TF_ROBOT_FRAME)

    def move_to_absolute(self, x, y, theta):
        return self._move_to(x, y, theta, self.cfg.TF_REF_FRAME)

    def _move_to(self, x, y, theta, frame_id):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = frame_id
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, theta)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        self.move_base_client.send_goal(goal)
        return self.move_base_client.wait_for_result()

    def publish_episode_goal(self, x, y, z):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.cfg.TF_REF_FRAME
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        self.episode_goal_pub.publish(pose)

    def on_bump(self, bump_msg):
        if bump_msg.state == BumperEvent.PRESSED:
            self.move_base_client.cancel_goal()
#            if bump_msg.bumper == BumperEvent.LEFT:
#                pass
#            elif bump_msg.bumper == BumperEvent.CENTER:
#                pass
#            elif bump_msg.bumper == BumperEvent.RIGHT:
#                pass
            with self.collided_lock:
                self.collided = True

    def has_collided(self):
        with self.collided_lock:
            return self.collided

    def clear_collided(self):
        with self.collided_lock:
            self.collided = False
