import threading
import numpy
import logging

import rospy
import message_filters
import cv_bridge
import tf2_ros
import tf2_geometry_msgs
import tf_conversions
import actionlib

from geometry_msgs.msg import PoseStamped, TransformStamped
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
            rospy.get_published_topics()
        except ConnectionRefusedError:
            raise RuntimeError("Unable to connect to ROS master.")
        rospy.init_node(cfg.NODE_NAME)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listerner = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_timeout = rospy.Duration(self.cfg.TF_TIMEOUT)

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
        self.map_resolution = None
        self.map_origin_transform = None
        self.map_grid = None
        self.map_free_points = None
        self.map_lock = threading.Lock()
        self.has_first_map = threading.Event()

        self.bump_sub = rospy.Subscriber(cfg.BUMPER_TOPIC, BumperEvent, self.on_bump)
        self.collided_lock = threading.Lock()
        self.collided = False

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

        self.rng = numpy.random.default_rng()

    def on_img(self, color_img_msg, depth_img_msg):
        try:
            raw_color = self.bridge.imgmsg_to_cv2(color_img_msg, "passthrough")
            raw_depth = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except cv_bridge.CvBridgeError as e:
            logging.warn(e)
            return
        with self.img_buffer_lock:
            self.raw_images_buffer = (raw_color, raw_depth)
        self.has_first_images.set()

    def get_raw_images(self):
        if not self.has_first_images.wait(self.cfg.GETTER_TIMEOUT):
            raise RuntimeError("Timed out waiting for raw image.")
        with self.img_buffer_lock:
            return self.raw_images_buffer

    def on_map(self, occ_grid_msg):
        try:
            origin = PoseStamped(header=occ_grid_msg.header, pose=occ_grid_msg.info.origin)
            origin = self.tf_buffer.transform(origin,
                                              self.cfg.TF_HABITAT_REF_FRAME,
                                              self.tf_timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            logging.warn(e)
            return
        transform = TransformStamped()
        transform.header.frame_id = self.cfg.TF_HABITAT_REF_FRAME
        transform.child_frame_id = "map_origin"
        transform.transform.translation.x = origin.pose.position.x
        transform.transform.translation.y = origin.pose.position.y
        transform.transform.translation.z = origin.pose.position.z
        transform.transform.rotation.x = origin.pose.orientation.x
        transform.transform.rotation.y = origin.pose.orientation.y
        transform.transform.rotation.z = origin.pose.orientation.z
        transform.transform.rotation.w = origin.pose.orientation.w

        grid = numpy.array(occ_grid_msg.data).reshape(occ_grid_msg.info.height,
                                                      occ_grid_msg.info.width)
        free_points = numpy.stack(numpy.nonzero((grid < self.cfg.MAP_FREE_THRESH)
                                                & (grid > -1)), -1)

        with self.map_lock:
            self.map_resolution = occ_grid_msg.info.resolution
            self.map_origin_transform = transform
            self.map_grid = grid
            self.map_free_points = free_points
        self.has_first_map.set()

    def get_map_grid(self):
        if not self.has_first_map.wait(self.cfg.GETTER_TIMEOUT):
            raise RuntimeError("Timed out waiting for map.")
        with self.map_lock:
            return self.map_grid

    def sample_free_point(self):
        if not self.has_first_map.wait(self.cfg.GETTER_TIMEOUT):
            raise RuntimeError("Timed out waiting for map.")
        with self.map_lock:
            pt = self.rng.choice(self.map_free_points) * self.map_resolution
            pose = PoseStamped()
            pose.pose.position.x = pt[1]
            pose.pose.position.y = pt[0]
            pose.pose.orientation.w = 1
            pose = tf2_geometry_msgs.do_transform_pose(pose, self.map_origin_transform)
        return [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]

    def on_bump(self, bump_msg):
        if bump_msg.state == BumperEvent.PRESSED:
            self.move_base_client.cancel_goal()
            with self.collided_lock:
                self.collided = True

    def has_collided(self):
        with self.collided_lock:
            return self.collided

    def clear_collided(self):
        with self.collided_lock:
            self.collided = False

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(self.cfg.TF_HABITAT_REF_FRAME,
                                                    self.cfg.TF_HABITAT_ROBOT_FRAME,
                                                    rospy.Time(0), self.tf_timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            logging.warn(e)
            return None, None
        p = (trans.transform.translation.x,
             trans.transform.translation.y,
             trans.transform.translation.z)
        q = (trans.transform.rotation.x, trans.transform.rotation.y,
             trans.transform.rotation.z, trans.transform.rotation.w)
        return p, q

    def _make_pose_stamped(self, pos, rot=None):
        if rot is None:
            rot = (0, 0, 0, 1)
        # we get pose of hab_robot_frame in hab_ref_frame

        # we build pose of robot_frame in hab_ref_frame
        try:
            tf = self.tf_buffer.lookup_transform(self.cfg.TF_HABITAT_ROBOT_FRAME,
                                                 self.cfg.TF_ROBOT_FRAME,
                                                 rospy.Time(0), self.tf_timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            logging.warn(e)
            return None
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.cfg.TF_HABITAT_REF_FRAME
        pose.pose.position.x = pos[0] + tf.transform.translation.x
        pose.pose.position.y = pos[1] + tf.transform.translation.y
        pose.pose.position.z = pos[2] + tf.transform.translation.z

        tf_q = (tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w)
        rot = tf_conversions.transformations.quaternion_multiply(rot, tf_q)
        pose.pose.orientation.x = rot[0]
        pose.pose.orientation.y = rot[1]
        pose.pose.orientation.z = rot[2]
        pose.pose.orientation.w = rot[3]

        # we transform it to ref_frame
        try:
            pose = self.tf_buffer.transform(pose, self.cfg.TF_REF_FRAME, self.tf_timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            logging.warn(e)
            return None
        return pose

    def get_distance(self, src, dst):
        start = self._make_pose_stamped(src)
        if start is None:
            return np.inf
        goal = self._make_pose_stamped(dst)
        if goal is None:
            return np.inf

        res = self.get_plan_proxy(start, goal, self.cfg.MOVE_BASE_PLAN_TOL)
        if res.plan.poses:
            dist = 0
            prv_x = res.plan.poses[0].pose.position.x
            prv_y = res.plan.poses[0].pose.position.y
            for pose in res.plan.poses[1:]:
                x = pose.pose.position.x
                y = pose.pose.position.y
                dist += numpy.sqrt((x - prv_x)**2 + (y - prv_y)**2)
                prv_x, prv_y = x, y
            return dist
        else:
            return numpy.inf

    def set_camera_tilt(self, tilt):
        self.tilt_reached_event.clear()
        self.tilt_target_value = int(self.cfg.DYNAMIXEL_TICK_OFFSET
                                     + self.cfg.DYNAMIXEL_TICK_PER_RAD * tilt)
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

    def move_to_relative(self, forward=0, turn=0):
        if forward == 0 and turn == 0:
            return True
        goal = PoseStamped()
        goal.header.stamp = rospy.Time(0)
        goal.header.frame_id = self.cfg.TF_ROBOT_FRAME
        goal.pose.position.x = forward
        goal.pose.orientation.z = numpy.sin(0.5 * turn)
        goal.pose.orientation.w = numpy.cos(0.5 * turn)
        try:
            goal = self.tf_buffer.transform(goal, self.cfg.TF_REF_FRAME, self.tf_timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            logging.warn(e)
            return False
        return self._move_to(goal)

    def move_to_absolute(self, pos, rot=None):
        goal = self._make_pose_stamped(pos, rot)
        if goal is None:
            return False
        return self._move_to(goal)

    def _move_to(self, target):
        goal = MoveBaseGoal(target_pose=target)
        self.move_base_client.send_goal(goal)
        return self.move_base_client.wait_for_result()

    def publish_episode_goal(self, pos):
        pose = self._make_pose_stamped(pos)
        self.episode_goal_pub.publish(pose)

    def seed_rng(self, seed):
        self.rng = numpy.random.default_rng(seed)
