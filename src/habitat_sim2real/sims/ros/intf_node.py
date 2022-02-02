import threading
import numpy as np
import logging

import rospy
import message_filters
import cv_bridge
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_multiply
import actionlib

from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped
from sensor_msgs.msg import Image, LaserScan
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
        except ConnectionRefusedError as e:
            raise RuntimeError("Unable to connect to ROS master.") from e
        rospy.init_node(cfg.NODE_NAME)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_timeout = rospy.Duration(self.cfg.TF_TIMEOUT)

        self.last_p = (0, 0, 0)
        self.last_q = (0, 0, 0, 1)

        self.color_sub = message_filters.Subscriber(cfg.COLOR_IMAGE_TOPIC, Image)
        self.depth_sub = message_filters.Subscriber(cfg.DEPTH_IMAGE_TOPIC, Image)
        self.img_sync = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub],
                                                         cfg.IMAGE_SYNC_QUEUE_SIZE)
        self.bridge = cv_bridge.CvBridge()
        self.raw_color_buffer = None
        self.raw_depth_buffer = None
        self.img_buffer_lock = threading.Lock()
        self.has_first_images = threading.Event()
        self.img_sync.registerCallback(self.on_img)

        self.scan_buffer = None
        self.scan_buffer_lock = threading.Lock()
        self.has_first_scan = threading.Event()
        self.scan_sub = rospy.Subscriber(cfg.SCAN_TOPIC, LaserScan, self.on_scan)

        self.map_resolution = None
        self.map_origin_transform = None
        self.map_grid = None
        self.map_free_points = None
        self.map_lock = threading.Lock()
        self.has_first_map = threading.Event()
        self.map_sub = rospy.Subscriber(cfg.MAP_TOPIC, OccupancyGrid, self.on_map)

        self.collided_lock = threading.Lock()
        self.collided = False
        self.cancel_move_on_bump = True
        self.bump_sub = rospy.Subscriber(cfg.BUMPER_TOPIC, BumperEvent, self.on_bump)

        self.move_base_client = actionlib.SimpleActionClient(cfg.MOVE_BASE_ACTION_SERVER,
                                                             MoveBaseAction)
        if not self.move_base_client.wait_for_server(timeout):
            raise RuntimeError("Unable to connect to move_base action server.")

        try:
            rospy.wait_for_service(cfg.DYNAMIXEL_SERVICE, timeout)
            self.dynamixel_cmd_proxy = rospy.ServiceProxy(cfg.DYNAMIXEL_SERVICE,
                                                          DynamixelCommand)
        except rospy.ROSException:
            self.dynamixel_cmd_proxy = None
        self.tilt_target_value = None
        self.tilt_target_event = threading.Event()
        self.tilt_reached_event = threading.Event()
        self.dynamixel_sub = rospy.Subscriber(cfg.DYNAMIXEL_STATE_TOPIC,
                                              DynamixelStateList,
                                              self.on_dynamixel_state)

        try:
            rospy.wait_for_service(cfg.MOVE_BASE_PLAN_SERVICE, timeout)
        except rospy.ROSException as e:
            raise RuntimeError("Unable to connect to get_plan service.") from e
        self.get_plan_proxy = rospy.ServiceProxy(cfg.MOVE_BASE_PLAN_SERVICE, GetPlan)

        self.episode_goal_pub = rospy.Publisher("habitat_episode_goal", PoseStamped,
                                                queue_size=1, latch=True)

        self.rng = np.random.default_rng()

        self.point_lock = threading.Lock()
        self.last_point = None
        self.has_new_point = threading.Event()
        self.pt_sub = rospy.Subscriber("/clicked_point", PointStamped, self.on_point)

    def on_img(self, color_img_msg, depth_img_msg):
        try:
            raw_color = self.bridge.imgmsg_to_cv2(color_img_msg, "passthrough")
            raw_depth = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except cv_bridge.CvBridgeError as e:
            logging.warn(e)
            return
        with self.img_buffer_lock:
            self.raw_color_buffer = raw_color
            self.raw_depth_buffer = raw_depth
        self.has_first_images.set()

    def on_scan(self, scan_msg):
        with self.scan_buffer_lock:
            self.raw_scan_buffer = np.array(scan_msg.ranges, dtype=np.float32)
        self.has_first_scan.set()

    def get_raw_observations(self):
        if not self.has_first_images.wait(self.cfg.GETTER_TIMEOUT):
            raise RuntimeError("Timed out waiting for raw image.")
        if not self.has_first_scan.wait(self.cfg.GETTER_TIMEOUT):
            raise RuntimeError("Timed out waiting for raw scan.")
        with self.img_buffer_lock:
            with self.scan_buffer_lock:
                return (self.raw_color_buffer, self.raw_depth_buffer, self.raw_scan_buffer)

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

        grid = np.array(occ_grid_msg.data).reshape(occ_grid_msg.info.height,
                                                   occ_grid_msg.info.width)
        grid = grid[::-1, ::-1].T
        free_points = np.stack(np.nonzero(
            (grid < self.cfg.MAP_FREE_THRESH) & (grid > -1)
        ), -1)

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

    def get_map_bounds(self):
        if not self.has_first_map.wait(self.cfg.GETTER_TIMEOUT):
            raise RuntimeError("Timed out waiting for map.")
        with self.map_lock:
            origin = self.map_origin_transform.transform.translation
            resolution = self.map_resolution
            shape = self.map_grid.shape
        high = np.array([origin.x, origin.y, origin.z])
        low = high - resolution * np.array([shape[0], 0.0, shape[1]])
        return low, high

    def on_point(self, pt_msg):
        try:
            pt = self.tf_buffer.transform(pt_msg, self.cfg.TF_HABITAT_REF_FRAME,
                                          self.tf_timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            logging.warn(e)
            return
        with self.point_lock:
            self.last_point = [pt.point.x, pt.point.y, pt.point.z]
        self.has_new_point.set()

    def sample_free_point(self):
        if self.cfg.SAMPLE_NAV_PT_METHOD == "RVIZ":
            print("Please use rviz to select a navigable point on the map")
            self.has_new_point.wait()
            with self.point_lock:
                pt = self.last_point
            print("Using point:", pt)
            self.has_new_point.clear()
            return pt
        else:
            if not self.has_first_map.wait(self.cfg.GETTER_TIMEOUT):
                raise RuntimeError("Timed out waiting for map.")
            with self.map_lock:
                pt = self.rng.choice(self.map_free_points) * self.map_resolution
                pose = PoseStamped()
                pose.pose.position.x = pt[0]
                pose.pose.position.y = pt[1]
                pose.pose.orientation.w = 1
                pose = tf2_geometry_msgs.do_transform_pose(pose, self.map_origin_transform)
            return [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]

    def on_bump(self, bump_msg):
        if bump_msg.state == BumperEvent.PRESSED and self.cancel_move_on_bump:
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
            return self.last_p, self.last_q
        p = (trans.transform.translation.x,
             trans.transform.translation.y,
             trans.transform.translation.z)
        q = (trans.transform.rotation.x, trans.transform.rotation.y,
             trans.transform.rotation.z, trans.transform.rotation.w)
        self.last_p, self.last_q = p, q
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
        rot = quaternion_multiply(rot, tf_q)
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

    def _get_raw_plan(self, src, dst):
        start = self._make_pose_stamped(src)
        if start is None:
            return []
        goal = self._make_pose_stamped(dst)
        if goal is None:
            return []
        res = self.get_plan_proxy(start, goal, self.cfg.MOVE_BASE_PLAN_TOL)
        return res.plan.poses

    def get_distance(self, src, dst):
        plan = self._get_raw_plan(src, dst)
        if plan:
            poses = np.array([[(p := pose.pose.position).x, p.y, p.z] for pose in plan])
            return np.sqrt(((poses[1:] - poses[:-1])**2).sum(axis=1)).sum()
        else:
            return np.inf

    def get_shortest_path(self, src, dst):
        plan = self._get_raw_plan(src, dst)
        if plan:
            try:
                shortest_path = []
                for pose in plan:
                    pose = self.tf_buffer.transform(pose, self.cfg.TF_HABITAT_REF_FRAME,
                                                    self.tf_timeout)
                    p = pose.pose.position
                    shortest_path.append([p.x, p.y, p.z])
                return shortest_path
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                logging.warn(e)
        return []

    @property
    def can_tilt_cam(self):
        return self.dynamixel_cmd_proxy is not None

    def set_camera_tilt(self, tilt):
        if self.dynamixel_cmd_proxy is None:
            return False

        self.tilt_reached_event.clear()
        self.tilt_target_value = int(self.cfg.DYNAMIXEL_TICK_OFFSET
                                     - self.cfg.DYNAMIXEL_TICK_PER_RAD * tilt)
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
        goal.pose.orientation.z = np.sin(0.5 * turn)
        goal.pose.orientation.w = np.cos(0.5 * turn)
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
        if pose is not None:
            self.episode_goal_pub.publish(pose)

    def seed_rng(self, seed):
        self.rng = np.random.default_rng(seed)
