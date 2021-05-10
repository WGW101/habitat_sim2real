import math
import rospy
import tf2_ros
import tf2_geometry_msgs
import tf_conversions
quaternion_multiply = tf_conversions.transformations.quaternion_multiply
from geometry_msgs.msg import PoseStamped


def _make_pose_stamped(pos, rot=None):
    if rot is None:
        rot = (0, 0, 0, 1)
    # we get pose of hab_robot_frame in hab_ref_frame

    # we build pose of robot_frame in hab_ref_frame
    try:
        tf = tf_buffer.lookup_transform("habitat_base_footprint",
                                        "base_footprint",
                                        rospy.Time(0), tf_timeout)
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        return None
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "habitat_map"
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
        pose = tf_buffer.transform(pose, "map", tf_timeout)
    except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        return None
    return pose


rospy.init_node("test_make_pose_stamped")
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)
tf_timeout = rospy.Duration(1)
pub = rospy.Publisher("test_pose", PoseStamped, queue_size=1, latch=True)

while not rospy.is_shutdown():
    pos = list(float(x) for x in input('Pos > ').split())
    try:
        a = math.radians(float(input('Rot > ')))
        rot = [0, math.sin(0.5 * a), 0, math.cos(0.5 * a)]
    except ValueError:
        rot = None
    pose = _make_pose_stamped(pos, rot)
    pub.publish(pose)

