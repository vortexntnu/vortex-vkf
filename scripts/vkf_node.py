#!/usr/bin/env python

#import debugpy
#print("Waiting for VSCode debugger...")
#debugpy.listen(5678)
#debugpy.wait_for_client()

##EKF imports
#from logging import exception
from re import X

from ekf_python2.gaussparams_py2 import MultiVarGaussian
from ekf_python2.dynamicmodels_py2 import landmark_gate, landmark_pose_world
from ekf_python2.measurementmodels_py2 import measurement_linear_landmark, LTV_full_measurement_model
from ekf_python2.ekf_py2 import EKF

#Math imports
import numpy as np

#ROS imports
import rospy
from std_msgs.msg import String
from vortex_msgs.msg import ObjectPosition
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf.transformations as tft
import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs


class VKFNode:

    def __init__(self):
        ########################################
        ####Things you can change yourself####
        ########################################

        #Name of the node
        node_name = "ekf_vision"

        #Frame names, e.g. "odom" and "cam"
        self.parent_frame = 'odom'
        self.child_frame = 'zed2_left_camera_frame'
        self.object_frame = ""

        self.current_object = ""

        #Subscribe topic
        object_topic_subscribe = "/pointcloud_processing/poseStamped/spy"
        mission_topic_subscribe = "/fsm/state"

        ##################
        ####EKF stuff#####
        ##################

        # Tuning parameters
        self.sigma_a = 3 / 5 * np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.sigma_z = 2 * np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Making gate model object
        self.landmark_model = landmark_pose_world(self.sigma_a)

        #Gauss prev values
        self.x_hat0 = np.array([0, 0, 0, 0, 0, 0])
        self.P_hat0 = np.diag(100 * self.sigma_z)
        self.prev_gauss = MultiVarGaussian(self.x_hat0, self.P_hat0)

        ################
        ###ROS stuff####
        ################

        # ROS node init
        rospy.init_node(node_name)
        self.last_time = rospy.get_time()

        now = rospy.get_rostime()
        rospy.loginfo("Current time %i %i", now.secs, now.nsecs)

        # Subscribe to mission topic
        self.mission_topic = self.current_object + "/execute"
        self.mission_topic_old = self.mission_topic[:]
        self.mission_topic_sub = rospy.Subscriber(mission_topic_subscribe,
                                                  String, self.update_mission)

        # LMS decision logic
        self.Fnorm_threshold = 0.1
        self.is_detected = True
        self.estimateConverged = False
        self.estimateFucked = False

        # Subscriber to gate pose and orientation
        self.object_pose_sub = rospy.Subscriber(object_topic_subscribe,
                                                PoseStamped,
                                                self.obj_pose_callback,
                                                queue_size=1)

        # Publisher to autonomous
        self.gate_pose_pub = rospy.Publisher('/fsm/object_positions_in',
                                             ObjectPosition,
                                             queue_size=1)

        #TF stuff
        self.__tfBuffer = tf2_ros.Buffer()
        self.__listener = tf2_ros.TransformListener(self.__tfBuffer)
        self.__tfBroadcaster = tf2_ros.TransformBroadcaster()

        self.pose_transformer = tf2_geometry_msgs.tf2_geometry_msgs

        #The init will only continue if a transform between parent frame and child frame can be found
        while self.__tfBuffer.can_transform(self.parent_frame,
                                            self.child_frame,
                                            rospy.Time()) == 0:
            try:
                rospy.loginfo("No transform between " +
                              str(self.parent_frame) + ' and ' +
                              str(self.child_frame))
                rospy.sleep(2)
            except:  #, tf2_ros.ExtrapolationException  (tf2_ros.LookupException, tf2_ros.ConnectivityException)
                rospy.sleep(2)
                continue

        rospy.loginfo("Transform between " + str(self.parent_frame) + ' and ' +
                      str(self.child_frame) + 'found.')

        ############
        ##Init end##
        ############

    def update_mission(self, mission):
        """
        Callback which updates the instance variable storing the fsm state information

        Input:
                mission:    ROS String message from fsm/state subscription

        """
        self.mission_topic = mission.data

    def get_Ts(self):
        """
        Returns the time step since last Kalman update. The previous time is stored in the last_time instance

        Output:
                Ts:   The time step in seconds since last Kalman Update
        """
        Ts = rospy.get_time() - self.last_time
        return Ts

    def ekf_function(self, z, ekf):
        """
        Performs a Kalman Update given the measurement and an ekf object

        Input:
                z:              A MultiVarGaussian object containing the result from the update step

        Output:
                gauss_x_pred:   A MultiVarGauss object resulting from the prediction of the state
                                Can be used in NEES and NIS analyses
                gauss_z_pred:   A MultiVarGauss object resulting from the prediction of the measurement
                                Can be used in NIS analysis of the filter performance
                gauss_est:      A MultiVarGauss object resulting from the update step
        """

        Ts = self.get_Ts()

        gauss_x_pred, gauss_z_pred, gauss_est = ekf.step_with_info(
            self.prev_gauss, z, Ts)

        self.last_time = rospy.get_time()
        self.prev_gauss = gauss_est

        return gauss_x_pred, gauss_z_pred, gauss_est

    def est_to_pose(self, gauss_est):
        """
        Extracts the mean of the estimate from a MultiVarGaussian object and splits it into position and orientation

        Input:
                gauss_est:      A MultiVarGaussian object containing the result from the update step

        Output:
                pos:            A list of length 3 with the position of the estimate
                euler_angs:     A list of length 3 with the euler angles from the estimate
        """
        x_hat = gauss_est.mean
        pos = [x_hat[0], x_hat[1], x_hat[2]]
        euler_angs = [x_hat[3], x_hat[4], x_hat[5]]

        return pos, euler_angs

    def process_measurement_message(self, msg):
        """
        Takes in the msg which is the output of PCP, extracts the needed components and does some calculations/mappings
        to get what we need in the different frames for the VKF.
        Input:
                msg: Output from PCP, a PoseStamped msg
        
        Output:
                z                       : a 6 dim measurement of positions of object wrt camera in camera frame, and orientation between object and camera in euler angles
                R_wc                    : rotation matrix between odom (world) and camera, needed for the LTV sensor model for the Kalman Filters
                cam_pose_position_wc    : position of camera in odom (world), needed for the LTV sensor model for the Kalman Filters
        """
        # Gets the transform from odom to camera
        self.tf_lookup_wc = self.__tfBuffer.lookup_transform(
            self.parent_frame, self.child_frame, rospy.Time(),
            rospy.Duration(5))

        # Run the measurement back through tf tree to get the object in odom
        msg_transformed_wg = self.pose_transformer.do_transform_pose(
            msg, self.tf_lookup_wc)

        #Broadcast to tf to make sure we get the correct transform. Uncomment for testing in rviz
        #self.transformbroadcast("odom", msg_transformed_wg)

        # Extract measurement of transformed and camera message

        # We need the position from the camera measurement
        obj_pose_position_cg = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # We need the orientation from the world measruement
        obj_pose_quat_wg = np.array([
            msg_transformed_wg.pose.orientation.x,
            msg_transformed_wg.pose.orientation.y,
            msg_transformed_wg.pose.orientation.z,
            msg_transformed_wg.pose.orientation.w
        ])

        # We need to detection in world frame for creating new hypotheses
        self.detection_position_odom = np.array([
            msg_transformed_wg.pose.position.x,
            msg_transformed_wg.pose.position.y,
            msg_transformed_wg.pose.position.z
        ])

        # Get time-varying transformation from world to camera

        # We need the position for the EKF
        cam_pose_position_wc = np.array([
            self.tf_lookup_wc.transform.translation.x,
            self.tf_lookup_wc.transform.translation.y,
            self.tf_lookup_wc.transform.translation.z
        ])

        # We need the quaternion to make into a Rot matrix for the EKF
        cam_pose_quat_wc = np.array([
            self.tf_lookup_wc.transform.rotation.x,
            self.tf_lookup_wc.transform.rotation.y,
            self.tf_lookup_wc.transform.rotation.z,
            self.tf_lookup_wc.transform.rotation.w
        ])

        # Rotation from world to camera, needed for the LTV model
        R_wc = tft.quaternion_matrix(cam_pose_quat_wc)
        R_wc = R_wc[:3, :3]

        # Prepare measurement vector
        z_phi, z_theta, z_psi = tft.euler_from_quaternion(obj_pose_quat_wg,
                                                          axes='sxyz')

        z = obj_pose_position_cg
        z = np.append(z, [z_phi, z_theta, z_psi])

        return z, R_wc, cam_pose_position_wc

    def check_filter_convergence(self, gauss_estimate):
        """
        Sets the convergence boolean to True if Frobenius norm under a threshold

        Input:
                gauss_estimate:     A MultiVarGaussian containing the result from the Kalman Update
        """
        cov = gauss_estimate.cov
        Fnorm_P = np.linalg.norm(cov, "fro")

        if Fnorm_P < self.Fnorm_threshold:
            self.estimateConverged = True
        else:
            pass

    def transformbroadcast(self, parent_frame, p):
        """
        Publishes a transform representing the result from the Kalman Update given the name of the parent
        frame and the objectPose message published for the landmark server. Used mainly for the purposes of
        visualization and debugging.

        Input:
                parent_frame:     A string which is the parent frame
                p           :     An objectPose which is the message published to landmark server
        """

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame
        t.child_frame_id = "object_" + str(p.objectID)
        t.transform.translation.x = p.objectPose.pose.position.x
        t.transform.translation.y = p.objectPose.pose.position.y
        t.transform.translation.z = p.objectPose.pose.position.z
        t.transform.rotation.x = p.objectPose.pose.orientation.x
        t.transform.rotation.y = p.objectPose.pose.orientation.y
        t.transform.rotation.z = p.objectPose.pose.orientation.z
        t.transform.rotation.w = p.objectPose.pose.orientation.w
        self.__tfBroadcaster.sendTransform(t)

    def publish_object(self, objectID, ekf_position, ekf_pose_quaterion):
        p = ObjectPosition()
        #p.pose.header[]
        p.objectID = objectID
        # p.objectPose.header = "object_" + str(objectID)
        p.objectPose.pose.position.x = ekf_position[0]
        p.objectPose.pose.position.y = ekf_position[1]
        p.objectPose.pose.position.z = ekf_position[2]
        p.objectPose.pose.orientation.x = ekf_pose_quaterion[0]
        p.objectPose.pose.orientation.y = ekf_pose_quaterion[1]
        p.objectPose.pose.orientation.z = ekf_pose_quaterion[2]
        p.objectPose.pose.orientation.w = ekf_pose_quaterion[3]

        p.isDetected = self.is_detected
        p.estimateConverged = self.estimateConverged
        p.estimateFucked = self.estimateFucked

        self.gate_pose_pub.publish(p)
        rospy.loginfo("Object published: %s", objectID)
        self.transformbroadcast(self.parent_frame, p)

    def obj_pose_callback(self, msg):

        objID = self.mission_topic.split("/")[0]
        rospy.loginfo("Object data recieved for: %s", objID)
        self.current_object = objID

        if self.mission_topic == self.current_object + "/execute":
            rospy.loginfo("Mission status: %s", objID)
            self.prev_gauss = MultiVarGaussian(self.x_hat0, self.P_hat0)
            self.last_time = rospy.get_time()
            return None

        # Process msg of measurement
        z, R_wc, cam_pose_position_wc = self.process_measurement_message(msg)

        # Initialization using the current measurement mapped to odom
        if sorted(self.prev_gauss.mean) == sorted(self.x_hat0):
            z = np.append(self.detection_position_odom[:3], z[3:])
            self.prev_gauss = MultiVarGaussian(z, self.P_hat0)
            self.last_time = rospy.get_time()
            return None

        # Create sensor model and ekf objects
        full_measurement_model = LTV_full_measurement_model(
            self.sigma_z, cam_pose_position_wc, R_wc)
        full_ekf = EKF(self.landmark_model, full_measurement_model)

        # Call EKF step and format the data
        _, _, gauss_est = self.ekf_function(z, full_ekf)

        # Format update Gaussian object
        ekf_position, ekf_pose = self.est_to_pose(gauss_est)
        ekf_pose_quaterion = tft.quaternion_from_euler(ekf_pose[0],
                                                       ekf_pose[1],
                                                       ekf_pose[2])

        # Publish data, update mission topic
        self.mission_topic_old = self.mission_topic
        self.check_filter_convergence(gauss_est)
        self.publish_object(objID, ekf_position, ekf_pose_quaterion)


if __name__ == '__main__':
    while not rospy.is_shutdown():
        try:
            ekf_vision = VKFNode()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
