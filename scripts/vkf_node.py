#!/usr/bin/env python

#import debugpy
#print("Waiting for VSCode debugger...")
#debugpy.listen(5678)
#debugpy.wait_for_client()

##EKF imports
#from logging import exception
from re import X

from ekf_python2.gaussparams_py2 import MultiVarGaussian
from ekf_python2.dynamicmodels_py2 import landmark_gate
from ekf_python2.measurementmodels_py2 import measurement_linear_landmark
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
        object_topic_subscribe = "/pointcloud_processing/object_pose/spy"
        mission_topic_subscribe = "/fsm/state"


        ##################
        ####EKF stuff#####
        ##################

        # Tuning parameters
        self.sigma_a = 3/5*np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.sigma_z = 2*np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Making gate model object
        self.landmark_model = landmark_gate(self.sigma_a)
        self.sensor_model = measurement_linear_landmark(self.sigma_z)
        self.my_ekf = EKF(self.landmark_model, self.sensor_model)

        #Gauss prev values
        self.x_hat0 = np.array([0, 0, 0, 0, 0, 0])
        self.P_hat0 = np.diag(3*self.sigma_z)
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
        self.mission_topic = self.current_object + "_execute"
        self.mission_topic_sub = rospy.Subscriber(mission_topic_subscribe, String, self.update_mission)
        
        # Subscriber to gate pose and orientation 
        self.object_pose_sub = rospy.Subscriber(object_topic_subscribe, ObjectPosition, self.obj_pose_callback, queue_size=1)
      
        # Publisher to autonomous
        self.gate_pose_pub = rospy.Publisher('/fsm/object_positions_in', ObjectPosition, queue_size=1)

        #TF stuff
        self.__tfBuffer = tf2_ros.Buffer()
        self.__listener = tf2_ros.TransformListener(self.__tfBuffer)
        self.__tfBroadcaster = tf2_ros.TransformBroadcaster()

        #The init will only continue if a transform between parent frame and child frame can be found
        while self.__tfBuffer.can_transform(self.parent_frame, self.child_frame, rospy.Time()) == 0:
            try:
                rospy.loginfo("No transform between "+str(self.parent_frame) +' and ' + str(self.child_frame))
                rospy.sleep(2)
            except: #, tf2_ros.ExtrapolationException  (tf2_ros.LookupException, tf2_ros.ConnectivityException)
                rospy.sleep(2)
                continue
        
        rospy.loginfo("Transform between "+str(self.parent_frame) +' and ' + str(self.child_frame) + 'found.')
        
        ############
        ##Init end##
        ############

    def update_mission(self, mission):
        self.mission_topic = mission.data

    def get_Ts(self):
        Ts = rospy.get_time() - self.last_time
        return Ts
    
    def ekf_function(self, z):

        Ts = self.get_Ts()

        gauss_x_pred, gauss_z_pred, gauss_est = self.my_ekf.step_with_info(self.prev_gauss, z, Ts)
        
        self.last_time = rospy.get_time()
        self.prev_gauss = gauss_est

        return gauss_x_pred, gauss_z_pred, gauss_est

    def est_to_pose(self, x_hat):
        x = x_hat[0]
        y = x_hat[1]
        z = x_hat[2]
        pos = [x, y, z]

        euler_angs = [x_hat[3], x_hat[4], x_hat[5]]
        return pos, euler_angs

    def transformbroadcast(self, parent_frame, p):
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
        p.objectPose.header = "object_" + str(objectID)
        p.objectPose.pose.position.x = ekf_position[0]
        p.objectPose.pose.position.y = ekf_position[1]
        p.objectPose.pose.position.z = ekf_position[2]
        p.objectPose.pose.orientation.x = ekf_pose_quaterion[0]
        p.objectPose.pose.orientation.y = ekf_pose_quaterion[1]
        p.objectPose.pose.orientation.z = ekf_pose_quaterion[2]
        p.objectPose.pose.orientation.w = ekf_pose_quaterion[3]
        
        self.gate_pose_pub.publish(p)
        rospy.loginfo("Object published: %s", objectID)
        self.transformbroadcast(self.parent_frame, p)

    def obj_pose_callback(self, msg):

        rospy.loginfo("Object data recieved for: %s", msg.objectID)
        self.current_object = msg.objectID
    
        if self.mission_topic == self.current_object + "_execute":
            rospy.loginfo("Mission status: %s", self.mission_topic)
            self.prev_gauss = MultiVarGaussian(self.x_hat0, self.P_hat0)
            self.last_time = rospy.get_time()
            return None
        

        # Gate in world frame for cyb pool
        obj_pose_position_wg = np.array([msg.objectPose.pose.position.x, 
                                         msg.objectPose.pose.position.y, 
                                         msg.objectPose.pose.position.z])

        obj_pose_pose_wg = np.array([msg.objectPose.pose.orientation.x,
                                     msg.objectPose.pose.orientation.y,
                                     msg.objectPose.pose.orientation.z,
                                     msg.objectPose.pose.orientation.w])

        # Prepare measurement vector
        z_phi, z_theta, z_psi = tft.euler_from_quaternion(obj_pose_pose_wg, axes='sxyz')

        z = obj_pose_position_wg
        z = np.append(z, [z_phi, z_theta, z_psi])
        
        if sorted(self.prev_gauss.mean) == sorted(self.x_hat0):
            self.prev_gauss = MultiVarGaussian(z, self.P_hat0)
            self.last_time = rospy.get_time()
            return None

        # Call EKF step and format the data
        _, _, gauss_est = self.ekf_function(z)
        x_hat = gauss_est.mean

        ekf_position, ekf_pose = self.est_to_pose(x_hat)
        ekf_pose_quaterion = tft.quaternion_from_euler(ekf_pose[0], ekf_pose[1], ekf_pose[2])

        # Publish data
        self.publish_object(msg.objectID, ekf_position, ekf_pose_quaterion)


if __name__ == '__main__':
    while not rospy.is_shutdown():     
        try:
            ekf_vision = VKFNode()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
    
