from ekf_python2.ekf_py2 import EKF
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros


class PublishNode():
    def __init__(self):
        self.sleep_rate = 0.5
        rospy.init_node("fake_publisher") #,  log_level=rospy.DEBUG
        
        self.pub = rospy.Publisher('/object_detection/object_pose/gate', PoseStamped, queue_size=1)
        
        #Frame names
        self.parent_frame = 'mocap'
        self.child_frame = 'auv/camerafront_link'
        self.object_estimated = 'object_estimated'
        self.object = "object_truth"

        self.__tfBuffer = tf2_ros.Buffer()# Add a tf buffer length? tf2_ros.Buffer(rospy.Duration(1200.0))
        self.__listener = tf2_ros.TransformListener(self.__tfBuffer)
        self.__tfBroadcaster = tf2_ros.TransformBroadcaster()

        #Checks if transform exists
        while self.__tfBuffer.can_transform(self.parent_frame, self.child_frame, rospy.Time()) == 0:
            try:
                rospy.loginfo("No transform between "+str(self.parent_frame) +' and ' + str(self.child_frame))
                rospy.sleep(2)
            except: #, tf2_ros.ExtrapolationException  (tf2_ros.LookupException, tf2_ros.ConnectivityException)
                rospy.sleep(2)
                continue
    
 
    def transformbroadcast(self, p):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.object
        t.transform.translation.x = p.pose.position.x
        t.transform.translation.y = p.pose.position.y
        t.transform.translation.z = p.pose.position.z
        t.transform.rotation.x = p.pose.orientation.x
        t.transform.rotation.y = p.pose.orientation.y
        t.transform.rotation.z = p.pose.orientation.z
        t.transform.rotation.w = p.pose.orientation.w
        self.__tfBroadcaster.sendTransform(t)

    def callback(self):
        while not rospy.is_shutdown():
            n = np.random.normal(0, 0.1**2, 3)
            n_2 = np.random.normal(0, 0.2**2, 3)
            print(n)
            gate = PoseStamped()
            gate.header.frame_id = self.object_estimated
            gate.pose.position.x = 2.30953173828    
            gate.pose.position.y = -0.274446075439  
            gate.pose.position.z = 1.12427368164    
            gate.pose.orientation.x = 0.0168736690947
            gate.pose.orientation.y = 0.0740790786578
            gate.pose.orientation.z = 0.0434789330927
            gate.pose.orientation.w = 0.99616120054
            rospy.loginfo(gate)
            self.pub.publish(gate)
            self.transformbroadcast(gate)
            rospy.sleep(self.sleep_rate) #Publish rate

if __name__ == '__main__':
    try:
        ekf_vision = PublishNode()
        ekf_vision.callback()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass