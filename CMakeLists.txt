cmake_minimum_required(VERSION 3.0.2)
project(vision_kf)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf
  nav_msgs
  tf2_ros
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ekf_node
#  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


