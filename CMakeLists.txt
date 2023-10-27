cmake_minimum_required(VERSION 3.5)
project(vortex-vkf)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-local-typedefs -fopenmp)
endif()

# Find dependencies
find_package(ament_cmake    REQUIRED)
find_package(rclcpp         REQUIRED)
find_package(std_msgs       REQUIRED)
find_package(geometry_msgs  REQUIRED)
find_package(Eigen3         REQUIRED)


# Add include
include_directories(include)



add_executable(${PROJECT_NAME}_node 
  src/nodes/kf_node.cpp
)
ament_target_dependencies(${PROJECT_NAME}_node 
  rclcpp 
  std_msgs
  geometry_msgs
  Eigen3
)

# Let ros2 run find the executables
install(
  DIRECTORY include/
  DESTINATION include
)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME})



if(BUILD_TESTING)
  # find_package(ament_lint_auto REQUIRED)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  # set(ament_cmake_cpplint_FOUND TRUE)
  # ament_lint_auto_find_test_dependencies()


  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_test 
    test/main.cpp
    # test/matrix_size_test.cpp
    # test/lti_model_test.cpp
    # test/six_dof_model_test.cpp
    # test/ukf_test.cpp
    # test/erk_test.cpp
    test/probability_test.cpp
    test/dynamic_model_test.cpp
    test/sensor_model_test.cpp
  )
  target_include_directories(${PROJECT_NAME}_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${PROJECT_NAME}_test
    std_msgs
  )

endif()

ament_package()
