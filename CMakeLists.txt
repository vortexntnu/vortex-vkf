cmake_minimum_required(VERSION 3.5)
project(vortex-vkf)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  endif()
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall -Wextra -Wpedantic 
    -Wno-unused-local-typedefs # Suppress warnings from model_definitions.hpp
    -fopenmp # For parallel processing with Eigen
  )
endif()

# Find dependencies
find_package(ament_cmake    REQUIRED)
find_package(rclcpp         REQUIRED)
find_package(std_msgs       REQUIRED)
find_package(geometry_msgs  REQUIRED)
find_package(Eigen3         REQUIRED)
find_package(OpenMP         REQUIRED) # For parallel processing with Eigen
find_package(Gnuplot REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams system filesystem) # for gnuplot


# Add include
include_directories(include)



add_executable(${PROJECT_NAME}_node 
  src/nodes/kf_node.cpp
)
ament_target_dependencies(${PROJECT_NAME}_node 
  rclcpp 
  std_msgs
  geometry_msgs
  Gnuplot
  Boost
)
target_link_libraries(${PROJECT_NAME}_node 
  Eigen3::Eigen
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
    test/ekf_test.cpp
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
    Gnuplot
    Boost
  )
  target_link_libraries(${PROJECT_NAME}_test 
    Eigen3::Eigen # Makes us able to use #include <Eigen/Dense> instead of #include <eigen3/Eigen/Dense>
    OpenMP::OpenMP_CXX
  )


endif()

ament_package()
