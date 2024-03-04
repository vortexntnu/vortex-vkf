find_package(ament_cmake_gtest REQUIRED)
ament_add_gtest(${PROJECT_NAME}_test 
  gtest_main.cpp
  gtest_assertions.cpp
  ../src/utils/plotting.cpp
  ../src/utils/ellipse.cpp

  dynamic_model_test.cpp
  ekf_test.cpp
  imm_test.cpp
  numerical_integration_test.cpp
  probability_test.cpp
  sensor_model_test.cpp
  ukf_test.cpp
)
target_include_directories(${PROJECT_NAME}_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
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