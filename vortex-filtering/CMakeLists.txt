cmake_minimum_required(VERSION 3.5)

# === Project name ===
project(vortex_filtering)

# === C++ standard ===
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  endif()

# === Compiler flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall -Wextra -Wpedantic 
    # -Wno-unused-local-typedefs # Suppress warnings from unused typedefs
    -fopenmp # For parallel processing with Eigen
  )
endif()

# === Dependencies ===
find_package(ament_cmake    REQUIRED)
find_package(rclcpp         REQUIRED)
find_package(std_msgs       REQUIRED)
find_package(geometry_msgs  REQUIRED)
find_package(Eigen3         REQUIRED)
find_package(OpenMP         REQUIRED) # For parallel processing with Eigen

find_package(Gnuplot REQUIRED)
find_package(Boost REQUIRED 
  COMPONENTS 
    iostreams 
    system 
    filesystem
) # for gnuplot


# === Include directories ===
include_directories(include)

# === Executables ===
# add_executable(${PROJECT_NAME}_node 
#   src/nodes/kf_node.cpp
# )
# ament_target_dependencies(${PROJECT_NAME}_node 
#   rclcpp 
#   std_msgs
#   geometry_msgs
#   Gnuplot
#   Boost
# )
# target_link_libraries(${PROJECT_NAME}_node 
#   Eigen3::Eigen
# )

# install(TARGETS
#   ${PROJECT_NAME}_node
#   DESTINATION lib/${PROJECT_NAME}
# )


# === Libraries ===
add_library(${PROJECT_NAME}_lib INTERFACE)

# = Specify the dependencies
set(lib_deps
  Eigen3
  OpenMP
)
set(lib_src
  src/plotting/utils.cpp
)

# = Specify the dependencies to link against
ament_target_dependencies(${PROJECT_NAME}_lib INTERFACE
  ${lib_deps}
)
# = Specify the namespaced dependencies to link against
target_link_libraries(${PROJECT_NAME}_lib INTERFACE
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)
# = Specify the include directories for the INTERFACE of the library
target_include_directories(${PROJECT_NAME}_lib INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)

# === Install libraries ===
install(TARGETS ${PROJECT_NAME}_lib
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# === Export libraries ===
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${lib_deps}) # Export the dependencies so the package using this package can find them


# === Install headers ===
install(
  DIRECTORY include/
  DESTINATION include
)

# === Install launch files ===
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(${PROJECT_NAME}_test 
    test/gtest_main.cpp
    test/gtest_assertions.cpp
    ${lib_src}

    test/dynamic_model_test.cpp
    test/ekf_test.cpp
    # test/filter_test.cpp
    test/imm_test.cpp
    test/numerical_integration_test.cpp
    test/probability_test.cpp
    test/sensor_model_test.cpp
    # test/ukf_test.cpp
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
