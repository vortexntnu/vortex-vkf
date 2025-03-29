cmake_minimum_required(VERSION 3.5)

# === Project name ===
project(vortex_filtering)

# === C++ standard ===
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  endif()

# === Compiler flags ===
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall -Wextra -Wpedantic
    -fopenmp # For parallel processing with Eigen
    -fconcepts-diagnostics-depth=3 # For better concepts error messages
    -Warray-bounds # For better array bounds error messages
  )
endif()

# === Dependencies ===
find_package(ament_cmake    REQUIRED)
find_package(rclcpp         REQUIRED)
find_package(std_msgs       REQUIRED)
find_package(geometry_msgs  REQUIRED)
find_package(Eigen3 3.4.0   REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
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

# = Specify the dependencies
set(lib_deps
  Eigen3
  OpenMP
)
set(lib_src
  src/utils/plotting.cpp
  src/utils/ellipse.cpp
)

# === Include directories ===
include_directories(include)

# === Libraries ===
add_library(${PROJECT_NAME}_lib
  ${lib_src}
)
# = Specify dependencies to link against
ament_target_dependencies(${PROJECT_NAME}_lib PUBLIC
  ${lib_deps}
)
# = Specify namespaced dependencies to link against
target_link_libraries(${PROJECT_NAME}_lib PUBLIC
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)
# = Specify the include directories for the library
target_include_directories(${PROJECT_NAME}_lib PUBLIC
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
ament_export_dependencies(eigen3_cmake_module)
ament_export_dependencies(${lib_deps}) # Export the dependencies so the package using this package can find them


# === Install headers ===
install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  # add_compile_definitions(GNUPLOT_ENABLE=1)
  add_subdirectory(test)
endif()

ament_package()
