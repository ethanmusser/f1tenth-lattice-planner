cmake_minimum_required(VERSION 3.5)
project(lattice_planner_pkg)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  trajectory_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Custom Services
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  srv/BasicService.srv
  DEPENDENCIES nav_msgs 
)
 
# Include Cpp "include" directory
include_directories(include)
include_directories(
)

# Create Cpp executable
# add_executable(lattice_planner_node src/lattice_planner_node.cpp)
# 
# ament_target_dependencies(lattice_planner_node 
#   rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2_ros
# )

# Install Cpp executables
install(TARGETS
  # lattice_planner_node
  DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/lattice_planner_node.py
  scripts/pure_pursuit_node.py
  scripts/waypoint_logger_service.py
  scripts/laser_scan_helpers.py
  scripts/graph_ltpl_helpers.py
  scripts/visualization_helpers.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch, config, inputs
install(
  DIRECTORY launch config inputs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



ament_package()