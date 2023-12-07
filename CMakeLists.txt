cmake_minimum_required(VERSION 3.26)
project(vulkan_hpp_tutorial)

set(CMAKE_CXX_STANDARD 20)

# 3rd_party libs
add_subdirectory(3rd_party)

# apps
add_subdirectory(src/triangle)
add_subdirectory(src/dynamic_rendering)
add_subdirectory(src/mesh_shader)
