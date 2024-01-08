# vulkan
find_package(Vulkan REQUIRED)

# GLFW
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "shared")
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "glfw examples")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "glfw tests")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "glfw docs")
set(GLFW_INSTALL OFF CACHE INTERNAL "glfw install")
set(GLFW_VULKAN_STATIC OFF CACHE INTERNAL "glfw vulkan")
add_subdirectory(glfw)

# HandmadeMath
add_library(hmm INTERFACE)
target_include_directories(hmm INTERFACE HandmadeMath)

# vma
add_library(vma INTERFACE)
target_include_directories(vma INTERFACE vma)

# vk-bootstrap
add_library(vkb vk-bootstrap/VkBootstrap.cpp vk-bootstrap/VkBootstrap.h vk-bootstrap/VkBootstrapDispatch.h)
target_include_directories(vkb PUBLIC vk-bootstrap ${Vulkan_INCLUDE_DIR})

# imgui
add_library(imgui imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp)
target_include_directories(imgui PUBLIC imgui)

## ozz-animation
add_library(ozzanim ozz/src/mesh.cc ozz/src/ozz_animation.cc ozz/src/ozz_base.cc)
target_include_directories(ozzanim PUBLIC ozz/include)

##  tinyobjloader
add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader INTERFACE TinyObj)

## stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb)

## thsvs
add_library(thsvs INTERFACE)
target_include_directories(thsvs INTERFACE thsvs)
