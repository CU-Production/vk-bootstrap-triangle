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

# glm
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE glm)

# vma
add_library(vma INTERFACE)
target_include_directories(vma INTERFACE vma)

# vk-bootstrap
add_library(vkb vk-bootstrap/VkBootstrap.cpp vk-bootstrap/VkBootstrap.h vk-bootstrap/VkBootstrapDispatch.h)
target_include_directories(vkb PUBLIC vk-bootstrap ${Vulkan_INCLUDE_DIR})