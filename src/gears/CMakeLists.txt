set(APP_NAME gears)

add_executable(${APP_NAME} main.cpp ../../3rd_party/vk-bootstrap/VkBootstrap.cpp)
target_link_libraries(${APP_NAME} PRIVATE glfw hmm vma vkb ${CMAKE_DL_LIBS})

# shader compile
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator")
endif()

set(GLSL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gear.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gear.vert)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    get_filename_component(FULL_LAST_EXT_NAME ${GLSL} LAST_EXT)
    string(REPLACE "." "" LAST_EXT_NAME ${FULL_LAST_EXT_NAME})
    set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv.h")
    add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV} --vn "${LAST_EXT_NAME}_code"
            DEPENDS ${GLSL}
            VERBATIM
    )
    target_sources(${APP_NAME} PRIVATE ${SPIRV})
endforeach(GLSL)
target_include_directories(${APP_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
