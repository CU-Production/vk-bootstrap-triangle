set(APP_NAME multi_queue)

add_executable(${APP_NAME} main.cpp ../../3rd_party/vk-bootstrap/VkBootstrap.cpp)
target_link_libraries(${APP_NAME} PRIVATE glfw hmm vma vkb ${CMAKE_DL_LIBS})

# shader compile
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator")
    set(DXC_EXE "$ENV{VULKAN_SDK}/Bin/dxc")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator")
    set(DXC_EXE "$ENV{VULKAN_SDK}/Bin32/dxc")
endif()

set(GLSL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quad.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/quad.vert)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
#    message(${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV})
    add_custom_command(
            TARGET ${APP_NAME} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL}
            VERBATIM
    )
endforeach(GLSL)

set(HLSL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Noise.hlsl)

foreach(HLSL ${HLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${HLSL} NAME)
    set(CS_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.comp.spv")
    add_custom_command(
            TARGET ${APP_NAME} PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${DXC_EXE} -spirv -T "cs_6_0" -E "mainCS" ${HLSL} -Fo ${CS_SPIRV}
            DEPENDS ${HLSL}
            VERBATIM
    )
endforeach(HLSL)
