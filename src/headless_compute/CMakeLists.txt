set(APP_NAME headless_compute)

add_executable(${APP_NAME} main.cpp ../../3rd_party/vk-bootstrap/VkBootstrap.cpp)
target_link_libraries(${APP_NAME} PRIVATE stb hmm vma vkb rdc ${CMAKE_DL_LIBS})

# shader compile
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator")
    set(DXC_EXE "$ENV{VULKAN_SDK}/Bin/dxc")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator")
    set(DXC_EXE "$ENV{VULKAN_SDK}/Bin32/dxc")
endif()

set(HLSL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Noise.hlsl)

foreach(HLSL ${HLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${HLSL} NAME)
    set(CS_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.comp.spv")
    add_custom_command(
            OUTPUT ${CS_SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${DXC_EXE} -spirv -T "cs_6_0" -E "mainCS" ${HLSL} -Fo ${CS_SPIRV}
            DEPENDS ${HLSL}
            VERBATIM
    )
    target_sources(${APP_NAME} PRIVATE ${CS_SPIRV})
endforeach(HLSL)