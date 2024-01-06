set(APP_NAME bindless)

add_executable(${APP_NAME}
        main.cpp
        ../../3rd_party/vk-bootstrap/VkBootstrap.cpp
        ../../3rd_party/imgui/backends/imgui_impl_vulkan.cpp
        ../../3rd_party/imgui/backends/imgui_impl_glfw.cpp)
target_link_libraries(${APP_NAME} PRIVATE glfw hmm vma vkb imgui tinyobjloader stb ${CMAKE_DL_LIBS})

# shader compile
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(DXC_EXE "$ENV{VULKAN_SDK}/Bin/dxc")
else()
    set(DXC_EXE "$ENV{VULKAN_SDK}/Bin32/dxc")
endif()

set(HLSL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ibl.hlsl
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/skybox.hlsl)

foreach(HLSL ${HLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${HLSL} NAME)
    set(VS_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.vert.spv")
    set(PS_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.frag.spv")
    add_custom_command(
            OUTPUT ${VS_SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${DXC_EXE} -spirv -T "vs_6_0" -E "mainVS" ${HLSL} -Fo ${VS_SPIRV}
            DEPENDS ${HLSL}
            VERBATIM
    )
    add_custom_command(
            OUTPUT ${PS_SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${DXC_EXE} -spirv -T "ps_6_0" -E "mainPS" ${HLSL} -Fo ${PS_SPIRV}
            DEPENDS ${HLSL}
            VERBATIM
    )
    target_sources(${APP_NAME} PRIVATE ${VS_SPIRV} ${PS_SPIRV})
endforeach(HLSL)

set(OBJ_MESH_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../pbr/data/uvNormalSphere.obj)

foreach(OBJ_MESH_FILE ${OBJ_MESH_FILES})
    add_custom_command(
            TARGET ${APP_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/data/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OBJ_MESH_FILE} "${CMAKE_CURRENT_BINARY_DIR}/data/"
            DEPENDS ${OZZ_DATA_FILE}
            VERBATIM
    )
endforeach(OBJ_MESH_FILE)

add_custom_command(
        TARGET ${APP_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/data/"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/../ibl/data "${CMAKE_CURRENT_BINARY_DIR}/data/"
        DEPENDS ${OZZ_DATA_FILE}
        VERBATIM
)

add_custom_command(
        TARGET ${APP_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/data/"
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different  ${CMAKE_CURRENT_SOURCE_DIR}/data "${CMAKE_CURRENT_BINARY_DIR}/data/"
        DEPENDS ${OZZ_DATA_FILE}
        VERBATIM
)
