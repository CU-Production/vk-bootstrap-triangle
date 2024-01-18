set(APP_NAME ozz_cs_skinning)

add_executable(${APP_NAME}
        main.cpp
        ../../3rd_party/vk-bootstrap/VkBootstrap.cpp
        ../../3rd_party/imgui/backends/imgui_impl_vulkan.cpp
        ../../3rd_party/imgui/backends/imgui_impl_glfw.cpp)
target_link_libraries(${APP_NAME} PRIVATE glfw hmm vma vkb imgui ozzanim ${CMAKE_DL_LIBS})

# shader compile
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator")
else()
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin32/glslangValidator")
endif()

set(GLSL_SOURCE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ozz.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ozz.vert)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
#    message(${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV})
    add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
            COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL}
            VERBATIM
    )
    target_sources(${APP_NAME} PRIVATE ${SPIRV})
endforeach(GLSL)

set(OZZ_DATA_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../ozz_demo/data/ozz_skin_animation.ozz
        ${CMAKE_CURRENT_SOURCE_DIR}/../ozz_demo/data/ozz_skin_mesh.ozz
        ${CMAKE_CURRENT_SOURCE_DIR}/../ozz_demo/data/ozz_skin_skeleton.ozz)

foreach(OZZ_DATA_FILE ${OZZ_DATA_FILES})
    add_custom_command(
            TARGET ${APP_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/data/"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OZZ_DATA_FILE} "${CMAKE_CURRENT_BINARY_DIR}/data/"
            DEPENDS ${OZZ_DATA_FILE}
            VERBATIM
    )
endforeach(OZZ_DATA_FILE)
