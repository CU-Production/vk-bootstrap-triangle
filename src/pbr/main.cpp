#include <stdio.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <format>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <HandmadeMath.h>

#define VMA_IMPLEMENTATION
#define VMA_VULKAN_VERSION 1003000 // Vulkan 1.3
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <VkBootstrap.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct Vertex {
    HMM_Vec3 position;
    HMM_Vec3 normal;
};

struct VertexShaderPushConstants {
    HMM_Mat4 vp_matrix;
    HMM_Vec4 cumtom_param;
};

struct PixelShaderPushConstants {
    HMM_Vec4 lightPositions[4];
    HMM_Vec4 cameraPosition;
    HMM_Vec4 albedo;
};

struct Init {
    GLFWwindow* window;
    vkb::Instance instance;
    vkb::InstanceDispatchTable inst_disp;
    VkSurfaceKHR surface;
    vkb::Device device;
    vkb::PhysicalDevice physical_device;
    vkb::DispatchTable disp;
    vkb::Swapchain swapchain;
    VmaAllocator allocator;
};

struct RenderData {
    VkQueue graphics_queue;
    VkQueue present_queue;

    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;

    VkImage depth_image;
    VmaAllocation depth_image_allocation;
    VkImageView depth_image_view;
    VkFormat depth_image_format;

    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;

    VkCommandPool command_pool;
    std::vector<VkCommandBuffer> command_buffers;

    std::vector<VkSemaphore> available_semaphores;
    std::vector<VkSemaphore> finished_semaphore;
    std::vector<VkFence> in_flight_fences;
    std::vector<VkFence> image_in_flight;
    size_t current_frame = 0;
    size_t number_of_frame = 0;

    VkBuffer vertex_buffer;
    VmaAllocation vertex_buffer_allocation;

    VkBuffer index_buffer;
    VmaAllocation index_buffer_allocation;

    VkDescriptorPool descriptor_pool;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    struct {
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        float custom_metallic = 0.0f;
        float custom_roughness = 0.0f;
        HMM_Vec4 albedo = {1.00f, 0.71f, 0.09f, 1.00f}; // Gold
    } imgui_state;
};

GLFWwindow* create_window_glfw(const char* window_name = "", bool resize = true) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (!resize) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    return glfwCreateWindow(1280, 720, window_name, NULL, NULL);
}

void destroy_window_glfw(GLFWwindow* window) {
    glfwDestroyWindow(window);
    glfwTerminate();
}

VkSurfaceKHR create_surface_glfw(VkInstance instance, GLFWwindow* window, VkAllocationCallbacks* allocator = nullptr) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult err = glfwCreateWindowSurface(instance, window, allocator, &surface);
    if (err) {
        const char* error_msg;
        int ret = glfwGetError(&error_msg);
        if (ret != 0) {
            std::cout << ret << " ";
            if (error_msg != nullptr) std::cout << error_msg;
            std::cout << "\n";
        }
        surface = VK_NULL_HANDLE;
    }
    return surface;
}

int device_initialization(Init& init) {
    init.window = create_window_glfw("Pbr Demo", true);

    vkb::InstanceBuilder instance_builder;
    auto instance_ret = instance_builder
        .use_default_debug_messenger()
        .set_app_name("triangle")
        .set_engine_name("Null engine")
        .request_validation_layers()
        .require_api_version(1, 3)
        .build();
    if (!instance_ret) {
        std::cout << instance_ret.error().message() << "\n";
        return -1;
    }
    init.instance = instance_ret.value();

    init.inst_disp = init.instance.make_table();

    init.surface = create_surface_glfw(init.instance, init.window);

    VkPhysicalDeviceVulkan13Features vulkan_13_features{};
    vulkan_13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan_13_features.dynamicRendering = true;
    vulkan_13_features.synchronization2 = true;

    vkb::PhysicalDeviceSelector phys_device_selector(init.instance);
    auto phys_device_ret = phys_device_selector
        .set_surface(init.surface)
        .set_minimum_version(1, 3)
        .require_dedicated_transfer_queue()
        // .add_required_extension("VK_KHR_timeline_semaphore")
        // .add_required_extension("VK_KHR_dynamic_rendering")
        .set_required_features_13(vulkan_13_features)
        .add_required_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)
        .select();
    if (!phys_device_ret) {
        std::cout << phys_device_ret.error().message() << "\n";
        return -1;
    }
    init.physical_device = phys_device_ret.value();

    vkb::DeviceBuilder device_builder{ init.physical_device };
    auto device_ret = device_builder.build();
    if (!device_ret) {
        std::cout << device_ret.error().message() << "\n";
        return -1;
    }
    init.device = device_ret.value();

    init.disp = init.device.make_table();

    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = init.instance.fp_vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = init.device.fp_vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.physicalDevice = init.device.physical_device;
    allocatorCreateInfo.device = init.device;
    allocatorCreateInfo.instance = init.instance;
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

    vmaCreateAllocator(&allocatorCreateInfo, &init.allocator);

    return 0;
}

int create_swapchain(Init& init) {

    vkb::SwapchainBuilder swapchain_builder{ init.device };
    auto swap_ret = swapchain_builder
        .set_old_swapchain(init.swapchain)
        // .set_desired_format({VK_FORMAT_R8G8B8A8_SRGB})
        .set_desired_format({VK_FORMAT_R8G8B8A8_UINT})
        .build();
    if (!swap_ret) {
        std::cout << swap_ret.error().message() << " " << swap_ret.vk_result() << "\n";
        return -1;
    }
    vkb::destroy_swapchain(init.swapchain);
    init.swapchain = swap_ret.value();
    return 0;
}

int get_queues(Init& init, RenderData& data) {
    auto gq = init.device.get_queue(vkb::QueueType::graphics);
    if (!gq.has_value()) {
        std::cout << "failed to get graphics queue: " << gq.error().message() << "\n";
        return -1;
    }
    data.graphics_queue = gq.value();

    auto pq = init.device.get_queue(vkb::QueueType::present);
    if (!pq.has_value()) {
        std::cout << "failed to get present queue: " << pq.error().message() << "\n";
        return -1;
    }
    data.present_queue = pq.value();
    return 0;
}

int create_depthimage(Init& init, RenderData& data) {
    data.depth_image_format = VK_FORMAT_D32_SFLOAT_S8_UINT;

    VkImageCreateInfo depth_image_info{};
    depth_image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depth_image_info.imageType = VK_IMAGE_TYPE_2D;
    depth_image_info.format = data.depth_image_format;
    depth_image_info.extent.width = init.swapchain.extent.width;
    depth_image_info.extent.height = init.swapchain.extent.height;
    depth_image_info.extent.depth = 1;
    depth_image_info.mipLevels = 1;
    depth_image_info.arrayLayers = 1;
    depth_image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    depth_image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo depthAllocInfo{};
    depthAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    depthAllocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vmaCreateImage(init.allocator, &depth_image_info, &depthAllocInfo, &data.depth_image, &data.depth_image_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "could not allocate depth buffer memory\n";
        return -1;
    }

    VkImageViewCreateInfo depth_image_view_info{};
    depth_image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depth_image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depth_image_view_info.image = data.depth_image;
    depth_image_view_info.format = data.depth_image_format;
    depth_image_view_info.subresourceRange.baseMipLevel = 0;
    depth_image_view_info.subresourceRange.levelCount = 1;
    depth_image_view_info.subresourceRange.baseArrayLayer = 0;
    depth_image_view_info.subresourceRange.layerCount = 1;
    depth_image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    if (init.disp.createImageView(&depth_image_view_info, nullptr, &data.depth_image_view) != VK_SUCCESS) {
        std::cout << "could not create depth buffer image view\n";
        return -1;
    }

    return 0;
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t file_size = (size_t)file.tellg();
    std::vector<char> buffer(file_size);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(file_size));

    file.close();

    return buffer;
}

VkShaderModule createShaderModule(Init& init, const std::vector<char>& code) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (init.disp.createShaderModule(&create_info, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE; // failed to create shader module
    }

    return shaderModule;
}

int create_graphics_pipeline(Init& init, RenderData& data) {
    auto vert_code = readFile("shaders/pbr.hlsl.vert.spv");
    auto frag_code = readFile("shaders/pbr.hlsl.frag.spv");

    VkShaderModule vert_module = createShaderModule(init, vert_code);
    VkShaderModule frag_module = createShaderModule(init, frag_code);
    if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
        std::cout << "failed to create shader module\n";
        return -1; // failed to create shader modules
    }

    VkPipelineShaderStageCreateInfo vert_stage_info = {};
    vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_info.module = vert_module;
    vert_stage_info.pName = "mainVS";

    VkPipelineShaderStageCreateInfo frag_stage_info = {};
    frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_info.module = frag_module;
    frag_stage_info.pName = "mainPS";

    VkPipelineShaderStageCreateInfo shader_stages[] = { vert_stage_info, frag_stage_info };

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertex_input_info.pVertexBindingDescriptions = &bindingDescription;
    vertex_input_info.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)init.swapchain.extent.width;
    viewport.height = (float)init.swapchain.extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = init.swapchain.extent;

    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &colorBlendAttachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;

    VkPushConstantRange vs_push_constant_range;
    vs_push_constant_range.offset = 0;
    vs_push_constant_range.size = sizeof(VertexShaderPushConstants);
    vs_push_constant_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPushConstantRange ps_push_constant_range;
    ps_push_constant_range.offset = sizeof(VertexShaderPushConstants);
    ps_push_constant_range.size = sizeof(PixelShaderPushConstants);
    ps_push_constant_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkPushConstantRange, 2> push_constant_ranges = {vs_push_constant_range, ps_push_constant_range};

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 0;
    // pipeline_layout_info.pSetLayouts = &data.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges.size());
    pipeline_layout_info.pPushConstantRanges = push_constant_ranges.data();

    if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &data.pipeline_layout) != VK_SUCCESS) {
        std::cout << "failed to create pipeline layout\n";
        return -1; // failed to create pipeline layout
    }

    std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamic_info = {};
    dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
    dynamic_info.pDynamicStates = dynamic_states.data();

    VkPipelineDepthStencilStateCreateInfo depthz_stencil_info = {};
    depthz_stencil_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthz_stencil_info.depthTestEnable = VK_TRUE;
    depthz_stencil_info.depthWriteEnable = VK_TRUE;
    depthz_stencil_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthz_stencil_info.depthBoundsTestEnable = VK_FALSE;
    depthz_stencil_info.minDepthBounds = 0.0f;
    depthz_stencil_info.maxDepthBounds = 1.0f;
    depthz_stencil_info.stencilTestEnable = VK_FALSE;

    VkPipelineRenderingCreateInfo pipeline_rendering_create_info{};
    pipeline_rendering_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipeline_rendering_create_info.colorAttachmentCount = 1;
    pipeline_rendering_create_info.pColorAttachmentFormats = &init.swapchain.image_format;
    pipeline_rendering_create_info.depthAttachmentFormat = data.depth_image_format;
    pipeline_rendering_create_info.stencilAttachmentFormat = data.depth_image_format;

    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = &pipeline_rendering_create_info;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_info;
    pipeline_info.pDepthStencilState = &depthz_stencil_info;
    pipeline_info.layout = data.pipeline_layout;
    // pipeline_info.renderPass = data.render_pass;
    pipeline_info.renderPass = nullptr;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

    if (init.disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &data.graphics_pipeline) != VK_SUCCESS) {
        std::cout << "failed to create pipline\n";
        return -1; // failed to create graphics pipeline
    }

    init.disp.destroyShaderModule(frag_module, nullptr);
    init.disp.destroyShaderModule(vert_module, nullptr);
    return 0;
}

int create_command_pool(Init& init, RenderData& data) {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = init.device.get_queue_index(vkb::QueueType::graphics).value();

    if (init.disp.createCommandPool(&pool_info, nullptr, &data.command_pool) != VK_SUCCESS) {
        std::cout << "failed to create command pool\n";
        return -1; // failed to create command pool
    }
    return 0;
}

int create_command_buffers(Init& init, RenderData& data) {
    data.swapchain_images = init.swapchain.get_images().value();
    data.swapchain_image_views = init.swapchain.get_image_views().value();
    data.command_buffers.resize(data.swapchain_image_views.size());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = data.command_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)data.command_buffers.size();

    if (init.disp.allocateCommandBuffers(&allocInfo, data.command_buffers.data()) != VK_SUCCESS) {
        return -1; // failed to allocate command buffers;
    }
    return 0;
}

int create_sync_objects(Init& init, RenderData& data) {
    data.available_semaphores.resize(init.swapchain.image_count);
    data.finished_semaphore.resize(init.swapchain.image_count);
    data.in_flight_fences.resize(init.swapchain.image_count);
    data.image_in_flight.resize(init.swapchain.image_count, VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < init.swapchain.image_count; i++) {
        if (init.disp.createSemaphore(&semaphore_info, nullptr, &data.available_semaphores[i]) != VK_SUCCESS ||
            init.disp.createSemaphore(&semaphore_info, nullptr, &data.finished_semaphore[i]) != VK_SUCCESS ||
            init.disp.createFence(&fence_info, nullptr, &data.in_flight_fences[i]) != VK_SUCCESS) {
            std::cout << "failed to create sync objects\n";
            return -1; // failed to create synchronization objects for a frame
        }
    }
    return 0;
}

int load_obj_data(Init& init, RenderData& data, const std::string& obj_file_path) {
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader sphere_obj_reader;

    sphere_obj_reader.ParseFromFile(obj_file_path, reader_config);

    auto& shapes = sphere_obj_reader.GetShapes();
    auto& attrib = sphere_obj_reader.GetAttrib();

    if (1 != shapes.size()) {
        std::cout << "This application only accepts one shape!\n";
        return -1;
    }

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        uint32_t index_offset = 0;
        uint32_t idxBufIdx = 0;
        for (uint32_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            uint32_t fv = shapes[s].mesh.num_face_vertices[f];

            // Loop over vertices in the face.
            for (uint32_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                float vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                float vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                float vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                assert(idx.normal_index >= 0, "The model doesn't have normal information but it is necessary.");
                float nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                float ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                float nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                data.vertices.push_back({{vx, vy, vz}, {nx, ny, nz}});
                data.indices.push_back(idxBufIdx);
                idxBufIdx++;
            }
            index_offset += fv;
        }
    }

    return 0;
}

int create_vertex_buffer(Init& init, RenderData& data) {
    const uint32_t buffer_size = sizeof(data.vertices[0]) * data.vertices.size();
    /* vertex buffer */
    VkBufferCreateInfo vertex_buffer_info{};
    vertex_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vertex_buffer_info.size = buffer_size;
    vertex_buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vertex_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vertex_buffer_alloc_info = {};
    vertex_buffer_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateBuffer(init.allocator, &vertex_buffer_info, &vertex_buffer_alloc_info, &data.vertex_buffer, &data.vertex_buffer_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create vertex buffer\n";
        return -1; // failed to create vertex buffer
    }

    /* staging buffer for copy */
    VkBuffer vertex_buffer_staging_buffer;
    VmaAllocation vertex_buffer_staging_buffer_allocation;
    VmaAllocationInfo vma_vertex_buffer_alloc_info;

    VkBufferCreateInfo staging_buffer_alloc_info{};
    staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_buffer_alloc_info.size = buffer_size;;
    staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo staging_alloc_info{};
    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &vertex_buffer_staging_buffer, &vertex_buffer_staging_buffer_allocation, &vma_vertex_buffer_alloc_info) != VK_SUCCESS) {
        std::cout << "failed to create vertex buffer\n";
        return -1; // failed to create vertex buffer
    }

    /* copy data to staging buffer*/
    memcpy(vma_vertex_buffer_alloc_info.pMappedData, data.vertices.data(), vertex_buffer_info.size);

    VkCommandBufferAllocateInfo cmdbuffer_alloc_info{};
    cmdbuffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdbuffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdbuffer_alloc_info.commandPool = data.command_pool;
    cmdbuffer_alloc_info.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    init.disp.allocateCommandBuffers(&cmdbuffer_alloc_info, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    init.disp.beginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = buffer_size;
    init.disp.cmdCopyBuffer(commandBuffer, vertex_buffer_staging_buffer, data.vertex_buffer, 1, &copyRegion);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &commandBuffer;

    init.disp.endCommandBuffer(commandBuffer);

    init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    init.disp.queueWaitIdle(data.graphics_queue);

    init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

    vmaDestroyBuffer(init.allocator, vertex_buffer_staging_buffer, vertex_buffer_staging_buffer_allocation);

    return 0;
}

int create_index_buffer(Init& init, RenderData& data) {
    const uint32_t buffer_size = sizeof(data.indices[0]) * data.indices.size();
    /* index buffer */
    VkBufferCreateInfo index_buffer_info{};
    index_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    index_buffer_info.size = buffer_size;
    index_buffer_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    index_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo index_buffer_alloc_info = {};
    index_buffer_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateBuffer(init.allocator, &index_buffer_info, &index_buffer_alloc_info, &data.index_buffer, &data.index_buffer_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create index buffer\n";
        return -1; // failed to create vertex buffer
    }

    /* staging buffer for copy */
    VkBuffer index_buffer_staging_buffer;
    VmaAllocation index_buffer_staging_buffer_allocation;
    VmaAllocationInfo vma_index_buffer_alloc_info;

    VkBufferCreateInfo staging_buffer_alloc_info{};
    staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_buffer_alloc_info.size = buffer_size;;
    staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo staging_alloc_info{};
    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &index_buffer_staging_buffer, &index_buffer_staging_buffer_allocation, &vma_index_buffer_alloc_info) != VK_SUCCESS) {
        std::cout << "failed to create index buffer\n";
        return -1; // failed to create vertex buffer
    }

    /* copy data to staging buffer*/
    memcpy(vma_index_buffer_alloc_info.pMappedData, data.indices.data(), index_buffer_info.size);

    VkCommandBufferAllocateInfo cmdbuffer_alloc_info{};
    cmdbuffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdbuffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdbuffer_alloc_info.commandPool = data.command_pool;
    cmdbuffer_alloc_info.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    init.disp.allocateCommandBuffers(&cmdbuffer_alloc_info, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    init.disp.beginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = buffer_size;
    init.disp.cmdCopyBuffer(commandBuffer, index_buffer_staging_buffer, data.index_buffer, 1, &copyRegion);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &commandBuffer;

    init.disp.endCommandBuffer(commandBuffer);

    init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    init.disp.queueWaitIdle(data.graphics_queue);

    init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

    vmaDestroyBuffer(init.allocator, index_buffer_staging_buffer, index_buffer_staging_buffer_allocation);

    return 0;
}

int create_descriptor_pool(Init& init, RenderData& data) {
    std::array<VkDescriptorPoolSize, 2> pool_size{};
    pool_size[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size[0].descriptorCount = init.swapchain.image_count;
    pool_size[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size[1].descriptorCount = init.swapchain.image_count;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = pool_size.size();
    pool_info.pPoolSizes = pool_size.data();
    pool_info.maxSets = init.swapchain.image_count * 2;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    init.disp.createDescriptorPool(&pool_info, nullptr, &data.descriptor_pool);

    return 0;
}

int imgui_initialization(Init& init, RenderData& data) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    const int image_count = init.swapchain.image_count;
    const int queue_family_index = init.device.get_queue_index(vkb::QueueType::graphics).value();
    ImGui_ImplGlfw_InitForVulkan(init.window, true);
    ImGui_ImplVulkan_LoadFunctions([](const char* function_name, void* user_data) {
        Init* init = (Init*)user_data;
        return init->instance.fp_vkGetInstanceProcAddr(init->instance, function_name);
    }, &init);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = init.instance;
    init_info.PhysicalDevice = init.physical_device;
    init_info.Device = init.device;
    init_info.QueueFamily = queue_family_index;
    init_info.Queue = data.graphics_queue;
    init_info.DescriptorPool = data.descriptor_pool;
    init_info.Subpass = 0;
    init_info.MinImageCount = init.swapchain.image_count;
    init_info.ImageCount = image_count;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = true;
    init_info.ColorAttachmentFormat = init.swapchain.image_format;
    ImGui_ImplVulkan_Init(&init_info, nullptr);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return a nullptr. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype for higher quality font rendering.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != nullptr);

    return 0;
}

int recreate_swapchain(Init& init, RenderData& data) {
    init.disp.deviceWaitIdle();

    init.disp.destroyCommandPool(data.command_pool, nullptr);

    init.swapchain.destroy_image_views(data.swapchain_image_views);

    if (0 != create_swapchain(init)) return -1;
    if (0 != create_command_pool(init, data)) return -1;
    if (0 != create_command_buffers(init, data)) return -1;
    return 0;
}

int draw_frame(Init& init, RenderData& data) {
    init.disp.waitForFences(1, &data.in_flight_fences[data.current_frame], VK_TRUE, UINT64_MAX);

    uint32_t image_index = 0;
    VkResult result = init.disp.acquireNextImageKHR(
        init.swapchain, UINT64_MAX, data.available_semaphores[data.current_frame], VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return recreate_swapchain(init, data);
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        std::cout << "failed to acquire swapchain image. Error " << result << "\n";
        return -1;
    }

    if (data.image_in_flight[image_index] != VK_NULL_HANDLE) {
        init.disp.waitForFences(1, &data.image_in_flight[image_index], VK_TRUE, UINT64_MAX);
    }
    data.image_in_flight[image_index] = data.in_flight_fences[data.current_frame];

    {
        const int i = image_index;

        init.disp.resetCommandBuffer(data.command_buffers[i], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (init.disp.beginCommandBuffer(data.command_buffers[i], &begin_info) != VK_SUCCESS) {
            return -1; // failed to begin recording command buffer
        }

        // Begin frame
        {
            //[ERROR: Validation]
            // Validation Error: [ UNASSIGNED-CoreValidation-DrawState-InvalidImageLayout ] Object 0: handle = 0x14ce6e2afc0, type = VK
            // _OBJECT_TYPE_COMMAND_BUFFER; Object 1: handle = 0xe7f79a0000000005, type = VK_OBJECT_TYPE_IMAGE; | MessageID = 0x4dae563
            // 5 | vkQueueSubmit(): pSubmits[0].pCommandBuffers[0] command buffer VkCommandBuffer 0x14ce6e2afc0[] expects VkImage 0xe7f
            // 79a0000000005[] (subresource: aspectMask 0x1 array layer 0, mip level 0) to be in layout VK_IMAGE_LAYOUT_COLOR_ATTACHMEN
            // T_OPTIMAL--instead, current layout is VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.

            VkImageSubresourceRange image_subresource_range{};
            image_subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_subresource_range.baseMipLevel = 0;
            image_subresource_range.levelCount = 1;
            image_subresource_range.baseArrayLayer = 0;
            image_subresource_range.layerCount = 1;

            VkImageMemoryBarrier2 image_memory_barrier2{};
            image_memory_barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            image_memory_barrier2.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            image_memory_barrier2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_memory_barrier2.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            image_memory_barrier2.srcStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            image_memory_barrier2.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            image_memory_barrier2.image = data.swapchain_images[i];
            image_memory_barrier2.subresourceRange = image_subresource_range;

            VkDependencyInfo dependency_info{};
            dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
            dependency_info.imageMemoryBarrierCount = 1;
            dependency_info.pImageMemoryBarriers = &image_memory_barrier2;

            init.disp.cmdPipelineBarrier2(data.command_buffers[i], &dependency_info);
        }

        VkDebugUtilsLabelEXT debug_utils_label{};
        debug_utils_label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        debug_utils_label.pLabelName = "Forward pass";
        debug_utils_label.color[0] = 0.3f;
        debug_utils_label.color[1] = 0.0f;
        debug_utils_label.color[2] = 0.7f;
        debug_utils_label.color[3] = 1.0f;

        {
            init.disp.cmdBeginDebugUtilsLabelEXT(data.command_buffers[i], &debug_utils_label);

            ImVec4 im_clear_color = data.imgui_state.clear_color;
            VkClearValue clearColor{ { { im_clear_color.x, im_clear_color.y, im_clear_color.z, 1.0f } } };
            VkClearValue clearDepth{{1.0, 0}};

            VkViewport viewport = {};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)init.swapchain.extent.width;
            viewport.height = (float)init.swapchain.extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            VkRect2D scissor = {};
            scissor.offset = { 0, 0 };
            scissor.extent = init.swapchain.extent;

            VkRenderingAttachmentInfo color_attachment_info{};
            color_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment_info.imageView = data.swapchain_image_views[i];
            color_attachment_info.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            color_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attachment_info.clearValue = clearColor;

            VkRenderingAttachmentInfo depth_stencil_attachment_info{};
            depth_stencil_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_stencil_attachment_info.imageView = data.depth_image_view;
            depth_stencil_attachment_info.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            depth_stencil_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_stencil_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_stencil_attachment_info.clearValue = clearDepth;

            VkRenderingInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea = {{0, 0}, {init.swapchain.extent.width, init.swapchain.extent.height}};
            rendering_info.layerCount = 1;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_attachment_info;
            rendering_info.pDepthAttachment = &depth_stencil_attachment_info;

            init.disp.cmdSetViewport(data.command_buffers[i], 0, 1, &viewport);
            init.disp.cmdSetScissor(data.command_buffers[i], 0, 1, &scissor);

            init.disp.cmdBeginRendering(data.command_buffers[i], &rendering_info);

            init.disp.cmdBindPipeline(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.graphics_pipeline);

            VkBuffer vertex_buffers[] = {data.vertex_buffer};
            VkDeviceSize offsets[] = {0};
            init.disp.cmdBindVertexBuffers(data.command_buffers[i], 0, 1, vertex_buffers, offsets);
            init.disp.cmdBindIndexBuffer(data.command_buffers[i], data.index_buffer, 0, VK_INDEX_TYPE_UINT32);

            HMM_Vec3 cam_pos = { 0.f,0.f,0.f };
            HMM_Mat4 view = HMM_LookAt_RH( cam_pos, {15, 0, 0}, {0, 1, 0});
            HMM_Mat4 projection = HMM_Perspective_RH_NO(70.f * HMM_DegToRad, 1700.f / 900.f, 0.1f, 200.0f);
            projection[1][1] *= -1;
            HMM_Mat4 mesh_matrix = projection * view;

            VertexShaderPushConstants vs_constants{};
            vs_constants.vp_matrix = mesh_matrix;
            vs_constants.cumtom_param.X = data.imgui_state.custom_metallic;
            vs_constants.cumtom_param.Y = data.imgui_state.custom_roughness;

            init.disp.cmdPushConstants(data.command_buffers[i], data.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VertexShaderPushConstants), &vs_constants);

            PixelShaderPushConstants ps_constants{};
            ps_constants.lightPositions[0] = {10.f,  3.f, -8.f, 0.0f,};
            ps_constants.lightPositions[1] = {10.f,  3.f,  8.f, 0.0f,};
            ps_constants.lightPositions[2] = {10.f, -3.f, -8.f, 0.0f,};
            ps_constants.lightPositions[3] = {10.f, -3.f,  8.f, 0.0f,};
            ps_constants.cameraPosition = {cam_pos.X, cam_pos.Y, cam_pos.Z, 0.0f,};
            ps_constants.albedo = data.imgui_state.albedo;

            init.disp.cmdPushConstants(data.command_buffers[i], data.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(VertexShaderPushConstants), sizeof(PixelShaderPushConstants), &ps_constants);

            init.disp.cmdDrawIndexed(data.command_buffers[i], data.indices.size(), 15, 0, 0, 0);

            init.disp.cmdEndRendering(data.command_buffers[i]);

            init.disp.cmdEndDebugUtilsLabelEXT(data.command_buffers[i]);
        }

        // imgui
        {
            // Start the Dear ImGui frame
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos({ 20, 20 }, ImGuiCond_Once);
            ImGui::SetNextWindowSize({ 220, 150 }, ImGuiCond_Once);
            ImGui::SetNextWindowBgAlpha(0.35f);
            if (ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::ColorEdit3("clear color", (float*)&data.imgui_state.clear_color);
                ImGui::SliderFloat("metallic", &data.imgui_state.custom_metallic, 0.0f, 1.0f);
                ImGui::SliderFloat("roughness", &data.imgui_state.custom_roughness, 0.0f, 1.0f);
                ImGui::ColorEdit3("albedo", (float*)&data.imgui_state.albedo);
            }
            ImGui::End();

            // Rendering
            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();

            debug_utils_label.pLabelName = "IMGUI pass";
            init.disp.cmdBeginDebugUtilsLabelEXT(data.command_buffers[i], &debug_utils_label);

            VkRenderingAttachmentInfo color_attachment_info{};
            color_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment_info.imageView = data.swapchain_image_views[i];
            color_attachment_info.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            color_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            // color_attachment_info.clearValue = clearColor;

            VkRenderingInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea = {{0, 0}, {init.swapchain.extent.width, init.swapchain.extent.height}};
            rendering_info.layerCount = 1;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_attachment_info;
            rendering_info.pDepthAttachment = VK_NULL_HANDLE;
            init.disp.cmdBeginRendering(data.command_buffers[i], &rendering_info);

            // Record dear imgui primitives into command buffer
            ImGui_ImplVulkan_RenderDrawData(draw_data, data.command_buffers[i]);

            init.disp.cmdEndRendering(data.command_buffers[i]);

            init.disp.cmdEndDebugUtilsLabelEXT(data.command_buffers[i]);

        }


        // End frame
        {
            //[ERROR: Validation]
            // Validation Error: [ VUID-VkPresentInfoKHR-pImageIndices-01430 ] Object 0: handle = 0x22de58d8da0, type = VK_OBJECT_TYPE_
            // QUEUE; | MessageID = 0x48ad24c6 | vkQueuePresentKHR(): pPresentInfo->pSwapchains[0] images passed to present must be in
            // layout VK_IMAGE_LAYOUT_PRESENT_SRC_KHR or VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR but is in VK_IMAGE_LAYOUT_UNDEFINED. The Vu
            // lkan spec states: Each element of pImageIndices must be the index of a presentable image acquired from the swapchain spe
            // cified by the corresponding element of the pSwapchains array, and the presented image subresource must be in the VK_IMAG
            // E_LAYOUT_PRESENT_SRC_KHR or VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR layout at the time the operation is executed on a VkDevic
            // e (https://vulkan.lunarg.com/doc/view/1.3.268.0/windows/1.3-extensions/vkspec.html#VUID-VkPresentInfoKHR-pImageIndices-0
            // 1430)

            VkImageSubresourceRange image_subresource_range{};
            image_subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_subresource_range.baseMipLevel = 0;
            image_subresource_range.levelCount = 1;
            image_subresource_range.baseArrayLayer = 0;
            image_subresource_range.layerCount = 1;

            VkImageMemoryBarrier2 image_memory_barrier2{};
            image_memory_barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            image_memory_barrier2.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            image_memory_barrier2.oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            image_memory_barrier2.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            image_memory_barrier2.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            image_memory_barrier2.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            image_memory_barrier2.image = data.swapchain_images[i];
            image_memory_barrier2.subresourceRange = image_subresource_range;

            VkDependencyInfo dependency_info{};
            dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
            dependency_info.imageMemoryBarrierCount = 1;
            dependency_info.pImageMemoryBarriers = &image_memory_barrier2;

            init.disp.cmdPipelineBarrier2(data.command_buffers[i], &dependency_info);
        }

        if (init.disp.endCommandBuffer(data.command_buffers[i]) != VK_SUCCESS) {
            std::cout << "failed to record command buffer\n";
            return -1; // failed to record command buffer!
        }
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore wait_semaphores[] = { data.available_semaphores[data.current_frame] };
    VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = wait_semaphores;
    submitInfo.pWaitDstStageMask = wait_stages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &data.command_buffers[image_index];

    VkSemaphore signal_semaphores[] = { data.finished_semaphore[data.current_frame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signal_semaphores;

    init.disp.resetFences(1, &data.in_flight_fences[data.current_frame]);

    if (init.disp.queueSubmit(data.graphics_queue, 1, &submitInfo, data.in_flight_fences[data.current_frame]) != VK_SUCCESS) {
        std::cout << "failed to submit draw command buffer\n";
        return -1; //"failed to submit draw command buffer
    }

    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;

    VkSwapchainKHR swapChains[] = { init.swapchain };
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapChains;

    present_info.pImageIndices = &image_index;

    result = init.disp.queuePresentKHR(data.present_queue, &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        return recreate_swapchain(init, data);
    } else if (result != VK_SUCCESS) {
        std::cout << "failed to present swapchain image\n";
        return -1;
    }

    data.current_frame = (data.current_frame + 1) % (init.swapchain.image_count - 1);
    data.number_of_frame += 1;
    return 0;
}

void cleanup(Init& init, RenderData& data) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    init.disp.destroyDescriptorPool(data.descriptor_pool, nullptr);

    vmaDestroyBuffer(init.allocator, data.index_buffer, data.index_buffer_allocation);

    vmaDestroyBuffer(init.allocator, data.vertex_buffer, data.vertex_buffer_allocation);

    for (size_t i = 0; i < init.swapchain.image_count; i++) {
        init.disp.destroySemaphore(data.finished_semaphore[i], nullptr);
        init.disp.destroySemaphore(data.available_semaphores[i], nullptr);
        init.disp.destroyFence(data.in_flight_fences[i], nullptr);
    }

    init.disp.destroyCommandPool(data.command_pool, nullptr);

    init.disp.destroyPipeline(data.graphics_pipeline, nullptr);
    init.disp.destroyPipelineLayout(data.pipeline_layout, nullptr);

    init.swapchain.destroy_image_views(data.swapchain_image_views);

    init.disp.destroyImageView(data.depth_image_view, nullptr);
    vmaDestroyImage(init.allocator, data.depth_image, data.depth_image_allocation);

    vkb::destroy_swapchain(init.swapchain);

    vmaDestroyAllocator(init.allocator);

    vkb::destroy_device(init.device);
    vkb::destroy_surface(init.instance, init.surface);
    vkb::destroy_instance(init.instance);
    destroy_window_glfw(init.window);
}

int main() {
    Init init;
    RenderData render_data;

    if (0 != device_initialization(init)) return -1;
    if (0 != create_swapchain(init)) return -1;
    if (0 != get_queues(init, render_data)) return -1;
    if (0 != create_depthimage(init, render_data)) return -1;
    if (0 != load_obj_data(init, render_data, "data/uvNormalSphere.obj")) return -1;
    if (0 != create_command_pool(init, render_data)) return -1;
    if (0 != create_vertex_buffer(init, render_data)) return -1;
    if (0 != create_index_buffer(init, render_data)) return -1;
    if (0 != create_descriptor_pool(init, render_data)) return -1;
    if (0 != create_graphics_pipeline(init, render_data)) return -1;
    if (0 != create_command_buffers(init, render_data)) return -1;
    if (0 != create_sync_objects(init, render_data)) return -1;
    if (0 != imgui_initialization(init, render_data)) return -1;

    while (!glfwWindowShouldClose(init.window)) {
        glfwPollEvents();
        int res = draw_frame(init, render_data);
        if (res != 0) {
            std::cout << "failed to draw frame \n";
            return -1;
        }
    }
    init.disp.deviceWaitIdle();

    cleanup(init, render_data);
    return 0;
}