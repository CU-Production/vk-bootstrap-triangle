#include <stdio.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <format>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <HandmadeMath.h>

#define VMA_IMPLEMENTATION
#define VMA_VULKAN_VERSION 1003000 // Vulkan 1.3
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <VkBootstrap.h>

struct Vertex {
    HMM_Vec3 pos;
    HMM_Vec2 uv;
};

const std::vector<Vertex> vertices = {
    {{-0.75f, -0.75f, 0.0f}, {0.0f, 0.0f}},
    {{ 0.75f, -0.75f, 0.0f}, {1.0f, 0.0f}},
    {{ 0.75f,  0.75f, 0.0f}, {1.0f, 1.0f}},

    {{-0.75f, -0.75f, 0.0f}, {0.0f, 0.0f}},
    {{ 0.75f,  0.75f, 0.0f}, {1.0f, 1.0f}},
    {{-0.75f,  0.75f, 0.0f}, {0.0f, 1.0f}},
};

struct VertexShaderPushConstants {
    HMM_Vec4 data;
    HMM_Mat4 mvp_matrix;
};

struct ComputeShaderPushConstants {
    float time;
};

const int MAX_FRAMES_IN_FLIGHT = 2;

struct Init {
    GLFWwindow* window;
    vkb::Instance instance;
    vkb::InstanceDispatchTable inst_disp;
    VkSurfaceKHR surface;
    vkb::Device device;
    vkb::DispatchTable disp;
    vkb::Swapchain swapchain;
    VmaAllocator allocator;
};

struct RenderData {
    VkQueue graphics_queue;
    VkQueue present_queue;

    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;

    VkPipelineLayout graphics_pipeline_layout;
    VkPipeline graphics_pipeline;
    VkDescriptorSetLayout graphics_descriptor_set_layout;

    VkPipelineLayout compute_pipeline_layout;
    VkPipeline compute_pipeline;
    VkDescriptorSetLayout compute_descriptor_set_layout;

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
    VkBuffer staging_buffer;
    VmaAllocation staging_buffer_allocation;

    struct {
        VkImage              image;
        VkImageView          image_view;
        VkSampler            sampler;
        VmaAllocation        vma_allocation;

        VkDescriptorImageInfo descriptor_image_info;
        VkDescriptorImageInfo cs_descriptor_image_info;
    } cs_noise_img;
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
    init.window = create_window_glfw("Vulkan Quad", true);

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
    vkb::PhysicalDevice physical_device = phys_device_ret.value();

    vkb::DeviceBuilder device_builder{ physical_device };
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
    auto swap_ret = swapchain_builder.set_old_swapchain(init.swapchain).build();
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

int create_graphics_descriptor_set_layout(Init& init, RenderData& data) {
    VkDescriptorSetLayoutBinding skybox_layout_binding{};
    skybox_layout_binding.binding = 1;
    skybox_layout_binding.descriptorCount = 1;
    skybox_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    skybox_layout_binding.stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorBindingFlags descriptor_binding_flags{};
    descriptor_binding_flags = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo set_layout_binding_flags{};
    set_layout_binding_flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    set_layout_binding_flags.bindingCount = 1;
    set_layout_binding_flags.pBindingFlags = &descriptor_binding_flags;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &skybox_layout_binding;
    // layout_info.pNext = &set_layout_binding_flags;

    if (init.disp.createDescriptorSetLayout(&layout_info, nullptr, &data.graphics_descriptor_set_layout) != VK_SUCCESS) {
        std::cout <<"failed to create skybox descriptor set layout!\n";
        return -1;
    }
    return 0;
}

int create_graphics_pipeline(Init& init, RenderData& data) {
    auto vert_code = readFile("shaders/quad.vert.spv");
    auto frag_code = readFile("shaders/quad.frag.spv");

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
    vert_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo frag_stage_info = {};
    frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_info.module = frag_module;
    frag_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo shader_stages[] = { vert_stage_info, frag_stage_info };

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, uv);

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
    // rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
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

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &data.graphics_descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &vs_push_constant_range;

    if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &data.graphics_pipeline_layout) != VK_SUCCESS) {
        std::cout << "failed to create pipeline layout\n";
        return -1; // failed to create pipeline layout
    }

    std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamic_info = {};
    dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
    dynamic_info.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo pipeline_rendering_create_info{};
    pipeline_rendering_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipeline_rendering_create_info.colorAttachmentCount = 1;
    pipeline_rendering_create_info.pColorAttachmentFormats = &init.swapchain.image_format;

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
    pipeline_info.layout = data.graphics_pipeline_layout;
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

int create_compute_descriptor_set_layout(Init& init, RenderData& data) {
    VkDescriptorSetLayoutBinding skybox_layout_binding{};
    skybox_layout_binding.binding = 1;
    skybox_layout_binding.descriptorCount = 1;
    skybox_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    skybox_layout_binding.stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorBindingFlags descriptor_binding_flags{};
    descriptor_binding_flags = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo set_layout_binding_flags{};
    set_layout_binding_flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    set_layout_binding_flags.bindingCount = 1;
    set_layout_binding_flags.pBindingFlags = &descriptor_binding_flags;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR | VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &skybox_layout_binding;
    // layout_info.pNext = &set_layout_binding_flags;

    if (init.disp.createDescriptorSetLayout(&layout_info, nullptr, &data.compute_descriptor_set_layout) != VK_SUCCESS) {
        std::cout <<"failed to create skybox descriptor set layout!\n";
        return -1;
    }
    return 0;
}

int create_compute_pipeline(Init& init, RenderData& data) {
    auto cs_code = readFile("shaders/Noise.hlsl.comp.spv");

    VkShaderModule comp_module = createShaderModule(init, cs_code);
    if (comp_module == VK_NULL_HANDLE) {
        std::cout << "failed to create shader module\n";
        return -1; // failed to create shader modules
    }

    VkPipelineShaderStageCreateInfo cs_stage_info = {};
    cs_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cs_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cs_stage_info.module = comp_module;
    cs_stage_info.pName = "mainCS";

    VkPipelineShaderStageCreateInfo shader_stages[] = { cs_stage_info };

    VkPushConstantRange cs_push_constant_range;
    cs_push_constant_range.offset = 0;
    cs_push_constant_range.size = sizeof(ComputeShaderPushConstants);
    cs_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &data.compute_descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &cs_push_constant_range;

    if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &data.compute_pipeline_layout) != VK_SUCCESS) {
        std::cout << "failed to create pipeline layout\n";
        return -1; // failed to create pipeline layout
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = data.compute_pipeline_layout;
    pipeline_info.stage = cs_stage_info;
    if (init.disp.createComputePipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &data.compute_pipeline) != VK_SUCCESS) {
        std::cout << "failed to create pipline\n";
        return -1; // failed to create compute pipeline
    }

    init.disp.destroyShaderModule(comp_module, nullptr);
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
    data.available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
    data.finished_semaphore.resize(MAX_FRAMES_IN_FLIGHT);
    data.in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
    data.image_in_flight.resize(init.swapchain.image_count, VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (init.disp.createSemaphore(&semaphore_info, nullptr, &data.available_semaphores[i]) != VK_SUCCESS ||
            init.disp.createSemaphore(&semaphore_info, nullptr, &data.finished_semaphore[i]) != VK_SUCCESS ||
            init.disp.createFence(&fence_info, nullptr, &data.in_flight_fences[i]) != VK_SUCCESS) {
            std::cout << "failed to create sync objects\n";
            return -1; // failed to create synchronization objects for a frame
        }
    }
    return 0;
}

int create_vertex_buffer(Init& init, RenderData& data) {
    const uint32_t buffer_size = sizeof(vertices[0]) * vertices.size();
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
    VkBufferCreateInfo staging_buffer_alloc_info{};
    staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_buffer_alloc_info.size = buffer_size;;
    staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo staging_alloc_info{};
    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &data.staging_buffer, &data.staging_buffer_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create vertex buffer\n";
        return -1; // failed to create vertex buffer
    }

    /* copy data to staging buffer*/
    void* mapped_data;
    vmaMapMemory(init.allocator, data.staging_buffer_allocation, &mapped_data);
    memcpy(mapped_data, vertices.data(), vertex_buffer_info.size);
    vmaUnmapMemory(init.allocator, data.staging_buffer_allocation);

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
    init.disp.cmdCopyBuffer(commandBuffer, data.staging_buffer, data.vertex_buffer, 1, &copyRegion);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &commandBuffer;

    init.disp.endCommandBuffer(commandBuffer);

    init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    init.disp.queueWaitIdle(data.graphics_queue);

    init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

    return 0;
}

int create_cs_noise_img(Init& init, RenderData& data) {
    VmaAllocationCreateInfo cs_img_vma_alloc_info{};
    cs_img_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    cs_img_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImageCreateInfo cs_img_img_info{};
    cs_img_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    cs_img_img_info.imageType = VK_IMAGE_TYPE_2D;
    cs_img_img_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    cs_img_img_info.extent.width = 512;
    cs_img_img_info.extent.height = 512;
    cs_img_img_info.extent.depth = 1;
    cs_img_img_info.mipLevels = 1;
    cs_img_img_info.arrayLayers = 1;
    cs_img_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    cs_img_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    // cs_img_img_info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    cs_img_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vmaCreateImage(init.allocator, &cs_img_img_info, &cs_img_vma_alloc_info, &data.cs_noise_img.image, &data.cs_noise_img.vma_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create cs_noise_img.image\n";
        return -1;
    }

    // staging buffer copy
    {
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

        // Transform the layout of the image to copy source
        VkImageMemoryBarrier undefToShaderReadOptBarrier{};
        undefToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        undefToShaderReadOptBarrier.image = data.cs_noise_img.image;
        undefToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        undefToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
        undefToShaderReadOptBarrier.subresourceRange.levelCount = 1;
        undefToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
        undefToShaderReadOptBarrier.subresourceRange.layerCount = 1;
        undefToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_NONE;
        undefToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
        undefToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        undefToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &undefToShaderReadOptBarrier);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &commandBuffer;

        init.disp.endCommandBuffer(commandBuffer);

        init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        init.disp.queueWaitIdle(data.graphics_queue);

        init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);
    }

    VkImageViewCreateInfo image_view_info{};
    image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_info.image = data.cs_noise_img.image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 1;

    if (init.disp.createImageView(&image_view_info, nullptr, &data.cs_noise_img.image_view) != VK_SUCCESS) {
        std::cout << "failed to create dummy_img.image_view\n";
        return -1;
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.minLod = -1000;
    sampler_info.maxLod = 1000;
    sampler_info.maxAnisotropy = 1.0f;


    if (init.disp.createSampler(&sampler_info, nullptr, &data.cs_noise_img.sampler) != VK_SUCCESS) {
        std::cout << "failed to create cs_noise_img.sampler\n";
        return -1;
    }

    data.cs_noise_img.descriptor_image_info.sampler = data.cs_noise_img.sampler;
    data.cs_noise_img.descriptor_image_info.imageView = data.cs_noise_img.image_view;
    data.cs_noise_img.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    data.cs_noise_img.cs_descriptor_image_info.sampler = data.cs_noise_img.sampler;
    data.cs_noise_img.cs_descriptor_image_info.imageView = data.cs_noise_img.image_view;
    data.cs_noise_img.cs_descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

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

        // dispatch noise compute shader
        {
            VkDebugUtilsLabelEXT debug_utils_label{};
            debug_utils_label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
            debug_utils_label.pLabelName = "CS pass";
            debug_utils_label.color[0] = 0.7f;
            debug_utils_label.color[1] = 0.0f;
            debug_utils_label.color[2] = 0.3f;
            debug_utils_label.color[3] = 1.0f;

            init.disp.cmdBeginDebugUtilsLabelEXT(data.command_buffers[i], &debug_utils_label);

            // SRV -> UAV
            {
                VkImageMemoryBarrier ShaderReadOptToUndefBarrier{};
                ShaderReadOptToUndefBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                ShaderReadOptToUndefBarrier.image = data.cs_noise_img.image;
                ShaderReadOptToUndefBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                ShaderReadOptToUndefBarrier.subresourceRange.baseMipLevel = 0;
                ShaderReadOptToUndefBarrier.subresourceRange.levelCount = 1;
                ShaderReadOptToUndefBarrier.subresourceRange.baseArrayLayer = 0;
                ShaderReadOptToUndefBarrier.subresourceRange.layerCount = 1;
                ShaderReadOptToUndefBarrier.srcAccessMask = VK_ACCESS_NONE;
                ShaderReadOptToUndefBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                ShaderReadOptToUndefBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                ShaderReadOptToUndefBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

                init.disp.cmdPipelineBarrier(data.command_buffers[i],
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &ShaderReadOptToUndefBarrier);
            }

            init.disp.cmdBindPipeline(data.command_buffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, data.compute_pipeline);

            ComputeShaderPushConstants constants{};
            constants.time = glfwGetTime();

            init.disp.cmdPushConstants(data.command_buffers[i], data.compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeShaderPushConstants), &constants);

            VkWriteDescriptorSet write_descriptor_set{};
            write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_descriptor_set.dstBinding = 1;
            write_descriptor_set.descriptorCount = 1;
            write_descriptor_set.pImageInfo = &data.cs_noise_img.cs_descriptor_image_info;

            init.disp.cmdPushDescriptorSetKHR(data.command_buffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, data.compute_pipeline_layout, 0, 1, &write_descriptor_set);

            init.disp.cmdDispatch(data.command_buffers[i], 512 / 8, 512 / 8, 1);

            // UAV -> SRV
            {
                VkImageMemoryBarrier undefToShaderReadOptBarrier{};
                undefToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                undefToShaderReadOptBarrier.image = data.cs_noise_img.image;
                undefToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                undefToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
                undefToShaderReadOptBarrier.subresourceRange.levelCount = 1;
                undefToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
                undefToShaderReadOptBarrier.subresourceRange.layerCount = 1;
                undefToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                undefToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
                undefToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                undefToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                init.disp.cmdPipelineBarrier(data.command_buffers[i],
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &undefToShaderReadOptBarrier);
            }

            init.disp.cmdEndDebugUtilsLabelEXT(data.command_buffers[i]);
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
        debug_utils_label.pLabelName = "Main pass";
        debug_utils_label.color[0] = 0.3f;
        debug_utils_label.color[1] = 0.0f;
        debug_utils_label.color[2] = 0.7f;
        debug_utils_label.color[3] = 1.0f;

        init.disp.cmdBeginDebugUtilsLabelEXT(data.command_buffers[i], &debug_utils_label);

        VkClearValue clearColor{ { { 0.0f, 0.0f, 0.0f, 1.0f } } };

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

        VkRenderingInfo rendering_info{};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = {{0, 0}, {init.swapchain.extent.width, init.swapchain.extent.height}};
        rendering_info.layerCount = 1;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachments = &color_attachment_info;

        init.disp.cmdSetViewport(data.command_buffers[i], 0, 1, &viewport);
        init.disp.cmdSetScissor(data.command_buffers[i], 0, 1, &scissor);

        init.disp.cmdBeginRendering(data.command_buffers[i], &rendering_info);

        init.disp.cmdBindPipeline(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.graphics_pipeline);

        VkBuffer vertex_buffers[] = {data.vertex_buffer};
        VkDeviceSize offsets[] = {0};
        init.disp.cmdBindVertexBuffers(data.command_buffers[i], 0, 1, vertex_buffers, offsets);

        HMM_Vec3 cam_pos = { 0.f,0.f,-2.f };
        HMM_Mat4 view = HMM_Translate(cam_pos);
        HMM_Mat4 projection = HMM_Perspective_RH_ZO(70.f * HMM_DegToRad, 1700.f / 900.f, 0.1f, 200.0f);
        projection[1][1] *= -1;
        // HMM_Mat4 model = HMM_Rotate_RH(data.number_of_frame * 0.1f * HMM_DegToRad, HMM_V3(0, 1, 0));
        HMM_Mat4 model = HMM_Rotate_RH(0, HMM_V3(0, 1, 0));
        HMM_Mat4 mesh_matrix = projection * view * model;

        VertexShaderPushConstants constants{};
        constants.mvp_matrix = mesh_matrix;

        init.disp.cmdPushConstants(data.command_buffers[i], data.graphics_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VertexShaderPushConstants), &constants);

        VkWriteDescriptorSet write_descriptor_set{};
        write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_descriptor_set.dstBinding = 1;
        write_descriptor_set.descriptorCount = 1;
        write_descriptor_set.pImageInfo = &data.cs_noise_img.descriptor_image_info;

        init.disp.cmdPushDescriptorSetKHR(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.graphics_pipeline_layout, 0, 1, &write_descriptor_set);

        init.disp.cmdDraw(data.command_buffers[i], 6, 1, 0, 0);

        init.disp.cmdEndRendering(data.command_buffers[i]);

        init.disp.cmdEndDebugUtilsLabelEXT(data.command_buffers[i]);

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

    data.current_frame = (data.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    data.number_of_frame += 1;
    return 0;
}

void cleanup(Init& init, RenderData& data) {
    vmaDestroyImage(init.allocator, data.cs_noise_img.image, data.cs_noise_img.vma_allocation);
    init.disp.destroyImageView(data.cs_noise_img.image_view, nullptr);
    init.disp.destroySampler(data.cs_noise_img.sampler, nullptr);

    vmaDestroyBuffer(init.allocator, data.staging_buffer, data.staging_buffer_allocation);
    vmaDestroyBuffer(init.allocator, data.vertex_buffer, data.vertex_buffer_allocation);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        init.disp.destroySemaphore(data.finished_semaphore[i], nullptr);
        init.disp.destroySemaphore(data.available_semaphores[i], nullptr);
        init.disp.destroyFence(data.in_flight_fences[i], nullptr);
    }

    init.disp.destroyCommandPool(data.command_pool, nullptr);

    init.disp.destroyDescriptorSetLayout(data.compute_descriptor_set_layout, nullptr);
    init.disp.destroyDescriptorSetLayout(data.graphics_descriptor_set_layout, nullptr);

    init.disp.destroyPipeline(data.graphics_pipeline, nullptr);
    init.disp.destroyPipelineLayout(data.graphics_pipeline_layout, nullptr);

    init.disp.destroyPipeline(data.compute_pipeline, nullptr);
    init.disp.destroyPipelineLayout(data.compute_pipeline_layout, nullptr);

    init.swapchain.destroy_image_views(data.swapchain_image_views);

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
    if (0 != create_graphics_descriptor_set_layout(init, render_data)) return -1;
    if (0 != create_compute_descriptor_set_layout(init, render_data)) return -1;
    if (0 != create_graphics_pipeline(init, render_data)) return -1;
    if (0 != create_compute_pipeline(init, render_data)) return -1;
    if (0 != create_command_pool(init, render_data)) return -1;
    if (0 != create_vertex_buffer(init, render_data)) return -1;
    if (0 != create_command_buffers(init, render_data)) return -1;
    if (0 != create_sync_objects(init, render_data)) return -1;
    if (0 != create_cs_noise_img(init, render_data)) return -1;

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