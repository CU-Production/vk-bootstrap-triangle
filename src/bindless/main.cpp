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

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_USE_CPP14
#include "tiny_gltf.h"

#define M_PI       3.14159265358979323846   // pi

constexpr uint32_t bindless_img_count = 32;

struct Vertex {
    HMM_Vec3 position;
    HMM_Vec3 normal;
    HMM_Vec4 tangent;
    HMM_Vec2 uv;
};

struct GltfVertexShaderPushConstants {
    HMM_Mat4 m_matrix;
    HMM_Mat4 vp_matrix;
};

struct GltfPixelShaderPushConstants {
    uint32_t texIdx[4];
    uint32_t gltfIdx[4];
    HMM_Vec4 lightPositions[4];
    HMM_Vec4 cameraPosition;
    uint32_t params[4];
};

struct SkyboxPixelShaderPushConstants {
    uint32_t texIdx[4];
    HMM_Vec4 view;
    HMM_Vec4 right;
    HMM_Vec4 up;
    HMM_Vec2 viewportWidthHeight;
    HMM_Vec2 nearWidthHeight; // Near plane's width and height in the world.
    float    camNear; // Pack it with 'up'.
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

    struct {
        VkPipelineLayout pipeline_layout;
        VkPipeline graphics_pipeline;
        VkDescriptorSetLayout descriptor_set_layout;
    } skybox_pipeline;

    struct {
        VkPipelineLayout pipeline_layout;
        VkPipeline graphics_pipeline;
        VkDescriptorSetLayout descriptor_set_layout;
    } spheres_pipeline;

    VkCommandPool command_pool;
    std::vector<VkCommandBuffer> command_buffers;

    std::vector<VkSemaphore> available_semaphores;
    std::vector<VkSemaphore> finished_semaphore;
    std::vector<VkFence> in_flight_fences;
    std::vector<VkFence> image_in_flight;
    size_t current_frame = 0;
    size_t number_of_frame = 0;

    VkDescriptorPool descriptor_pool;

    struct {
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        bool enable_light = false;
        bool enable_ibl_diffuse = true;
        bool enable_ibl_specular = true;
    } imgui_state;

    struct {
        VkImage              image;
        VkImageView          image_view;
        VkSampler            sampler;
        VmaAllocation        vma_allocation;

        VkDescriptorImageInfo descriptor_image_info;
    } hdr_cubemap;

    struct {
        VkImage              image;
        VkImageView          image_view;
        VkSampler            sampler;
        VmaAllocation        vma_allocation;

        VkDescriptorImageInfo descriptor_image_info;
    } diffuse_irradiance_cubemap;

    struct {
        VkImage              image;
        VkImageView          image_view;
        VkSampler            sampler;
        VmaAllocation        vma_allocation;

        VkDescriptorImageInfo descriptor_image_info;
    } prefilter_env_cubemap;

    struct {
        VkImage              image;
        VkImageView          image_view;
        VkSampler            sampler;
        VmaAllocation        vma_allocation;

        VkDescriptorImageInfo descriptor_image_info;
    } envBrdf_img;

    struct {
        VkImage              image;
        VkImageView          image_view;
        VkSampler            sampler;
        VmaAllocation        vma_allocation;

        VkDescriptorImageInfo descriptor_image_info;
    } dummy_img;

    struct {
        VkBuffer vertex_buffer;
        VmaAllocation vertex_buffer_allocation;

        VkBuffer index_buffer;
        VmaAllocation index_buffer_allocation;

        std::vector<Vertex> vertices;
        std::vector<uint16_t> indices;

        struct {
            struct {
                VkImage              image;
                VkImageView          image_view;
                VkSampler            sampler;
                VmaAllocation        vma_allocation;

                VkDescriptorImageInfo descriptor_image_info;
            } albedo;
            struct {
                VkImage              image;
                VkImageView          image_view;
                VkSampler            sampler;
                VmaAllocation        vma_allocation;

                VkDescriptorImageInfo descriptor_image_info;
            } AOMetalRoughness;
            struct {
                VkImage              image;
                VkImageView          image_view;
                VkSampler            sampler;
                VmaAllocation        vma_allocation;

                VkDescriptorImageInfo descriptor_image_info;
            } emissive;
            struct {
                VkImage              image;
                VkImageView          image_view;
                VkSampler            sampler;
                VmaAllocation        vma_allocation;

                VkDescriptorImageInfo descriptor_image_info;
            } normal;
        } textures;
    } gltf_model;
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
    init.window = create_window_glfw("bindless Demo", true);

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

    VkPhysicalDeviceVulkan12Features vulkan_12_features{};
    vulkan_12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan_12_features.descriptorIndexing = true;
    vulkan_12_features.shaderSampledImageArrayNonUniformIndexing = true;
    vulkan_12_features.descriptorBindingSampledImageUpdateAfterBind = true;
    vulkan_12_features.runtimeDescriptorArray = true;
    vulkan_12_features.descriptorBindingVariableDescriptorCount = true;

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
        .set_required_features_12(vulkan_12_features)
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
        .set_desired_present_mode({VK_PRESENT_MODE_FIFO_KHR})
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

int create_skybox_graphics_pipeline(Init& init, RenderData& data) {
    auto vert_code = readFile("shaders/skybox.hlsl.vert.spv");
    auto frag_code = readFile("shaders/skybox.hlsl.frag.spv");

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

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 0;
    vertex_input_info.vertexAttributeDescriptionCount = 0;

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

    VkPushConstantRange ps_push_constant_range;
    ps_push_constant_range.offset = 0;
    ps_push_constant_range.size = sizeof(SkyboxPixelShaderPushConstants);
    ps_push_constant_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &data.skybox_pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &ps_push_constant_range;

    if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &data.skybox_pipeline.pipeline_layout) != VK_SUCCESS) {
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
    pipeline_info.layout = data.skybox_pipeline.pipeline_layout;
    // pipeline_info.renderPass = data.render_pass;
    pipeline_info.renderPass = nullptr;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

    if (init.disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &data.skybox_pipeline.graphics_pipeline) != VK_SUCCESS) {
        std::cout << "failed to create pipline\n";
        return -1; // failed to create graphics pipeline
    }

    init.disp.destroyShaderModule(frag_module, nullptr);
    init.disp.destroyShaderModule(vert_module, nullptr);
    return 0;
}

int create_spheres_graphics_pipeline(Init& init, RenderData& data) {
    auto vert_code = readFile("shaders/ibl.hlsl.vert.spv");
    auto frag_code = readFile("shaders/ibl.hlsl.frag.spv");

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

    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, normal);
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, tangent);
    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Vertex, uv);

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
    vs_push_constant_range.size = sizeof(GltfVertexShaderPushConstants);
    vs_push_constant_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkPushConstantRange ps_push_constant_range;
    ps_push_constant_range.offset = sizeof(GltfVertexShaderPushConstants);
    ps_push_constant_range.size = sizeof(GltfPixelShaderPushConstants);
    ps_push_constant_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkPushConstantRange, 2> push_constant_ranges = {vs_push_constant_range, ps_push_constant_range};

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &data.spheres_pipeline.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges.size());
    pipeline_layout_info.pPushConstantRanges = push_constant_ranges.data();

    if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &data.spheres_pipeline.pipeline_layout) != VK_SUCCESS) {
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
    pipeline_info.layout = data.spheres_pipeline.pipeline_layout;
    // pipeline_info.renderPass = data.render_pass;
    pipeline_info.renderPass = nullptr;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

    if (init.disp.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &data.spheres_pipeline.graphics_pipeline) != VK_SUCCESS) {
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

int load_gltf_data(Init& init, RenderData& data, const std::string& gltf_file_path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    std::string ext = tinygltf::GetFilePathExtension(gltf_file_path);
    bool ret = false;
    if (ext == "glb") {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltf_file_path);
    } else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltf_file_path);
    }

    if (!warn.empty()) {
        printf("tinygltf Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("tinygltf ERR: %s\n", err.c_str());
    }
    if (!ret) {
        printf("Failed to load .glTF : %s\n", gltf_file_path.c_str());
        return -1;
    }

    assert(model.meshes.size() == 1, "DamagedHelmet only hase one model!");
    const tinygltf::Mesh& mesh = model.meshes[0];

    // NOTE: TinyGltf loader has already loaded the binary buffer data and the images data.
    const auto& binaryBuffer = model.buffers[0].data;
    const unsigned char* pBufferData = binaryBuffer.data();

    // load mesh
    {
        int posIdx = mesh.primitives[0].attributes.at("POSITION");
        int normalIdx = mesh.primitives[0].attributes.at("NORMAL");
        int tangentIdx = mesh.primitives[0].attributes.at("TANGENT");
        int uvIdx = mesh.primitives[0].attributes.at("TEXCOORD_0");
        int indicesIdx = mesh.primitives[0].indices;
        int materialIdx = mesh.primitives[0].material;

        // Elements notes:
        // Position: float3, normal: float3, tangent: float4, texcoord: float2.

        // Setup the vertex buffer and the index buffer
        const auto& posAccessor = model.accessors[posIdx];
        int posAccessorByteOffset = posAccessor.byteOffset;
        int posAccessorEleCnt = posAccessor.count; // Assume a position element is a float3.

        const auto& normalAccessor = model.accessors[normalIdx];
        int normalAccessorByteOffset = normalAccessor.byteOffset;
        int normalAccessorEleCnt = normalAccessor.count;

        const auto& tangentAccessor = model.accessors[tangentIdx];
        int tangentAccessorByteOffset = tangentAccessor.byteOffset;
        int tangentAccessorEleCnt = tangentAccessor.count;

        const auto& uvAccessor = model.accessors[uvIdx];
        int uvAccessorByteOffset = uvAccessor.byteOffset;
        int uvAccessorEleCnt = uvAccessor.count;

        const auto& idxAccessor = model.accessors[indicesIdx];
        int idxAccessorByteOffset = idxAccessor.byteOffset;
        int idxAccessorEleCnt = idxAccessor.count;

        // NOTE: Buffer views are just division of the buffer for the flight helmet model.
        // SCALAR is in one buffer view. FLOAT2 in one. FLOAT3 in one. and FLOAT3 in one...
        // Maybe they can be more
        // A buffer view represents a contiguous segment of data in a buffer, defined by a byte offset into the buffer specified
        // in the byteOffset property and a total byte length specified by the byteLength property of the buffer view.
        const auto& posBufferView = model.bufferViews[posAccessor.bufferView];
        const auto& normalBufferView = model.bufferViews[normalAccessor.bufferView];
        const auto& tangentBufferView = model.bufferViews[tangentAccessor.bufferView];
        const auto& uvBufferView = model.bufferViews[uvAccessor.bufferView];
        const auto& idxBufferView = model.bufferViews[idxAccessor.bufferView];

        // We assume that the idx, position, normal, uv and tangent are not interleaved.
        // TODO: Even though they are interleaved, we can use a function to read out the data by making use of the stride bytes count.

        // Assmue the data and element type of the index is uint16_t.
        int idxBufferOffset = idxAccessorByteOffset + idxBufferView.byteOffset;
        int idxBufferByteCnt = sizeof(uint16_t) * idxAccessor.count;
        data.gltf_model.indices.resize(idxAccessor.count);
        memcpy(data.gltf_model.indices.data(), &pBufferData[idxBufferOffset], idxBufferByteCnt);

        // Assmue the data and element type of the position is float3
        int posBufferOffset = posAccessorByteOffset + posBufferView.byteOffset;
        int posBufferByteCnt = sizeof(float) * 3 * posAccessor.count;
        float* pPosData = new float[3 * posAccessor.count];
        memcpy(pPosData, &pBufferData[posBufferOffset], posBufferByteCnt);

        // Assmue the data and element type of the normal is float3.
        int normalBufferOffset = normalAccessorByteOffset + normalBufferView.byteOffset;
        int normalBufferByteCnt = sizeof(float) * 3 * normalAccessor.count;
        float* pNomralData = new float[3 * normalAccessor.count];
        memcpy(pNomralData, &pBufferData[normalBufferOffset], normalBufferByteCnt);

        // Assmue the data and element type of the tangent is float4.
        int tangentBufferOffset = tangentAccessorByteOffset + tangentBufferView.byteOffset;
        int tangentBufferByteCnt = sizeof(float) * 4 * tangentAccessor.count;
        float* pTangentData = new float[4 * tangentAccessor.count];
        memcpy(pTangentData, &pBufferData[tangentBufferOffset], tangentBufferByteCnt);

        // Assume the data and element type of the texcoord is float2.
        int uvBufferOffset = uvAccessorByteOffset + uvBufferView.byteOffset;
        int uvBufferByteCnt = sizeof(float) * 2 * uvAccessor.count;
        float* pUvData = new float[2 * uvAccessor.count];
        memcpy(pUvData, &pBufferData[uvBufferOffset], uvBufferByteCnt);

        data.gltf_model.vertices.resize(posAccessor.count);
        for (uint32_t vertIdx = 0; vertIdx < posAccessor.count; vertIdx++) {
            // pos -- 3 floats
            data.gltf_model.vertices[vertIdx].position.X = pPosData[3 * vertIdx + 0];
            data.gltf_model.vertices[vertIdx].position.Y = pPosData[3 * vertIdx + 1];
            data.gltf_model.vertices[vertIdx].position.Z = pPosData[3 * vertIdx + 2];

            // normal -- 3 floats
            data.gltf_model.vertices[vertIdx].normal.X = pNomralData[3 * vertIdx + 0];
            data.gltf_model.vertices[vertIdx].normal.Y = pNomralData[3 * vertIdx + 1];
            data.gltf_model.vertices[vertIdx].normal.Z = pNomralData[3 * vertIdx + 2];

            // tangent -- 4 floats
            data.gltf_model.vertices[vertIdx].tangent.X = pTangentData[4 * vertIdx + 0];
            data.gltf_model.vertices[vertIdx].tangent.Y = pTangentData[4 * vertIdx + 1];
            data.gltf_model.vertices[vertIdx].tangent.Z = pTangentData[4 * vertIdx + 2];
            data.gltf_model.vertices[vertIdx].tangent.W = pTangentData[4 * vertIdx + 3];

            // uv -- 2 floats
            data.gltf_model.vertices[vertIdx].uv.X = pUvData[2 * vertIdx + 0];
            data.gltf_model.vertices[vertIdx].uv.Y = pUvData[2 * vertIdx + 1];
        }

        free(pPosData);
        free(pNomralData);
        free(pTangentData);
        free(pUvData);

        // Send image info to GPU and set relevant data
        const auto& material = model.materials[materialIdx];

        // A texture binding is defined by an index of a texture object and an optional index of texture coordinates.
        // Its green channel contains roughness values and its blue channel contains metalness values.
        int baseColorTexIdx = material.pbrMetallicRoughness.baseColorTexture.index;
        int metallicRoughnessTexIdx = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
        int occlusionTexIdx = material.occlusionTexture.index;
        int normalTexIdx = material.normalTexture.index;
        int emissiveTexIdx = material.emissiveTexture.index;

        assert(metallicRoughnessTexIdx == occlusionTexIdx, "Conbine AO and metallicRoughness texture");

        // A texture is defined by an image index, denoted by the source property and a sampler index (sampler).
        // Assmue that all textures are 8 bits per channel. They are all xxx / 255. They all have 4 components.
        const auto& baseColorTex = model.textures[baseColorTexIdx];
        const auto& aoMetallicRoughnessTex = model.textures[metallicRoughnessTexIdx];
        const auto& normalTex = model.textures[normalTexIdx];
        const auto& emissiveTex = model.textures[emissiveTexIdx];

        int baseColorTexImgIdx = baseColorTex.source;
        int aoMetallicRoughnessTexImgIdx = aoMetallicRoughnessTex.source;
        int normalTexImgIdx = normalTex.source;
        int emissiveTexImgIdx = emissiveTex.source;

        const auto& baseColorImg = model.images[baseColorTexImgIdx];
        const auto& aoMetalllicRoughnessImg = model.images[aoMetallicRoughnessTexImgIdx];
        const auto& normalImg = model.images[normalTexImgIdx];
        const auto& emissiveImg = model.images[emissiveTexImgIdx];

        // load textures
        {
            // base color img
            {
                VmaAllocationCreateInfo base_color_img_vma_alloc_info{};
                base_color_img_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
                base_color_img_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

                VkImageCreateInfo base_color_img_img_info{};
                base_color_img_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                base_color_img_img_info.imageType = VK_IMAGE_TYPE_2D;
                base_color_img_img_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                base_color_img_img_info.extent.width = baseColorImg.width;
                base_color_img_img_info.extent.height = baseColorImg.height;
                base_color_img_img_info.extent.depth = 1;
                base_color_img_img_info.mipLevels = 1;
                base_color_img_img_info.arrayLayers = 1;
                base_color_img_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
                base_color_img_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                base_color_img_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                if (vmaCreateImage(init.allocator, &base_color_img_img_info, &base_color_img_vma_alloc_info, &data.gltf_model.textures.albedo.image, &data.gltf_model.textures.albedo.vma_allocation, nullptr) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.albedo.image\n";
                    return -1;
                }

                // staging buffer copy
                {
                    const size_t buffer_size = 4 * sizeof(uint8_t) * baseColorImg.width * baseColorImg.height;
                    VkBuffer staging_buffer;
                    VmaAllocation staging_buffer_allocation;
                    VmaAllocationInfo vma_alloc_info;

                    VkBufferCreateInfo staging_buffer_alloc_info{};
                    staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    staging_buffer_alloc_info.size = buffer_size;;
                    staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

                    VmaAllocationCreateInfo staging_alloc_info{};
                    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                    if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
                        std::cout << "failed to staging buffer\n";
                        return -1; // failed to create vertex buffer
                    }

                    /* copy data to staging buffer*/
                    memcpy(vma_alloc_info.pMappedData, baseColorImg.image.data(), vma_alloc_info.size);

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
                    VkImageMemoryBarrier undefToDstBarrier{};
                    undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    undefToDstBarrier.image = data.gltf_model.textures.albedo.image;
                    undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    undefToDstBarrier.subresourceRange.baseMipLevel = 0;
                    undefToDstBarrier.subresourceRange.levelCount = 1;
                    undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
                    undefToDstBarrier.subresourceRange.layerCount = 1;
                    undefToDstBarrier.srcAccessMask = 0;
                    undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &undefToDstBarrier);

                    VkBufferImageCopy copyBufferToImage{};
                    copyBufferToImage.bufferRowLength = baseColorImg.width;
                    copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    copyBufferToImage.imageSubresource.mipLevel = 0;
                    copyBufferToImage.imageSubresource.baseArrayLayer = 0;
                    copyBufferToImage.imageSubresource.layerCount = 1;
                    copyBufferToImage.imageExtent.width = baseColorImg.width;
                    copyBufferToImage.imageExtent.height = baseColorImg.height;
                    copyBufferToImage.imageExtent.depth = 1;
                    init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.gltf_model.textures.albedo.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

                    // Transform the layout of the image to shader access resource
                    VkImageMemoryBarrier dstToShaderReadOptBarrier{};
                    dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    dstToShaderReadOptBarrier.image = data.gltf_model.textures.albedo.image;
                    dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
                    dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
                    dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
                    dstToShaderReadOptBarrier.subresourceRange.layerCount = 1;
                    dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
                    dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &dstToShaderReadOptBarrier);

                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &commandBuffer;

                    init.disp.endCommandBuffer(commandBuffer);

                    init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
                    init.disp.queueWaitIdle(data.graphics_queue);

                    init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

                    vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
                }

                VkImageViewCreateInfo image_view_info{};
                image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                image_view_info.image = data.gltf_model.textures.albedo.image;
                image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                image_view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                image_view_info.subresourceRange.levelCount = 1;
                image_view_info.subresourceRange.layerCount = 1;

                if (init.disp.createImageView(&image_view_info, nullptr, &data.gltf_model.textures.albedo.image_view) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.albedo.image_view\n";
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

                if (init.disp.createSampler(&sampler_info, nullptr, &data.gltf_model.textures.albedo.sampler) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.albedo.sampler\n";
                    return -1;
                }

                data.gltf_model.textures.albedo.descriptor_image_info.sampler = data.gltf_model.textures.albedo.sampler;
                data.gltf_model.textures.albedo.descriptor_image_info.imageView = data.gltf_model.textures.albedo.image_view;
                data.gltf_model.textures.albedo.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            // AO and MetalllicRoughness img
            {
                VmaAllocationCreateInfo ao_metallic_roughness_img_vma_alloc_info{};
                ao_metallic_roughness_img_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
                ao_metallic_roughness_img_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

                VkImageCreateInfo ao_metallic_roughness_img_img_info{};
                ao_metallic_roughness_img_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ao_metallic_roughness_img_img_info.imageType = VK_IMAGE_TYPE_2D;
                ao_metallic_roughness_img_img_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                ao_metallic_roughness_img_img_info.extent.width = aoMetalllicRoughnessImg.width;
                ao_metallic_roughness_img_img_info.extent.height = aoMetalllicRoughnessImg.height;
                ao_metallic_roughness_img_img_info.extent.depth = 1;
                ao_metallic_roughness_img_img_info.mipLevels = 1;
                ao_metallic_roughness_img_img_info.arrayLayers = 1;
                ao_metallic_roughness_img_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
                ao_metallic_roughness_img_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                ao_metallic_roughness_img_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                if (vmaCreateImage(init.allocator, &ao_metallic_roughness_img_img_info, &ao_metallic_roughness_img_vma_alloc_info, &data.gltf_model.textures.AOMetalRoughness.image, &data.gltf_model.textures.AOMetalRoughness.vma_allocation, nullptr) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.AOMetalRoughness.image\n";
                    return -1;
                }

                // staging buffer copy
                {
                    const size_t buffer_size = 4 * sizeof(uint8_t) * aoMetalllicRoughnessImg.width * aoMetalllicRoughnessImg.height;
                    VkBuffer staging_buffer;
                    VmaAllocation staging_buffer_allocation;
                    VmaAllocationInfo vma_alloc_info;

                    VkBufferCreateInfo staging_buffer_alloc_info{};
                    staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    staging_buffer_alloc_info.size = buffer_size;;
                    staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

                    VmaAllocationCreateInfo staging_alloc_info{};
                    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                    if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
                        std::cout << "failed to staging buffer\n";
                        return -1; // failed to create vertex buffer
                    }

                    /* copy data to staging buffer*/
                    memcpy(vma_alloc_info.pMappedData, aoMetalllicRoughnessImg.image.data(), vma_alloc_info.size);

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
                    VkImageMemoryBarrier undefToDstBarrier{};
                    undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    undefToDstBarrier.image = data.gltf_model.textures.AOMetalRoughness.image;
                    undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    undefToDstBarrier.subresourceRange.baseMipLevel = 0;
                    undefToDstBarrier.subresourceRange.levelCount = 1;
                    undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
                    undefToDstBarrier.subresourceRange.layerCount = 1;
                    undefToDstBarrier.srcAccessMask = 0;
                    undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &undefToDstBarrier);

                    VkBufferImageCopy copyBufferToImage{};
                    copyBufferToImage.bufferRowLength = aoMetalllicRoughnessImg.width;
                    copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    copyBufferToImage.imageSubresource.mipLevel = 0;
                    copyBufferToImage.imageSubresource.baseArrayLayer = 0;
                    copyBufferToImage.imageSubresource.layerCount = 1;
                    copyBufferToImage.imageExtent.width = aoMetalllicRoughnessImg.width;
                    copyBufferToImage.imageExtent.height = aoMetalllicRoughnessImg.height;
                    copyBufferToImage.imageExtent.depth = 1;
                    init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.gltf_model.textures.AOMetalRoughness.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

                    // Transform the layout of the image to shader access resource
                    VkImageMemoryBarrier dstToShaderReadOptBarrier{};
                    dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    dstToShaderReadOptBarrier.image = data.gltf_model.textures.AOMetalRoughness.image;
                    dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
                    dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
                    dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
                    dstToShaderReadOptBarrier.subresourceRange.layerCount = 1;
                    dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
                    dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &dstToShaderReadOptBarrier);

                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &commandBuffer;

                    init.disp.endCommandBuffer(commandBuffer);

                    init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
                    init.disp.queueWaitIdle(data.graphics_queue);

                    init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

                    vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
                }

                VkImageViewCreateInfo image_view_info{};
                image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                image_view_info.image = data.gltf_model.textures.AOMetalRoughness.image;
                image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                image_view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                image_view_info.subresourceRange.levelCount = 1;
                image_view_info.subresourceRange.layerCount = 1;

                if (init.disp.createImageView(&image_view_info, nullptr, &data.gltf_model.textures.AOMetalRoughness.image_view) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.AOMetalRoughness.image_view\n";
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

                if (init.disp.createSampler(&sampler_info, nullptr, &data.gltf_model.textures.AOMetalRoughness.sampler) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.AOMetalRoughness.sampler\n";
                    return -1;
                }

                data.gltf_model.textures.AOMetalRoughness.descriptor_image_info.sampler = data.gltf_model.textures.AOMetalRoughness.sampler;
                data.gltf_model.textures.AOMetalRoughness.descriptor_image_info.imageView = data.gltf_model.textures.AOMetalRoughness.image_view;
                data.gltf_model.textures.AOMetalRoughness.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            // normal img
            {
                VmaAllocationCreateInfo normal_img_vma_alloc_info{};
                normal_img_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
                normal_img_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

                VkImageCreateInfo normal_img_img_info{};
                normal_img_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                normal_img_img_info.imageType = VK_IMAGE_TYPE_2D;
                normal_img_img_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                normal_img_img_info.extent.width = normalImg.width;
                normal_img_img_info.extent.height = normalImg.height;
                normal_img_img_info.extent.depth = 1;
                normal_img_img_info.mipLevels = 1;
                normal_img_img_info.arrayLayers = 1;
                normal_img_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
                normal_img_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                normal_img_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                if (vmaCreateImage(init.allocator, &normal_img_img_info, &normal_img_vma_alloc_info, &data.gltf_model.textures.normal.image, &data.gltf_model.textures.normal.vma_allocation, nullptr) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.normal.image\n";
                    return -1;
                }

                // staging buffer copy
                {
                    const size_t buffer_size = 4 * sizeof(uint8_t) * normalImg.width * normalImg.height;
                    VkBuffer staging_buffer;
                    VmaAllocation staging_buffer_allocation;
                    VmaAllocationInfo vma_alloc_info;

                    VkBufferCreateInfo staging_buffer_alloc_info{};
                    staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    staging_buffer_alloc_info.size = buffer_size;;
                    staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

                    VmaAllocationCreateInfo staging_alloc_info{};
                    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                    if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
                        std::cout << "failed to staging buffer\n";
                        return -1; // failed to create vertex buffer
                    }

                    /* copy data to staging buffer*/
                    memcpy(vma_alloc_info.pMappedData, normalImg.image.data(), vma_alloc_info.size);

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
                    VkImageMemoryBarrier undefToDstBarrier{};
                    undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    undefToDstBarrier.image = data.gltf_model.textures.normal.image;
                    undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    undefToDstBarrier.subresourceRange.baseMipLevel = 0;
                    undefToDstBarrier.subresourceRange.levelCount = 1;
                    undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
                    undefToDstBarrier.subresourceRange.layerCount = 1;
                    undefToDstBarrier.srcAccessMask = 0;
                    undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &undefToDstBarrier);

                    VkBufferImageCopy copyBufferToImage{};
                    copyBufferToImage.bufferRowLength = normalImg.width;
                    copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    copyBufferToImage.imageSubresource.mipLevel = 0;
                    copyBufferToImage.imageSubresource.baseArrayLayer = 0;
                    copyBufferToImage.imageSubresource.layerCount = 1;
                    copyBufferToImage.imageExtent.width = normalImg.width;
                    copyBufferToImage.imageExtent.height = normalImg.height;
                    copyBufferToImage.imageExtent.depth = 1;
                    init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.gltf_model.textures.normal.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

                    // Transform the layout of the image to shader access resource
                    VkImageMemoryBarrier dstToShaderReadOptBarrier{};
                    dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    dstToShaderReadOptBarrier.image = data.gltf_model.textures.normal.image;
                    dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
                    dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
                    dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
                    dstToShaderReadOptBarrier.subresourceRange.layerCount = 1;
                    dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
                    dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &dstToShaderReadOptBarrier);

                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &commandBuffer;

                    init.disp.endCommandBuffer(commandBuffer);

                    init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
                    init.disp.queueWaitIdle(data.graphics_queue);

                    init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

                    vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
                }

                VkImageViewCreateInfo image_view_info{};
                image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                image_view_info.image = data.gltf_model.textures.normal.image;
                image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                image_view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                image_view_info.subresourceRange.levelCount = 1;
                image_view_info.subresourceRange.layerCount = 1;

                if (init.disp.createImageView(&image_view_info, nullptr, &data.gltf_model.textures.normal.image_view) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.normal.image_view\n";
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

                if (init.disp.createSampler(&sampler_info, nullptr, &data.gltf_model.textures.normal.sampler) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.normal.sampler\n";
                    return -1;
                }

                data.gltf_model.textures.normal.descriptor_image_info.sampler = data.gltf_model.textures.normal.sampler;
                data.gltf_model.textures.normal.descriptor_image_info.imageView = data.gltf_model.textures.normal.image_view;
                data.gltf_model.textures.normal.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            // emissive img
            {
                VmaAllocationCreateInfo emissive_img_vma_alloc_info{};
                emissive_img_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
                emissive_img_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

                VkImageCreateInfo emissive_img_img_info{};
                emissive_img_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                emissive_img_img_info.imageType = VK_IMAGE_TYPE_2D;
                emissive_img_img_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                emissive_img_img_info.extent.width = emissiveImg.width;
                emissive_img_img_info.extent.height = emissiveImg.height;
                emissive_img_img_info.extent.depth = 1;
                emissive_img_img_info.mipLevels = 1;
                emissive_img_img_info.arrayLayers = 1;
                emissive_img_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
                emissive_img_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                emissive_img_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

                if (vmaCreateImage(init.allocator, &emissive_img_img_info, &emissive_img_vma_alloc_info, &data.gltf_model.textures.emissive.image, &data.gltf_model.textures.emissive.vma_allocation, nullptr) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.emissive.image\n";
                    return -1;
                }

                // staging buffer copy
                {
                    const size_t buffer_size = 4 * sizeof(uint8_t) * emissiveImg.width * emissiveImg.height;
                    VkBuffer staging_buffer;
                    VmaAllocation staging_buffer_allocation;
                    VmaAllocationInfo vma_alloc_info;

                    VkBufferCreateInfo staging_buffer_alloc_info{};
                    staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    staging_buffer_alloc_info.size = buffer_size;;
                    staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

                    VmaAllocationCreateInfo staging_alloc_info{};
                    staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                    staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                    if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
                        std::cout << "failed to staging buffer\n";
                        return -1; // failed to create vertex buffer
                    }

                    /* copy data to staging buffer*/
                    memcpy(vma_alloc_info.pMappedData, emissiveImg.image.data(), vma_alloc_info.size);

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
                    VkImageMemoryBarrier undefToDstBarrier{};
                    undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    undefToDstBarrier.image = data.gltf_model.textures.emissive.image;
                    undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    undefToDstBarrier.subresourceRange.baseMipLevel = 0;
                    undefToDstBarrier.subresourceRange.levelCount = 1;
                    undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
                    undefToDstBarrier.subresourceRange.layerCount = 1;
                    undefToDstBarrier.srcAccessMask = 0;
                    undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &undefToDstBarrier);

                    VkBufferImageCopy copyBufferToImage{};
                    copyBufferToImage.bufferRowLength = emissiveImg.width;
                    copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    copyBufferToImage.imageSubresource.mipLevel = 0;
                    copyBufferToImage.imageSubresource.baseArrayLayer = 0;
                    copyBufferToImage.imageSubresource.layerCount = 1;
                    copyBufferToImage.imageExtent.width = emissiveImg.width;
                    copyBufferToImage.imageExtent.height = emissiveImg.height;
                    copyBufferToImage.imageExtent.depth = 1;
                    init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.gltf_model.textures.emissive.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

                    // Transform the layout of the image to shader access resource
                    VkImageMemoryBarrier dstToShaderReadOptBarrier{};
                    dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    dstToShaderReadOptBarrier.image = data.gltf_model.textures.emissive.image;
                    dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
                    dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
                    dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
                    dstToShaderReadOptBarrier.subresourceRange.layerCount = 1;
                    dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
                    dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &dstToShaderReadOptBarrier);

                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &commandBuffer;

                    init.disp.endCommandBuffer(commandBuffer);

                    init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
                    init.disp.queueWaitIdle(data.graphics_queue);

                    init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

                    vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
                }

                VkImageViewCreateInfo image_view_info{};
                image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                image_view_info.image = data.gltf_model.textures.emissive.image;
                image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                image_view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
                image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                image_view_info.subresourceRange.levelCount = 1;
                image_view_info.subresourceRange.layerCount = 1;

                if (init.disp.createImageView(&image_view_info, nullptr, &data.gltf_model.textures.emissive.image_view) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.emissive.image_view\n";
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

                if (init.disp.createSampler(&sampler_info, nullptr, &data.gltf_model.textures.emissive.sampler) != VK_SUCCESS) {
                    std::cout << "failed to create data.gltf_model.textures.emissive.sampler\n";
                    return -1;
                }

                data.gltf_model.textures.emissive.descriptor_image_info.sampler = data.gltf_model.textures.emissive.sampler;
                data.gltf_model.textures.emissive.descriptor_image_info.imageView = data.gltf_model.textures.emissive.image_view;
                data.gltf_model.textures.emissive.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
    }

    return 0;
}

int create_vertex_buffer(Init& init, RenderData& data) {
    const uint32_t buffer_size = sizeof(data.gltf_model.vertices[0]) * data.gltf_model.vertices.size();
    /* vertex buffer */
    VkBufferCreateInfo vertex_buffer_info{};
    vertex_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vertex_buffer_info.size = buffer_size;
    vertex_buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vertex_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vertex_buffer_alloc_info = {};
    vertex_buffer_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateBuffer(init.allocator, &vertex_buffer_info, &vertex_buffer_alloc_info, &data.gltf_model.vertex_buffer, &data.gltf_model.vertex_buffer_allocation, nullptr) != VK_SUCCESS) {
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
    memcpy(vma_vertex_buffer_alloc_info.pMappedData, data.gltf_model.vertices.data(), vertex_buffer_info.size);

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
    init.disp.cmdCopyBuffer(commandBuffer, vertex_buffer_staging_buffer, data.gltf_model.vertex_buffer, 1, &copyRegion);

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
    const uint32_t buffer_size = sizeof(data.gltf_model.indices[0]) * data.gltf_model.indices.size();
    /* index buffer */
    VkBufferCreateInfo index_buffer_info{};
    index_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    index_buffer_info.size = buffer_size;
    index_buffer_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    index_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo index_buffer_alloc_info = {};
    index_buffer_alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateBuffer(init.allocator, &index_buffer_info, &index_buffer_alloc_info, &data.gltf_model.index_buffer, &data.gltf_model.index_buffer_allocation, nullptr) != VK_SUCCESS) {
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
    memcpy(vma_index_buffer_alloc_info.pMappedData, data.gltf_model.indices.data(), index_buffer_info.size);

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
    init.disp.cmdCopyBuffer(commandBuffer, index_buffer_staging_buffer, data.gltf_model.index_buffer, 1, &copyRegion);

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
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

    if (init.disp.createDescriptorPool(&pool_info, nullptr, &data.descriptor_pool) != VK_SUCCESS ) {
        std::cout << "failed to create descriptor pool\n";
        return -1;
    }

    return 0;
}

int create_skybox_descriptor_set_layout(Init& init, RenderData& data) {
    VkDescriptorSetLayoutBinding skybox_layout_binding{};
    skybox_layout_binding.binding = 1;
    skybox_layout_binding.descriptorCount = bindless_img_count;
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

    if (init.disp.createDescriptorSetLayout(&layout_info, nullptr, &data.skybox_pipeline.descriptor_set_layout) != VK_SUCCESS) {
        std::cout <<"failed to create skybox descriptor set layout!\n";
        return -1;
    }
    return 0;
}

int create_spheres_descriptor_set_layout(Init& init, RenderData& data) {
    VkDescriptorSetLayoutBinding spheres_layout_binding{};
    spheres_layout_binding.binding = 1;
    spheres_layout_binding.descriptorCount = bindless_img_count;
    spheres_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    spheres_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    spheres_layout_binding.pImmutableSamplers = nullptr; // Optional

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
    layout_info.pBindings = &spheres_layout_binding;
    // layout_info.pNext = &set_layout_binding_flags;

    if (init.disp.createDescriptorSetLayout(&layout_info, nullptr, &data.spheres_pipeline.descriptor_set_layout) != VK_SUCCESS) {
        std::cout <<"failed to create spheres descriptor set layout!\n";
        return -1;
    }
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

int load_hdr_cubemap(Init& init, RenderData& data, const std::string& img_file_path) {
    int width, height, nrComponents;
    void* img_data = stbi_loadf(img_file_path.c_str(), &width, &height, &nrComponents, 4);

    VmaAllocationCreateInfo hdr_cubemap_vma_alloc_info{};
    hdr_cubemap_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    hdr_cubemap_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImageCreateInfo hdr_cubemap_img_info{};
    hdr_cubemap_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    hdr_cubemap_img_info.imageType = VK_IMAGE_TYPE_2D;
    hdr_cubemap_img_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    hdr_cubemap_img_info.extent.width = width;
    hdr_cubemap_img_info.extent.height = height / 6;
    hdr_cubemap_img_info.extent.depth = 1;
    hdr_cubemap_img_info.mipLevels = 1;
    hdr_cubemap_img_info.arrayLayers = 6;
    hdr_cubemap_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    // hdr_cubemap_img_info.tiling = VK_IMAGE_TILING_LINEAR;
    hdr_cubemap_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    hdr_cubemap_img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    hdr_cubemap_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vmaCreateImage(init.allocator, &hdr_cubemap_img_info, &hdr_cubemap_vma_alloc_info, &data.hdr_cubemap.image, &data.hdr_cubemap.vma_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create hdr_cubemap.image\n";
        return -1;
    }

    // staging buffer copy
    {
        const size_t buffer_size = 4 * sizeof(float) * width * height;
        VkBuffer staging_buffer;
        VmaAllocation staging_buffer_allocation;
        VmaAllocationInfo vma_alloc_info;

        VkBufferCreateInfo staging_buffer_alloc_info{};
        staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_alloc_info.size = buffer_size;;
        staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo staging_alloc_info{};
        staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
            std::cout << "failed to staging buffer\n";
            return -1; // failed to create vertex buffer
        }

        /* copy data to staging buffer*/
        memcpy(vma_alloc_info.pMappedData, img_data, vma_alloc_info.size);

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
        VkImageMemoryBarrier undefToDstBarrier{};
        undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        undefToDstBarrier.image = data.hdr_cubemap.image;
        undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        undefToDstBarrier.subresourceRange.baseMipLevel = 0;
        undefToDstBarrier.subresourceRange.levelCount = 1;
        undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
        undefToDstBarrier.subresourceRange.layerCount = 6;
        undefToDstBarrier.srcAccessMask = 0;
        undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &undefToDstBarrier);

        VkBufferImageCopy copyBufferToImage{};
        copyBufferToImage.bufferRowLength = width;
        copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyBufferToImage.imageSubresource.mipLevel = 0;
        copyBufferToImage.imageSubresource.baseArrayLayer = 0;
        copyBufferToImage.imageSubresource.layerCount = 6;
        copyBufferToImage.imageExtent.width = width;
        copyBufferToImage.imageExtent.height = height / 6;
        copyBufferToImage.imageExtent.depth = 1;
        init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.hdr_cubemap.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

        // Transform the layout of the image to shader access resource
        VkImageMemoryBarrier dstToShaderReadOptBarrier{};
        dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dstToShaderReadOptBarrier.image = data.hdr_cubemap.image;
        dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
        dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
        dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
        dstToShaderReadOptBarrier.subresourceRange.layerCount = 6;
        dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
        dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &dstToShaderReadOptBarrier);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &commandBuffer;

        init.disp.endCommandBuffer(commandBuffer);

        init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        init.disp.queueWaitIdle(data.graphics_queue);

        init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

        vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
    }

    VkImageViewCreateInfo image_view_info{};
    image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_info.image = data.hdr_cubemap.image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    image_view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 6;

    if (init.disp.createImageView(&image_view_info, nullptr, &data.hdr_cubemap.image_view) != VK_SUCCESS) {
        std::cout << "failed to create hdr_cubemap.image_view\n";
        return -1;
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.minLod = -1000;
    sampler_info.maxLod = 1000;
    sampler_info.maxAnisotropy = 1.0f;

    if (init.disp.createSampler(&sampler_info, nullptr, &data.hdr_cubemap.sampler) != VK_SUCCESS) {
        std::cout << "failed to create hdr_cubemap.sampler\n";
        return -1;
    }

    free(img_data);

    data.hdr_cubemap.descriptor_image_info.sampler = data.hdr_cubemap.sampler;
    data.hdr_cubemap.descriptor_image_info.imageView = data.hdr_cubemap.image_view;
    data.hdr_cubemap.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    return 0;
}

int load_diffuse_irradiance_cubemap(Init& init, RenderData& data, const std::string& img_file_path) {
    int width, height, nrComponents;
    void* img_data = stbi_loadf(img_file_path.c_str(), &width, &height, &nrComponents, 4);

    VmaAllocationCreateInfo diffuse_irradiance_cubemap_vma_alloc_info{};
    diffuse_irradiance_cubemap_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    diffuse_irradiance_cubemap_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImageCreateInfo diffuse_irradiance_cubemap_img_info{};
    diffuse_irradiance_cubemap_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    diffuse_irradiance_cubemap_img_info.imageType = VK_IMAGE_TYPE_2D;
    diffuse_irradiance_cubemap_img_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    diffuse_irradiance_cubemap_img_info.extent.width = width;
    diffuse_irradiance_cubemap_img_info.extent.height = height / 6;
    diffuse_irradiance_cubemap_img_info.extent.depth = 1;
    diffuse_irradiance_cubemap_img_info.mipLevels = 1;
    diffuse_irradiance_cubemap_img_info.arrayLayers = 6;
    diffuse_irradiance_cubemap_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    // diffuse_irradiance_cubemap_img_info.tiling = VK_IMAGE_TILING_LINEAR;
    diffuse_irradiance_cubemap_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    diffuse_irradiance_cubemap_img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    diffuse_irradiance_cubemap_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vmaCreateImage(init.allocator, &diffuse_irradiance_cubemap_img_info, &diffuse_irradiance_cubemap_vma_alloc_info, &data.diffuse_irradiance_cubemap.image, &data.diffuse_irradiance_cubemap.vma_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create diffuse_irradiance_cubemap.image\n";
        return -1;
    }

    // staging buffer copy
    {
        const size_t buffer_size = 4 * sizeof(float) * width * height;
        VkBuffer staging_buffer;
        VmaAllocation staging_buffer_allocation;
        VmaAllocationInfo vma_alloc_info;

        VkBufferCreateInfo staging_buffer_alloc_info{};
        staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_alloc_info.size = buffer_size;;
        staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo staging_alloc_info{};
        staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
            std::cout << "failed to staging buffer\n";
            return -1; // failed to create vertex buffer
        }

        /* copy data to staging buffer*/
        memcpy(vma_alloc_info.pMappedData, img_data, vma_alloc_info.size);

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
        VkImageMemoryBarrier undefToDstBarrier{};
        undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        undefToDstBarrier.image = data.diffuse_irradiance_cubemap.image;
        undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        undefToDstBarrier.subresourceRange.baseMipLevel = 0;
        undefToDstBarrier.subresourceRange.levelCount = 1;
        undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
        undefToDstBarrier.subresourceRange.layerCount = 6;
        undefToDstBarrier.srcAccessMask = 0;
        undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &undefToDstBarrier);

        VkBufferImageCopy copyBufferToImage{};
        copyBufferToImage.bufferRowLength = width;
        copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyBufferToImage.imageSubresource.mipLevel = 0;
        copyBufferToImage.imageSubresource.baseArrayLayer = 0;
        copyBufferToImage.imageSubresource.layerCount = 6;
        copyBufferToImage.imageExtent.width = width;
        copyBufferToImage.imageExtent.height = height / 6;
        copyBufferToImage.imageExtent.depth = 1;
        init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.diffuse_irradiance_cubemap.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

        // Transform the layout of the image to shader access resource
        VkImageMemoryBarrier dstToShaderReadOptBarrier{};
        dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dstToShaderReadOptBarrier.image = data.diffuse_irradiance_cubemap.image;
        dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
        dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
        dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
        dstToShaderReadOptBarrier.subresourceRange.layerCount = 6;
        dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
        dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &dstToShaderReadOptBarrier);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &commandBuffer;

        init.disp.endCommandBuffer(commandBuffer);

        init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        init.disp.queueWaitIdle(data.graphics_queue);

        init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

        vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
    }

    VkImageViewCreateInfo image_view_info{};
    image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_info.image = data.diffuse_irradiance_cubemap.image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    image_view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 6;

    if (init.disp.createImageView(&image_view_info, nullptr, &data.diffuse_irradiance_cubemap.image_view) != VK_SUCCESS) {
        std::cout << "failed to create diffuse_irradiance_cubemap.image_view\n";
        return -1;
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.minLod = -1000;
    sampler_info.maxLod = 1000;
    sampler_info.maxAnisotropy = 1.0f;

    if (init.disp.createSampler(&sampler_info, nullptr, &data.diffuse_irradiance_cubemap.sampler) != VK_SUCCESS) {
        std::cout << "failed to create diffuse_irradiance_cubemap.sampler\n";
        return -1;
    }

    free(img_data);

    data.diffuse_irradiance_cubemap.descriptor_image_info.sampler = data.diffuse_irradiance_cubemap.sampler;
    data.diffuse_irradiance_cubemap.descriptor_image_info.imageView = data.diffuse_irradiance_cubemap.image_view;
    data.diffuse_irradiance_cubemap.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    return 0;
}

int load_prefilter_environment_cubemap(Init& init, RenderData& data, const std::string& img_file_base_path) {
    const int mip_count = 8;

    struct ImgDataWH {
        void* data;
        uint32_t width;
        uint32_t height;

        VkBuffer staging_buffer;
        VmaAllocation staging_buffer_allocation;
        VmaAllocationInfo vma_alloc_info;
    };

    std::vector<ImgDataWH> img_infos;
    img_infos.resize(mip_count);

    for (uint32_t i = 0; i < mip_count; i++) {
        int width, height, nrComponents;
        std::string img_file_path = img_file_base_path + "prefilterMip" + std::to_string(i) + ".hdr";
        img_infos[i].data = stbi_loadf(img_file_path.c_str(), &width, &height, &nrComponents, 4);
        img_infos[i].width = width;
        img_infos[i].height = height;
    }

    VmaAllocationCreateInfo prefilter_environment_cubemap_vma_alloc_info{};
    prefilter_environment_cubemap_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    prefilter_environment_cubemap_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImageCreateInfo prefilter_environment_cubemap_img_info{};
    prefilter_environment_cubemap_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    prefilter_environment_cubemap_img_info.imageType = VK_IMAGE_TYPE_2D;
    prefilter_environment_cubemap_img_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    prefilter_environment_cubemap_img_info.extent.width = img_infos[0].width;
    prefilter_environment_cubemap_img_info.extent.height = img_infos[0].height / 6;
    prefilter_environment_cubemap_img_info.extent.depth = 1;
    prefilter_environment_cubemap_img_info.mipLevels = mip_count;
    prefilter_environment_cubemap_img_info.arrayLayers = 6;
    prefilter_environment_cubemap_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    // prefilter_environment_cubemap_img_info.tiling = VK_IMAGE_TILING_LINEAR;
    prefilter_environment_cubemap_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    prefilter_environment_cubemap_img_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    prefilter_environment_cubemap_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vmaCreateImage(init.allocator, &prefilter_environment_cubemap_img_info, &prefilter_environment_cubemap_vma_alloc_info, &data.prefilter_env_cubemap.image, &data.prefilter_env_cubemap.vma_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create prefilter_env_cubemap.image\n";
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



        // copy mips
        {
            for (uint32_t i = 0; i < mip_count; i++) {
                {
                    // Transform the layout of the image to copy source
                    VkImageMemoryBarrier undefToDstBarrier{};
                    undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    undefToDstBarrier.image = data.prefilter_env_cubemap.image;
                    undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    undefToDstBarrier.subresourceRange.baseMipLevel = i;
                    undefToDstBarrier.subresourceRange.levelCount = 1;
                    undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
                    undefToDstBarrier.subresourceRange.layerCount = 6;
                    undefToDstBarrier.srcAccessMask = 0;
                    undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &undefToDstBarrier);
                }
                const size_t buffer_size = 4 * sizeof(float) * img_infos[i].width * img_infos[i].height;

                VkBufferCreateInfo staging_buffer_alloc_info{};
                staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                staging_buffer_alloc_info.size = buffer_size;;
                staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

                VmaAllocationCreateInfo staging_alloc_info{};
                staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &img_infos[i].staging_buffer, &img_infos[i].staging_buffer_allocation, &img_infos[i].vma_alloc_info) != VK_SUCCESS) {
                    std::cout << "failed to staging buffer\n";
                    return -1; // failed to create vertex buffer
                }

                /* copy data to staging buffer*/
                memcpy(img_infos[i].vma_alloc_info.pMappedData, img_infos[i].data, img_infos[i].vma_alloc_info.size);

                VkBufferImageCopy copyBufferToImage{};
                copyBufferToImage.bufferRowLength = img_infos[i].width;
                copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyBufferToImage.imageSubresource.mipLevel = i;
                copyBufferToImage.imageSubresource.baseArrayLayer = 0;
                copyBufferToImage.imageSubresource.layerCount = 6;
                copyBufferToImage.imageExtent.width = img_infos[i].width;
                copyBufferToImage.imageExtent.height = img_infos[i].height / 6;
                copyBufferToImage.imageExtent.depth = 1;
                init.disp.cmdCopyBufferToImage(commandBuffer, img_infos[i].staging_buffer, data.prefilter_env_cubemap.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

                {
                    // Transform the layout of the image to shader access resource
                    VkImageMemoryBarrier dstToShaderReadOptBarrier{};
                    dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    dstToShaderReadOptBarrier.image = data.prefilter_env_cubemap.image;
                    dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = i;
                    dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
                    dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
                    dstToShaderReadOptBarrier.subresourceRange.layerCount = 6;
                    dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
                    dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                    init.disp.cmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0,
                        0, nullptr,
                        0, nullptr,
                        1, &dstToShaderReadOptBarrier);
                }
            }
        }



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
    image_view_info.image = data.prefilter_env_cubemap.image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    image_view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = mip_count;
    image_view_info.subresourceRange.layerCount = 6;

    if (init.disp.createImageView(&image_view_info, nullptr, &data.prefilter_env_cubemap.image_view) != VK_SUCCESS) {
        std::cout << "failed to create prefilter_env_cubemap.image_view\n";
        return -1;
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.minLod = -1000;
    sampler_info.maxLod = 1000;
    sampler_info.maxAnisotropy = 1.0f;

    if (init.disp.createSampler(&sampler_info, nullptr, &data.prefilter_env_cubemap.sampler) != VK_SUCCESS) {
        std::cout << "failed to create prefilter_env_cubemap.sampler\n";
        return -1;
    }

    for (uint32_t i = 0; i < mip_count; i++) {
        free(img_infos[i].data);
        vmaDestroyBuffer(init.allocator, img_infos[i].staging_buffer, img_infos[i].staging_buffer_allocation);
    }

    data.prefilter_env_cubemap.descriptor_image_info.sampler = data.prefilter_env_cubemap.sampler;
    data.prefilter_env_cubemap.descriptor_image_info.imageView = data.prefilter_env_cubemap.image_view;
    data.prefilter_env_cubemap.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    return 0;
}

int load_environment_brdf_map(Init& init, RenderData& data, const std::string& img_file_path) {
    int width, height, nrComponents;
    void* img_data = stbi_loadf(img_file_path.c_str(), &width, &height, &nrComponents, 4);

    VmaAllocationCreateInfo envBrdf_img_vma_alloc_info{};
    envBrdf_img_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    envBrdf_img_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImageCreateInfo envBrdf_img_img_info{};
    envBrdf_img_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    envBrdf_img_img_info.imageType = VK_IMAGE_TYPE_2D;
    envBrdf_img_img_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    envBrdf_img_img_info.extent.width = width;
    envBrdf_img_img_info.extent.height = height;
    envBrdf_img_img_info.extent.depth = 1;
    envBrdf_img_img_info.mipLevels = 1;
    envBrdf_img_img_info.arrayLayers = 1;
    envBrdf_img_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    // envBrdf_img_img_info.tiling = VK_IMAGE_TILING_LINEAR;
    envBrdf_img_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    envBrdf_img_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vmaCreateImage(init.allocator, &envBrdf_img_img_info, &envBrdf_img_vma_alloc_info, &data.envBrdf_img.image, &data.envBrdf_img.vma_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create envBrdf_img.image\n";
        return -1;
    }

    // staging buffer copy
    {
        const size_t buffer_size = 4 * sizeof(float) * width * height;
        VkBuffer staging_buffer;
        VmaAllocation staging_buffer_allocation;
        VmaAllocationInfo vma_alloc_info;

        VkBufferCreateInfo staging_buffer_alloc_info{};
        staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_alloc_info.size = buffer_size;;
        staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo staging_alloc_info{};
        staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
            std::cout << "failed to staging buffer\n";
            return -1; // failed to create vertex buffer
        }

        /* copy data to staging buffer*/
        memcpy(vma_alloc_info.pMappedData, img_data, vma_alloc_info.size);

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
        VkImageMemoryBarrier undefToDstBarrier{};
        undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        undefToDstBarrier.image = data.envBrdf_img.image;
        undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        undefToDstBarrier.subresourceRange.baseMipLevel = 0;
        undefToDstBarrier.subresourceRange.levelCount = 1;
        undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
        undefToDstBarrier.subresourceRange.layerCount = 1;
        undefToDstBarrier.srcAccessMask = 0;
        undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &undefToDstBarrier);

        VkBufferImageCopy copyBufferToImage{};
        copyBufferToImage.bufferRowLength = width;
        copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyBufferToImage.imageSubresource.mipLevel = 0;
        copyBufferToImage.imageSubresource.baseArrayLayer = 0;
        copyBufferToImage.imageSubresource.layerCount = 1;
        copyBufferToImage.imageExtent.width = width;
        copyBufferToImage.imageExtent.height = height;
        copyBufferToImage.imageExtent.depth = 1;
        init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.envBrdf_img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

        // Transform the layout of the image to shader access resource
        VkImageMemoryBarrier dstToShaderReadOptBarrier{};
        dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dstToShaderReadOptBarrier.image = data.envBrdf_img.image;
        dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
        dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
        dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
        dstToShaderReadOptBarrier.subresourceRange.layerCount = 1;
        dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
        dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &dstToShaderReadOptBarrier);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &commandBuffer;

        init.disp.endCommandBuffer(commandBuffer);

        init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        init.disp.queueWaitIdle(data.graphics_queue);

        init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

        vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
    }

    VkImageViewCreateInfo image_view_info{};
    image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_info.image = data.envBrdf_img.image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 1;

    if (init.disp.createImageView(&image_view_info, nullptr, &data.envBrdf_img.image_view) != VK_SUCCESS) {
        std::cout << "failed to create envBrdf_img.image_view\n";
        return -1;
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.minLod = -1000;
    sampler_info.maxLod = 1000;
    sampler_info.maxAnisotropy = 1.0f;

    if (init.disp.createSampler(&sampler_info, nullptr, &data.envBrdf_img.sampler) != VK_SUCCESS) {
        std::cout << "failed to create envBrdf_img.sampler\n";
        return -1;
    }

    free(img_data);

    data.envBrdf_img.descriptor_image_info.sampler = data.envBrdf_img.sampler;
    data.envBrdf_img.descriptor_image_info.imageView = data.envBrdf_img.image_view;
    data.envBrdf_img.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    return 0;
}

int create_dummy_img(Init& init, RenderData& data) {
    constexpr int width = 4, height = 4;
    uint8_t img_data[16 * 4] = {
        0,0,0,255,        255,255,255,255,        0,0,0,255,  255,255,255,255,
        255,255,255,255,        0,0,0,255,  255,255,255,255,        0,0,0,255,
        0,0,0,255,        255,255,255,255,        0,0,0,255,  255,255,255,255,
        255,255,255,255,        0,0,0,255,  255,255,255,255,        0,0,0,255,
    };

    VmaAllocationCreateInfo dummy_img_vma_alloc_info{};
    dummy_img_vma_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    dummy_img_vma_alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImageCreateInfo dummy_img_img_info{};
    dummy_img_img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    dummy_img_img_info.imageType = VK_IMAGE_TYPE_2D;
    dummy_img_img_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    dummy_img_img_info.extent.width = width;
    dummy_img_img_info.extent.height = height;
    dummy_img_img_info.extent.depth = 1;
    dummy_img_img_info.mipLevels = 1;
    dummy_img_img_info.arrayLayers = 1;
    dummy_img_img_info.samples = VK_SAMPLE_COUNT_1_BIT;
    dummy_img_img_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    dummy_img_img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vmaCreateImage(init.allocator, &dummy_img_img_info, &dummy_img_vma_alloc_info, &data.dummy_img.image, &data.dummy_img.vma_allocation, nullptr) != VK_SUCCESS) {
        std::cout << "failed to create dummy_img.image\n";
        return -1;
    }

    // staging buffer copy
    {
        const size_t buffer_size = 4 * sizeof(uint8_t) * width * height;
        VkBuffer staging_buffer;
        VmaAllocation staging_buffer_allocation;
        VmaAllocationInfo vma_alloc_info;

        VkBufferCreateInfo staging_buffer_alloc_info{};
        staging_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_alloc_info.size = buffer_size;;
        staging_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo staging_alloc_info{};
        staging_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(init.allocator, &staging_buffer_alloc_info, &staging_alloc_info, &staging_buffer, &staging_buffer_allocation, &vma_alloc_info) != VK_SUCCESS) {
            std::cout << "failed to staging buffer\n";
            return -1; // failed to create vertex buffer
        }

        /* copy data to staging buffer*/
        memcpy(vma_alloc_info.pMappedData, img_data, vma_alloc_info.size);

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
        VkImageMemoryBarrier undefToDstBarrier{};
        undefToDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        undefToDstBarrier.image = data.dummy_img.image;
        undefToDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        undefToDstBarrier.subresourceRange.baseMipLevel = 0;
        undefToDstBarrier.subresourceRange.levelCount = 1;
        undefToDstBarrier.subresourceRange.baseArrayLayer = 0;
        undefToDstBarrier.subresourceRange.layerCount = 1;
        undefToDstBarrier.srcAccessMask = 0;
        undefToDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        undefToDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        undefToDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &undefToDstBarrier);

        VkBufferImageCopy copyBufferToImage{};
        copyBufferToImage.bufferRowLength = width;
        copyBufferToImage.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyBufferToImage.imageSubresource.mipLevel = 0;
        copyBufferToImage.imageSubresource.baseArrayLayer = 0;
        copyBufferToImage.imageSubresource.layerCount = 1;
        copyBufferToImage.imageExtent.width = width;
        copyBufferToImage.imageExtent.height = height;
        copyBufferToImage.imageExtent.depth = 1;
        init.disp.cmdCopyBufferToImage(commandBuffer, staging_buffer, data.dummy_img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyBufferToImage);

        // Transform the layout of the image to shader access resource
        VkImageMemoryBarrier dstToShaderReadOptBarrier{};
        dstToShaderReadOptBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        dstToShaderReadOptBarrier.image = data.dummy_img.image;
        dstToShaderReadOptBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        dstToShaderReadOptBarrier.subresourceRange.baseMipLevel = 0;
        dstToShaderReadOptBarrier.subresourceRange.levelCount = 1;
        dstToShaderReadOptBarrier.subresourceRange.baseArrayLayer = 0;
        dstToShaderReadOptBarrier.subresourceRange.layerCount = 1;
        dstToShaderReadOptBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstToShaderReadOptBarrier.dstAccessMask = VK_ACCESS_NONE;
        dstToShaderReadOptBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dstToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        init.disp.cmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &dstToShaderReadOptBarrier);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &commandBuffer;

        init.disp.endCommandBuffer(commandBuffer);

        init.disp.queueSubmit(data.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        init.disp.queueWaitIdle(data.graphics_queue);

        init.disp.freeCommandBuffers(data.command_pool, 1, &commandBuffer);

        vmaDestroyBuffer(init.allocator, staging_buffer, staging_buffer_allocation);
    }

    VkImageViewCreateInfo image_view_info{};
    image_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_info.image = data.dummy_img.image;
    image_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
    image_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_info.subresourceRange.levelCount = 1;
    image_view_info.subresourceRange.layerCount = 1;

    if (init.disp.createImageView(&image_view_info, nullptr, &data.dummy_img.image_view) != VK_SUCCESS) {
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


    if (init.disp.createSampler(&sampler_info, nullptr, &data.dummy_img.sampler) != VK_SUCCESS) {
        std::cout << "failed to create dummy_img.sampler\n";
        return -1;
    }

    data.dummy_img.descriptor_image_info.sampler = data.dummy_img.sampler;
    data.dummy_img.descriptor_image_info.imageView = data.dummy_img.image_view;
    data.dummy_img.descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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
        debug_utils_label.color[0] = 0.3f;
        debug_utils_label.color[1] = 0.0f;
        debug_utils_label.color[2] = 0.7f;
        debug_utils_label.color[3] = 1.0f;

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

        init.disp.cmdSetViewport(data.command_buffers[i], 0, 1, &viewport);
        init.disp.cmdSetScissor(data.command_buffers[i], 0, 1, &scissor);

        std::vector<VkDescriptorImageInfo> texture_descriptors(bindless_img_count);
        for (size_t img_idx = 0; img_idx < bindless_img_count; img_idx++) {
            if (10 == img_idx) {texture_descriptors[img_idx] = data.hdr_cubemap.descriptor_image_info;                continue;}
            if (11 == img_idx) {texture_descriptors[img_idx] = data.diffuse_irradiance_cubemap.descriptor_image_info; continue;}
            if (12 == img_idx) {texture_descriptors[img_idx] = data.prefilter_env_cubemap.descriptor_image_info;      continue;}
            if (13 == img_idx) {texture_descriptors[img_idx] = data.envBrdf_img.descriptor_image_info;                continue;}

            if (20 == img_idx) {texture_descriptors[img_idx] = data.gltf_model.textures.albedo.descriptor_image_info;           continue;}
            if (21 == img_idx) {texture_descriptors[img_idx] = data.gltf_model.textures.AOMetalRoughness.descriptor_image_info; continue;}
            if (22 == img_idx) {texture_descriptors[img_idx] = data.gltf_model.textures.normal.descriptor_image_info;           continue;}
            if (23 == img_idx) {texture_descriptors[img_idx] = data.gltf_model.textures.emissive.descriptor_image_info;         continue;}

            texture_descriptors[img_idx] = data.dummy_img.descriptor_image_info;
        }

        VkWriteDescriptorSet bindless_write_descriptor_set{};
        bindless_write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bindless_write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bindless_write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindless_write_descriptor_set.dstBinding = 1;
        bindless_write_descriptor_set.descriptorCount = bindless_img_count;
        bindless_write_descriptor_set.pImageInfo = texture_descriptors.data();

        // skybox pass
        {
            debug_utils_label.pLabelName = "skybox pass";
            init.disp.cmdBeginDebugUtilsLabelEXT(data.command_buffers[i], &debug_utils_label);

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

            init.disp.cmdBeginRendering(data.command_buffers[i], &rendering_info);

            init.disp.cmdBindPipeline(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.skybox_pipeline.graphics_pipeline);

            SkyboxPixelShaderPushConstants constants{};
            constants.view = {1, 0, 0, 1};
            constants.up = {0, 1, 0, 1};
            constants.right = {0, 0, -1, 1};
            constants.viewportWidthHeight = {viewport.width, viewport.height};
            constants.camNear = 0.1f;
            const float fov = 70.f * M_PI / 180.f;
            const float aspect = 1700.f / 900.f;
            const float nearPlaneHeight = 2.f * constants.camNear * tanf(fov / 2.f);
            const float nearPlaneWidth  = aspect * nearPlaneHeight;
            constants.nearWidthHeight = {nearPlaneWidth, nearPlaneHeight};
            constants.texIdx[0] = 10;
            constants.texIdx[1] = 11;
            constants.texIdx[2] = 12;
            constants.texIdx[3] = 13;

            init.disp.cmdPushConstants(data.command_buffers[i], data.skybox_pipeline.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkyboxPixelShaderPushConstants), &constants);

            init.disp.cmdPushDescriptorSetKHR(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.skybox_pipeline.pipeline_layout, 0, 1, &bindless_write_descriptor_set);

            init.disp.cmdDraw(data.command_buffers[i], 6, 1, 0, 0);

            init.disp.cmdEndRendering(data.command_buffers[i]);

            init.disp.cmdEndDebugUtilsLabelEXT(data.command_buffers[i]);
        }











        // gltf pass
        {
            debug_utils_label.pLabelName = "gltf pass";
            init.disp.cmdBeginDebugUtilsLabelEXT(data.command_buffers[i], &debug_utils_label);

            VkRenderingAttachmentInfo color_attachment_info{};
            color_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment_info.imageView = data.swapchain_image_views[i];
            color_attachment_info.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            color_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            // color_attachment_info.clearValue = clearColor;

            VkRenderingAttachmentInfo depth_stencil_attachment_info{};
            depth_stencil_attachment_info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_stencil_attachment_info.imageView = data.depth_image_view;
            depth_stencil_attachment_info.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
            depth_stencil_attachment_info.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth_stencil_attachment_info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            // depth_stencil_attachment_info.clearValue = clearDepth;

            VkRenderingInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea = {{0, 0}, {init.swapchain.extent.width, init.swapchain.extent.height}};
            rendering_info.layerCount = 1;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_attachment_info;
            rendering_info.pDepthAttachment = &depth_stencil_attachment_info;

            init.disp.cmdBeginRendering(data.command_buffers[i], &rendering_info);

            init.disp.cmdBindPipeline(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.spheres_pipeline.graphics_pipeline);

            VkBuffer vertex_buffers[] = {data.gltf_model.vertex_buffer};
            VkDeviceSize offsets[] = {0};
            init.disp.cmdBindVertexBuffers(data.command_buffers[i], 0, 1, vertex_buffers, offsets);
            init.disp.cmdBindIndexBuffer(data.command_buffers[i], data.gltf_model.index_buffer, 0, VK_INDEX_TYPE_UINT16);

            HMM_Vec3 cam_pos = { 0.f,0.f,0.f };
            HMM_Mat4 view = HMM_LookAt_RH( cam_pos, {15, 0, 0}, {0, 1, 0});
            const float fov = 70.f;
            float aspect = 1700.f / 900.f;
            HMM_Mat4 projection = HMM_Perspective_RH_NO(fov * HMM_DegToRad, aspect, 0.1f, 200.0f);
            projection[1][1] *= -1;
            HMM_Mat4 model_matrix = HMM_Translate({3, 0, 0}) * HMM_Rotate_RH(data.number_of_frame * 0.3f * HMM_DegToRad, HMM_V3(0, 1, 0)) * HMM_Rotate_RH(90.0f * HMM_DegToRad, HMM_V3(1, 0, 0));
            HMM_Mat4 vp_matrix = projection * view;

            GltfVertexShaderPushConstants vs_constants{};
            vs_constants.m_matrix = model_matrix;
            vs_constants.vp_matrix = vp_matrix;

            init.disp.cmdPushConstants(data.command_buffers[i], data.spheres_pipeline.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GltfVertexShaderPushConstants), &vs_constants);

            GltfPixelShaderPushConstants ps_constants{};
            ps_constants.lightPositions[0] = {10.f,  3.f, -8.f, 0.0f,};
            ps_constants.lightPositions[1] = {10.f,  3.f,  8.f, 0.0f,};
            ps_constants.lightPositions[2] = {10.f, -3.f, -8.f, 0.0f,};
            ps_constants.lightPositions[3] = {10.f, -3.f,  8.f, 0.0f,};
            ps_constants.cameraPosition = {cam_pos.X, cam_pos.Y, cam_pos.Z, 0.0f,};
            ps_constants.params[0] = data.imgui_state.enable_light ? 1 : 0;
            ps_constants.params[1] = data.imgui_state.enable_ibl_diffuse ? 1 : 0;
            ps_constants.params[2] = data.imgui_state.enable_ibl_specular ? 1 : 0;
            ps_constants.params[3] = 8 - 1; // mip_count = 8, max mip level is 7
            ps_constants.texIdx[0] = 10;
            ps_constants.texIdx[1] = 11;
            ps_constants.texIdx[2] = 12;
            ps_constants.texIdx[3] = 13;
            ps_constants.gltfIdx[0] = 20;
            ps_constants.gltfIdx[1] = 21;
            ps_constants.gltfIdx[2] = 22;
            ps_constants.gltfIdx[3] = 23;

            init.disp.cmdPushConstants(data.command_buffers[i], data.spheres_pipeline.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(GltfVertexShaderPushConstants), sizeof(GltfPixelShaderPushConstants), &ps_constants);

            init.disp.cmdPushDescriptorSetKHR(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.spheres_pipeline.pipeline_layout, 0, 1, &bindless_write_descriptor_set);

            init.disp.cmdDrawIndexed(data.command_buffers[i], data.gltf_model.indices.size(), 1, 0, 0, 0);

            init.disp.cmdEndRendering(data.command_buffers[i]);

            init.disp.cmdEndDebugUtilsLabelEXT(data.command_buffers[i]);
        }

        // imgui pass
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
                ImGui::Checkbox("enable light", &data.imgui_state.enable_light);
                ImGui::Checkbox("enable ibl diffuse", &data.imgui_state.enable_ibl_diffuse);
                ImGui::Checkbox("enable ibl specular", &data.imgui_state.enable_ibl_specular);
            }
            ImGui::End();

            // Rendering
            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();

            debug_utils_label.pLabelName = "UI pass";
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
    // gltf imgs
    {
        vmaDestroyImage(init.allocator, data.gltf_model.textures.albedo.image, data.gltf_model.textures.albedo.vma_allocation);
        init.disp.destroyImageView(data.gltf_model.textures.albedo.image_view, nullptr);
        init.disp.destroySampler(data.gltf_model.textures.albedo.sampler, nullptr);

        vmaDestroyImage(init.allocator, data.gltf_model.textures.AOMetalRoughness.image, data.gltf_model.textures.AOMetalRoughness.vma_allocation);
        init.disp.destroyImageView(data.gltf_model.textures.AOMetalRoughness.image_view, nullptr);
        init.disp.destroySampler(data.gltf_model.textures.AOMetalRoughness.sampler, nullptr);

        vmaDestroyImage(init.allocator, data.gltf_model.textures.normal.image, data.gltf_model.textures.normal.vma_allocation);
        init.disp.destroyImageView(data.gltf_model.textures.normal.image_view, nullptr);
        init.disp.destroySampler(data.gltf_model.textures.normal.sampler, nullptr);

        vmaDestroyImage(init.allocator, data.gltf_model.textures.emissive.image, data.gltf_model.textures.emissive.vma_allocation);
        init.disp.destroyImageView(data.gltf_model.textures.emissive.image_view, nullptr);
        init.disp.destroySampler(data.gltf_model.textures.emissive.sampler, nullptr);
    }
    // dummy_img
    {
        vmaDestroyImage(init.allocator, data.dummy_img.image, data.dummy_img.vma_allocation);
        init.disp.destroyImageView(data.dummy_img.image_view, nullptr);
        init.disp.destroySampler(data.dummy_img.sampler, nullptr);
    }
    //envBrdf_img
    {
        vmaDestroyImage(init.allocator, data.envBrdf_img.image, data.envBrdf_img.vma_allocation);
        init.disp.destroyImageView(data.envBrdf_img.image_view, nullptr);
        init.disp.destroySampler(data.envBrdf_img.sampler, nullptr);
    }
    //prefilter_env_cubemap
    {
        vmaDestroyImage(init.allocator, data.prefilter_env_cubemap.image, data.prefilter_env_cubemap.vma_allocation);
        init.disp.destroyImageView(data.prefilter_env_cubemap.image_view, nullptr);
        init.disp.destroySampler(data.prefilter_env_cubemap.sampler, nullptr);
    }
    //diffuse_irradiance_cubemap
    {
        vmaDestroyImage(init.allocator, data.diffuse_irradiance_cubemap.image, data.diffuse_irradiance_cubemap.vma_allocation);
        init.disp.destroyImageView(data.diffuse_irradiance_cubemap.image_view, nullptr);
        init.disp.destroySampler(data.diffuse_irradiance_cubemap.sampler, nullptr);
    }
    //hdr_cubemap
    {
        vmaDestroyImage(init.allocator, data.hdr_cubemap.image, data.hdr_cubemap.vma_allocation);
        init.disp.destroyImageView(data.hdr_cubemap.image_view, nullptr);
        init.disp.destroySampler(data.hdr_cubemap.sampler, nullptr);
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    init.disp.destroyDescriptorSetLayout(data.spheres_pipeline.descriptor_set_layout, nullptr);
    init.disp.destroyDescriptorSetLayout(data.skybox_pipeline.descriptor_set_layout, nullptr);

    init.disp.destroyDescriptorPool(data.descriptor_pool, nullptr);

    vmaDestroyBuffer(init.allocator, data.gltf_model.index_buffer, data.gltf_model.index_buffer_allocation);

    vmaDestroyBuffer(init.allocator, data.gltf_model.vertex_buffer, data.gltf_model.vertex_buffer_allocation);

    for (size_t i = 0; i < init.swapchain.image_count; i++) {
        init.disp.destroySemaphore(data.finished_semaphore[i], nullptr);
        init.disp.destroySemaphore(data.available_semaphores[i], nullptr);
        init.disp.destroyFence(data.in_flight_fences[i], nullptr);
    }

    init.disp.destroyCommandPool(data.command_pool, nullptr);

    init.disp.destroyPipeline(data.skybox_pipeline.graphics_pipeline, nullptr);
    init.disp.destroyPipelineLayout(data.skybox_pipeline.pipeline_layout, nullptr);

    init.disp.destroyPipeline(data.spheres_pipeline.graphics_pipeline, nullptr);
    init.disp.destroyPipelineLayout(data.spheres_pipeline.pipeline_layout, nullptr);

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
    if (0 != create_command_pool(init, render_data)) return -1;
    if (0 != create_descriptor_pool(init, render_data)) return -1;
    if (0 != create_skybox_descriptor_set_layout(init, render_data)) return -1;
    if (0 != create_spheres_descriptor_set_layout(init, render_data)) return -1;
    if (0 != create_skybox_graphics_pipeline(init, render_data)) return -1;
    if (0 != create_spheres_graphics_pipeline(init, render_data)) return -1;
    if (0 != create_command_buffers(init, render_data)) return -1;
    if (0 != create_sync_objects(init, render_data)) return -1;
    if (0 != imgui_initialization(init, render_data)) return -1;
    if (0 != load_hdr_cubemap(init, render_data, "data/background_cubemap.hdr")) return -1;
    if (0 != load_diffuse_irradiance_cubemap(init, render_data, "data/diffuse_irradiance_cubemap.hdr")) return -1;
    if (0 != load_prefilter_environment_cubemap(init, render_data, "data/prefilterEnvMaps/")) return -1;
    if (0 != load_environment_brdf_map(init, render_data, "data/envBrdf.hdr")) return -1;
    if (0 != create_dummy_img(init, render_data)) return -1;
    if (0 != load_gltf_data(init, render_data, "data/DamagedHelmet/glTF/DamagedHelmet.gltf")) return -1;
    if (0 != create_vertex_buffer(init, render_data)) return -1;
    if (0 != create_index_buffer(init, render_data)) return -1;

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