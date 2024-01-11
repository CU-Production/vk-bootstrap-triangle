#include <stdio.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <format>

#include <vulkan/vulkan.h>
#include <HandmadeMath.h>

#define VMA_IMPLEMENTATION
#define VMA_VULKAN_VERSION 1003000 // Vulkan 1.3
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <VkBootstrap.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define RDC_DEBUGING 1
#if RDC_DEBUGING
#include <renderdoc_app.h>
#include <windows.h>
RENDERDOC_API_1_1_2 *rdoc_api = NULL;
#endif

struct ComputeShaderPushConstants {
    float time;
};

const int MAX_FRAMES_IN_FLIGHT = 2;

struct Init {
    vkb::Instance instance;
    vkb::InstanceDispatchTable inst_disp;
    vkb::Device device;
    vkb::DispatchTable disp;
    VmaAllocator allocator;
};

struct RenderData {
    VkQueue compute_queue;

    VkPipelineLayout compute_pipeline_layout;
    VkPipeline compute_pipeline;
    VkDescriptorSetLayout compute_descriptor_set_layout;

    VkCommandPool command_pool;

    struct {
        VkImage              image;
        VkImageView          image_view;
        VkSampler            sampler;
        VmaAllocation        vma_allocation;
    } cs_noise_img;

    struct {
        VkBuffer             buffer;
        VmaAllocation        vma_allocation;
    } host_buffer;
};

int device_initialization(Init& init) {
    vkb::InstanceBuilder instance_builder;
    auto instance_ret = instance_builder
        .use_default_debug_messenger()
        .set_app_name("triangle")
        .set_engine_name("Null engine")
        .request_validation_layers()
        .require_api_version(1, 3)
        .set_headless(true)
        .build();
    if (!instance_ret) {
        std::cout << instance_ret.error().message() << "\n";
        return -1;
    }
    init.instance = instance_ret.value();

    init.inst_disp = init.instance.make_table();

    VkPhysicalDeviceVulkan13Features vulkan_13_features{};
    vulkan_13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan_13_features.dynamicRendering = true;
    vulkan_13_features.synchronization2 = true;

    vkb::PhysicalDeviceSelector phys_device_selector(init.instance);
    auto phys_device_ret = phys_device_selector
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

int renderdoc_app_initialization(Init& init) {
#if _MSC_VER
    // At init, on windows
    if(HMODULE mod = GetModuleHandleA("renderdoc.dll"))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
        assert(ret == 1);
    }
#else
    // At init, on linux/android.
    // For android replace librenderdoc.so with libVkLayer_GLES_RenderDoc.so
    if(void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
        assert(ret == 1);
    }
#endif

    return 0;
}

int get_queues(Init& init, RenderData& data) {
    auto gq = init.device.get_queue(vkb::QueueType::graphics);
    if (!gq.has_value()) {
        std::cout << "failed to get compute queue: " << gq.error().message() << "\n";
        return -1;
    }
    data.compute_queue = gq.value();

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

int create_host_buffer(Init& init, RenderData& data) {
    const size_t buffer_size = 4 * sizeof(uint8_t) * 512 * 512;

    VkBufferCreateInfo host_buffer_alloc_info{};
    host_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    host_buffer_alloc_info.size = buffer_size;;
    host_buffer_alloc_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo host_alloc_info{};
    host_alloc_info.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    host_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    if (vmaCreateBuffer(init.allocator, &host_buffer_alloc_info, &host_alloc_info, &data.host_buffer.buffer, &data.host_buffer.vma_allocation, VK_NULL_HANDLE) != VK_SUCCESS) {
        std::cout << "failed to staging buffer\n";
        return -1; // failed to create vertex buffer
    }

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
    cs_img_img_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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
        undefToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

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

        init.disp.queueSubmit(data.compute_queue, 1, &submit_info, VK_NULL_HANDLE);
        init.disp.queueWaitIdle(data.compute_queue);

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

    return 0;
}

int dispatch_cs(Init& init, RenderData& data) {

    {
        VkCommandBufferAllocateInfo cmdbuffer_alloc_info{};
        cmdbuffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdbuffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdbuffer_alloc_info.commandPool = data.command_pool;
        cmdbuffer_alloc_info.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        init.disp.allocateCommandBuffers(&cmdbuffer_alloc_info, &commandBuffer);

        init.disp.resetCommandBuffer(commandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (init.disp.beginCommandBuffer(commandBuffer, &begin_info) != VK_SUCCESS) {
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

            init.disp.cmdBeginDebugUtilsLabelEXT(commandBuffer, &debug_utils_label);

            init.disp.cmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, data.compute_pipeline);

            ComputeShaderPushConstants constants{};
            constants.time = 0;

            init.disp.cmdPushConstants(commandBuffer, data.compute_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputeShaderPushConstants), &constants);

            VkDescriptorImageInfo cs_descriptor_image_info;
            cs_descriptor_image_info.sampler = data.cs_noise_img.sampler;
            cs_descriptor_image_info.imageView = data.cs_noise_img.image_view;
            cs_descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write_descriptor_set{};
            write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_descriptor_set.dstBinding = 1;
            write_descriptor_set.descriptorCount = 1;
            write_descriptor_set.pImageInfo = &cs_descriptor_image_info;

            init.disp.cmdPushDescriptorSetKHR(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, data.compute_pipeline_layout, 0, 1, &write_descriptor_set);

            init.disp.cmdDispatch(commandBuffer, 512 / 8, 512 / 8, 1);

            // UAV -> copySrc
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
                undefToShaderReadOptBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                // undefToShaderReadOptBarrier.srcQueueFamilyIndex = data.compute_queue_index;
                // undefToShaderReadOptBarrier.dstQueueFamilyIndex = data.graphics_queue_index;

                init.disp.cmdPipelineBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &undefToShaderReadOptBarrier);
            }

            init.disp.cmdEndDebugUtilsLabelEXT(commandBuffer);
        }

        {
            if (init.disp.endCommandBuffer(commandBuffer) != VK_SUCCESS) {
                std::cout << "failed to record cs command buffer\n";
                return -1; // failed to record cs command buffer!
            }

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            if (init.disp.queueSubmit(data.compute_queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
                std::cout << "failed to submit compute command buffer\n";
                return -1; //"failed to submit compute command buffer
            }

            init.disp.queueWaitIdle(data.compute_queue);
        }
    }

    return 0;
}

int copy_and_save_image(Init& init, RenderData& data) {
    VkCommandBufferAllocateInfo cmdbuffer_alloc_info{};
    cmdbuffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdbuffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdbuffer_alloc_info.commandPool = data.command_pool;
    cmdbuffer_alloc_info.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    init.disp.allocateCommandBuffers(&cmdbuffer_alloc_info, &commandBuffer);

    init.disp.resetCommandBuffer(commandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (init.disp.beginCommandBuffer(commandBuffer, &begin_info) != VK_SUCCESS) {
        return -1; // failed to begin recording command buffer
    }

    {
        VkBufferImageCopy image_copy{};
        image_copy.bufferOffset = 0;
        image_copy.imageExtent = {512, 512, 1};
        image_copy.imageOffset = {0, 0, 0};
        image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy.imageSubresource.mipLevel = 0;
        image_copy.imageSubresource.baseArrayLayer = 0;
        image_copy.imageSubresource.layerCount = 1;
        image_copy.bufferRowLength = 512;

        init.disp.cmdCopyImageToBuffer(commandBuffer, data.cs_noise_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, data.host_buffer.buffer, 1, &image_copy);
    }

    if (init.disp.endCommandBuffer(commandBuffer) != VK_SUCCESS) {
        std::cout << "failed to record command buffer\n";
        return -1; // failed to record command buffer!
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (init.disp.queueSubmit(data.compute_queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        std::cout << "failed to submit draw command buffer\n";
        return -1; //"failed to submit draw command buffer
    }

    init.disp.queueWaitIdle(data.compute_queue);

    void* mapped_data;
    vmaMapMemory(init.allocator, data.host_buffer.vma_allocation, &mapped_data);
    stbi_write_png("noise.png", 512, 512, 4, mapped_data, 512*4);
    stbi_write_jpg("noise.jpg", 512, 512, 4, mapped_data, 100);
    vmaUnmapMemory(init.allocator, data.host_buffer.vma_allocation);

    return 0;
}

void cleanup(Init& init, RenderData& data) {
    vmaDestroyBuffer(init.allocator, data.host_buffer.buffer, data.host_buffer.vma_allocation);

    vmaDestroyImage(init.allocator, data.cs_noise_img.image, data.cs_noise_img.vma_allocation);
    init.disp.destroyImageView(data.cs_noise_img.image_view, nullptr);
    init.disp.destroySampler(data.cs_noise_img.sampler, nullptr);

    init.disp.destroyCommandPool(data.command_pool, nullptr);

    init.disp.destroyDescriptorSetLayout(data.compute_descriptor_set_layout, nullptr);

    init.disp.destroyPipeline(data.compute_pipeline, nullptr);
    init.disp.destroyPipelineLayout(data.compute_pipeline_layout, nullptr);

    vmaDestroyAllocator(init.allocator);

    vkb::destroy_device(init.device);
    vkb::destroy_instance(init.instance);
}

int main() {
    Init init;
    RenderData render_data;

    if (0 != device_initialization(init)) return -1;
    if (0 != renderdoc_app_initialization(init)) return -1;
    if (0 != get_queues(init, render_data)) return -1;
    if (0 != create_compute_descriptor_set_layout(init, render_data)) return -1;
    if (0 != create_compute_pipeline(init, render_data)) return -1;
    if (0 != create_command_pool(init, render_data)) return -1;
    if (0 != create_host_buffer(init, render_data)) return -1;
    if (0 != create_cs_noise_img(init, render_data)) return -1;

    if(rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL);
    if (0 != dispatch_cs(init, render_data)) return -1;
    if (0 != copy_and_save_image(init, render_data)) return -1;
    if(rdoc_api) rdoc_api->EndFrameCapture(NULL, NULL);

    init.disp.deviceWaitIdle();

    cleanup(init, render_data);
    return 0;
}