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

#include <shaders/gear.vert.spv.h>
#include <shaders/gear.frag.spv.h>

#ifdef _MSC_VER
#pragma comment (linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")
#endif

const uint32_t NUM_GEARS = 3;
const float M_PI = 3.14159265358979323846f;// pi
const uint32_t SCR_WIDTH = 256;
const uint32_t SCR_HEIGHT = 256;

// https://github.com/SaschaWillems/Vulkan/blob/master/examples/gears/gears.cpp
// Used for passing the definition of a gear during construction
struct GearDefinition {
	float innerRadius;
	float outerRadius;
	float width;
	int numTeeth;
	float toothDepth;
	HMM_Vec3 color;
	HMM_Vec3 pos;
	float rotSpeed;
	float rotOffset;
};

/*
 * Gear
 * This class contains the properties of a single gear and a function to generate vertices and indices
 */
class Gear
{
public:
	// Definition for the vertex data used to render the gears
	struct Vertex {
		HMM_Vec3 position;
		HMM_Vec3 normal;
		HMM_Vec3 color;
	};

	HMM_Vec3 color;
	HMM_Vec3 pos;
	float rotSpeed{ 0.0f };
	float rotOffset{ 0.0f };
	// These are used at draw time to offset into the single buffers
	uint32_t indexCount{ 0 };
	uint32_t indexStart{ 0 };

	// Generates the indices and vertices for this gear
	// They are added to the vertex and index buffers passed into the function
	// This way we can put all gears into single vertex and index buffers instead of having to allocate single buffers for each gear (which would be bad practice)
	void generate(GearDefinition& gearDefinition, std::vector<Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer) {
		this->color = gearDefinition.color;
		this->pos = gearDefinition.pos;
		this->rotOffset = gearDefinition.rotOffset;
		this->rotSpeed = gearDefinition.rotSpeed;

		int i;
		float r0, r1, r2;
		float ta, da;
		float u1, v1, u2, v2, len;
		float cos_ta, cos_ta_1da, cos_ta_2da, cos_ta_3da, cos_ta_4da;
		float sin_ta, sin_ta_1da, sin_ta_2da, sin_ta_3da, sin_ta_4da;
		int32_t ix0, ix1, ix2, ix3, ix4, ix5;

		// We need to know where this triangle's indices start within the single index buffer
		indexStart = static_cast<uint32_t>(indexBuffer.size());

		r0 = gearDefinition.innerRadius;
		r1 = gearDefinition.outerRadius - gearDefinition.toothDepth / 2.0f;
		r2 = gearDefinition.outerRadius + gearDefinition.toothDepth / 2.0f;
		da = static_cast <float>(2.0 * M_PI / gearDefinition.numTeeth / 4.0);

		HMM_Vec3 normal;

		// Use lambda functions to simplify vertex and face creation
		auto addFace = [&indexBuffer](int a, int b, int c) {
			indexBuffer.push_back(a);
			indexBuffer.push_back(b);
			indexBuffer.push_back(c);
			};

		auto addVertex = [this, &vertexBuffer](float x, float y, float z, HMM_Vec3 normal) {
			Vertex v{};
			v.position = { x, y, z };
			v.normal = normal;
			v.color = this->color;
			vertexBuffer.push_back(v);
			return static_cast<int32_t>(vertexBuffer.size()) - 1;
			};

		for (i = 0; i < gearDefinition.numTeeth; i++) {
			ta = i * static_cast <float>(2.0 * M_PI / gearDefinition.numTeeth);

			cos_ta = cos(ta);
			cos_ta_1da = cos(ta + da);
			cos_ta_2da = cos(ta + 2.0f * da);
			cos_ta_3da = cos(ta + 3.0f * da);
			cos_ta_4da = cos(ta + 4.0f * da);
			sin_ta = sin(ta);
			sin_ta_1da = sin(ta + da);
			sin_ta_2da = sin(ta + 2.0f * da);
			sin_ta_3da = sin(ta + 3.0f * da);
			sin_ta_4da = sin(ta + 4.0f * da);

			u1 = r2 * cos_ta_1da - r1 * cos_ta;
			v1 = r2 * sin_ta_1da - r1 * sin_ta;
			len = sqrt(u1 * u1 + v1 * v1);
			u1 /= len;
			v1 /= len;
			u2 = r1 * cos_ta_3da - r2 * cos_ta_2da;
			v2 = r1 * sin_ta_3da - r2 * sin_ta_2da;

			// Front face
			normal = HMM_V3(0.0f, 0.0f, 1.0f);
			ix0 = addVertex(r0 * cos_ta, r0 * sin_ta, gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r1 * cos_ta, r1 * sin_ta, gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r0 * cos_ta, r0 * sin_ta, gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, gearDefinition.width * 0.5f, normal);
			ix4 = addVertex(r0 * cos_ta_4da, r0 * sin_ta_4da, gearDefinition.width * 0.5f, normal);
			ix5 = addVertex(r1 * cos_ta_4da, r1 * sin_ta_4da, gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);
			addFace(ix2, ix3, ix4);
			addFace(ix3, ix5, ix4);

			// Teeth front face
			normal = HMM_V3(0.0f, 0.0f, 1.0f);
			ix0 = addVertex(r1 * cos_ta, r1 * sin_ta, gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r2 * cos_ta_1da, r2 * sin_ta_1da, gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);

			// Back face
			normal = HMM_V3(0.0f, 0.0f, -1.0f);
			ix0 = addVertex(r1 * cos_ta, r1 * sin_ta, -gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r0 * cos_ta, r0 * sin_ta, -gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, -gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r0 * cos_ta, r0 * sin_ta, -gearDefinition.width * 0.5f, normal);
			ix4 = addVertex(r1 * cos_ta_4da, r1 * sin_ta_4da, -gearDefinition.width * 0.5f, normal);
			ix5 = addVertex(r0 * cos_ta_4da, r0 * sin_ta_4da, -gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);
			addFace(ix2, ix3, ix4);
			addFace(ix3, ix5, ix4);

			// Teeth back face
			normal = HMM_V3(0.0f, 0.0f, -1.0f);
			ix0 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, -gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, -gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r1 * cos_ta, r1 * sin_ta, -gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r2 * cos_ta_1da, r2 * sin_ta_1da, -gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);

			// Outard teeth faces
			normal = HMM_V3(v1, -u1, 0.0f);
			ix0 = addVertex(r1 * cos_ta, r1 * sin_ta, gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r1 * cos_ta, r1 * sin_ta, -gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r2 * cos_ta_1da, r2 * sin_ta_1da, gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r2 * cos_ta_1da, r2 * sin_ta_1da, -gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);

			normal = HMM_V3(cos_ta, sin_ta, 0.0f);
			ix0 = addVertex(r2 * cos_ta_1da, r2 * sin_ta_1da, gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r2 * cos_ta_1da, r2 * sin_ta_1da, -gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, -gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);

			normal = HMM_V3(v2, -u2, 0.0f);
			ix0 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, -gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, -gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);

			normal = HMM_V3(cos_ta, sin_ta, 0.0f);
			ix0 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, gearDefinition.width * 0.5f, normal);
			ix1 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, -gearDefinition.width * 0.5f, normal);
			ix2 = addVertex(r1 * cos_ta_4da, r1 * sin_ta_4da, gearDefinition.width * 0.5f, normal);
			ix3 = addVertex(r1 * cos_ta_4da, r1 * sin_ta_4da, -gearDefinition.width * 0.5f, normal);
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);

			// Inside cylinder faces
			ix0 = addVertex(r0 * cos_ta, r0 * sin_ta, -gearDefinition.width * 0.5f, HMM_V3(-cos_ta, -sin_ta, 0.0f));
			ix1 = addVertex(r0 * cos_ta, r0 * sin_ta, gearDefinition.width * 0.5f, HMM_V3(-cos_ta, -sin_ta, 0.0f));
			ix2 = addVertex(r0 * cos_ta_4da, r0 * sin_ta_4da, -gearDefinition.width * 0.5f, HMM_V3(-cos_ta_4da, -sin_ta_4da, 0.0f));
			ix3 = addVertex(r0 * cos_ta_4da, r0 * sin_ta_4da, gearDefinition.width * 0.5f, HMM_V3(-cos_ta_4da, -sin_ta_4da, 0.0f));
			addFace(ix0, ix1, ix2);
			addFace(ix1, ix3, ix2);
		}

		// We need to know how many indices this triangle has at draw time
		indexCount = static_cast<uint32_t>(indexBuffer.size()) - indexStart;
	}
};

const int MAX_FRAMES_IN_FLIGHT = 2;

struct UniformData
{
	HMM_Mat4 projection;
	HMM_Mat4 view;
	HMM_Vec4 lightPos;
	// The model matrix is used to rotate a given gear, so we have one mat4 per gear
	HMM_Mat4 model[NUM_GEARS];
};

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

    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;
	VkDescriptorSetLayout descriptor_set_layout;

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
	VkBuffer uniform_buffer;
	VmaAllocation uniform_buffer_allocation;
	VkDescriptorBufferInfo uniform_buffer_info;

	std::vector<Gear> gears{};
	std::vector<Gear::Vertex> vertices{};
	std::vector<uint32_t> indices{};

	UniformData uniform_data;
};

GLFWwindow* create_window_glfw(const char* window_name = "", bool resize = true) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (!resize) glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    return glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, window_name, NULL, NULL);
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
    init.window = create_window_glfw("Gears", false);

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

VkShaderModule createShaderModule(Init& init, const uint32_t* code, uint32_t code_size) {
    VkShaderModuleCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code_size;
    create_info.pCode = code;

    VkShaderModule shaderModule;
    if (init.disp.createShaderModule(&create_info, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE; // failed to create shader module
    }

    return shaderModule;
}

int create_descriptor_set_layout(Init& init, RenderData& data) {
	std::array<VkDescriptorSetLayoutBinding, 1> layout_bindings{};
	layout_bindings[0].binding = 0;
	layout_bindings[0].descriptorCount = 1;
	layout_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layout_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
	layout_info.bindingCount = layout_bindings.size();
	layout_info.pBindings = layout_bindings.data();

	if (init.disp.createDescriptorSetLayout(&layout_info, nullptr, &data.descriptor_set_layout) != VK_SUCCESS) {
		std::cout <<"failed to create descriptor set layout!\n";
		return -1;
	}

	return 0;
}

int create_graphics_pipeline(Init& init, RenderData& data) {
    // auto vert_code = readFile("shaders/triangle.vert.spv");
    // auto frag_code = readFile("shaders/triangle.frag.spv");
	uint32_t vert_code_size = sizeof(vert_code);
	uint32_t frag_code_size = sizeof(frag_code);

    VkShaderModule vert_module = createShaderModule(init, vert_code, vert_code_size);
    VkShaderModule frag_module = createShaderModule(init, frag_code, frag_code_size);
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
    bindingDescription.stride = sizeof(Gear::Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Gear::Vertex, position);
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Gear::Vertex, normal);
	attributeDescriptions[2].binding = 0;
	attributeDescriptions[2].location = 2;
	attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributeDescriptions[2].offset = offsetof(Gear::Vertex, color);

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

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts = &data.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 0;

    if (init.disp.createPipelineLayout(&pipeline_layout_info, nullptr, &data.pipeline_layout) != VK_SUCCESS) {
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

int create_gears(Init& init, RenderData& data) {
	// Set up three differntly shaped and colored gears
	std::vector<GearDefinition> gearDefinitions(3);

	// Large red gear
	gearDefinitions[0].innerRadius = 1.0f;
	gearDefinitions[0].outerRadius = 4.0f;
	gearDefinitions[0].width = 1.0f;
	gearDefinitions[0].numTeeth = 20;
	gearDefinitions[0].toothDepth = 0.7f;
	gearDefinitions[0].color = { 1.0f, 0.0f, 0.0f };
	gearDefinitions[0].pos = { -3.0f, 0.0f, 0.0f };
	gearDefinitions[0].rotSpeed = 1.0f;
	gearDefinitions[0].rotOffset = 0.0f;

	// Medium sized green gear
	gearDefinitions[1].innerRadius = 0.5f;
	gearDefinitions[1].outerRadius = 2.0f;
	gearDefinitions[1].width = 2.0f;
	gearDefinitions[1].numTeeth = 10;
	gearDefinitions[1].toothDepth = 0.7f;
	gearDefinitions[1].color = { 0.0f, 1.0f, 0.2f };
	gearDefinitions[1].pos = { 3.1f, 0.0f, 0.0f };
	gearDefinitions[1].rotSpeed = -2.0f;
	gearDefinitions[1].rotOffset = -9.0f;

	// Small blue gear
	gearDefinitions[2].innerRadius = 1.3f;
	gearDefinitions[2].outerRadius = 2.0f;
	gearDefinitions[2].width = 0.5f;
	gearDefinitions[2].numTeeth = 10;
	gearDefinitions[2].toothDepth = 0.7f;
	gearDefinitions[2].color = { 0.0f, 0.0f, 1.0f };
	gearDefinitions[2].pos = { -3.1f, -6.2f, 0.0f };
	gearDefinitions[2].rotSpeed = -2.0f;
	gearDefinitions[2].rotOffset = -30.0f;

	// Fills the vertex and index buffers for each of the gear
	data.gears.resize(gearDefinitions.size());
	for (int32_t i = 0; i < data.gears.size(); i++) {
		data.gears[i].generate(gearDefinitions[i], data.vertices, data.indices);
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

int create_uniform_buffer(Init& init, RenderData& data) {
	const uint32_t buffer_size = sizeof(data.uniform_data);
	/* uniform buffer */
	VkBufferCreateInfo uniform_buffer_info{};
	uniform_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	uniform_buffer_info.size = buffer_size;
	uniform_buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	uniform_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo uniform_buffer_alloc_info = {};
	uniform_buffer_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	uniform_buffer_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

	if (vmaCreateBuffer(init.allocator, &uniform_buffer_info, &uniform_buffer_alloc_info, &data.uniform_buffer, &data.uniform_buffer_allocation, nullptr) != VK_SUCCESS) {
		std::cout << "failed to create index buffer\n";
		return -1; // failed to create vertex buffer
	}

	data.uniform_buffer_info.buffer = data.uniform_buffer;
	data.uniform_buffer_info.offset = 0;
	data.uniform_buffer_info.range = buffer_size;

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

    	VkDeviceSize offsets[] = {0};
    	init.disp.cmdBindVertexBuffers(data.command_buffers[i], 0, 1, &data.vertex_buffer, offsets);
    	init.disp.cmdBindIndexBuffer(data.command_buffers[i], data.index_buffer, 0, VK_INDEX_TYPE_UINT32);

        HMM_Vec3 cam_pos = { 0.0f, 2.5f, -16.0f };
        HMM_Mat4 view = HMM_Translate(cam_pos);
        HMM_Mat4 projection = HMM_Perspective_RH_ZO(60.f * HMM_DegToRad, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 200.0f);
        projection[1][1] *= -1;

    	float degree = glfwGetTime() * 360.0f * 0.5f;
    	data.uniform_data.view = view;
    	data.uniform_data.projection = projection;
    	data.uniform_data.lightPos = {0.0f, 0.0f, 2.5f, 1.0f};
    	for (auto gid = 0; gid < NUM_GEARS; gid++) {
    		Gear gear = data.gears[gid];
    		data.uniform_data.model[gid] = HMM_Translate(gear.pos);
    		data.uniform_data.model[gid] = data.uniform_data.model[gid] * HMM_Rotate_RH(((gear.rotSpeed * degree) + gear.rotOffset) * HMM_DegToRad, {0.0f, 0.0f, 1.0f});
    	}

    	void* mapped_data;
    	vmaMapMemory(init.allocator, data.uniform_buffer_allocation, &mapped_data);
    	memcpy(mapped_data, &data.uniform_data, sizeof(data.uniform_data));
    	vmaUnmapMemory(init.allocator, data.uniform_buffer_allocation);

    	VkWriteDescriptorSet write_descriptor_set{};
    	write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    	write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    	write_descriptor_set.dstBinding = 0;
    	write_descriptor_set.descriptorCount = 1;
    	write_descriptor_set.pBufferInfo = &data.uniform_buffer_info;

    	init.disp.cmdPushDescriptorSetKHR(data.command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, data.pipeline_layout, 0, 1, &write_descriptor_set);

    	for (auto gid = 0; gid < NUM_GEARS; gid++) {
    		// We use the instance index (last argument) to pass the index of the triangle to the shader
    		// With this we can index into the model matrices array of the uniform buffer like this (see gears.vert):
    		// ubo.model[gl_InstanceIndex];
    		init.disp.cmdDrawIndexed(data.command_buffers[i], data.gears[gid].indexCount, 1, data.gears[gid].indexStart, 0, gid);
    	}

        init.disp.cmdEndRendering(data.command_buffers[i]);

        init.disp.cmdEndDebugUtilsLabelEXT(data.command_buffers[i]);

        // End frame
        {
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
    vmaDestroyBuffer(init.allocator, data.uniform_buffer, data.uniform_buffer_allocation);
    vmaDestroyBuffer(init.allocator, data.index_buffer, data.index_buffer_allocation);
    vmaDestroyBuffer(init.allocator, data.vertex_buffer, data.vertex_buffer_allocation);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        init.disp.destroySemaphore(data.finished_semaphore[i], nullptr);
        init.disp.destroySemaphore(data.available_semaphores[i], nullptr);
        init.disp.destroyFence(data.in_flight_fences[i], nullptr);
    }

    init.disp.destroyCommandPool(data.command_pool, nullptr);

	init.disp.destroyDescriptorSetLayout(data.descriptor_set_layout, nullptr);
    init.disp.destroyPipeline(data.graphics_pipeline, nullptr);
    init.disp.destroyPipelineLayout(data.pipeline_layout, nullptr);

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
	if (0 != create_descriptor_set_layout(init, render_data)) return -1;
    if (0 != create_graphics_pipeline(init, render_data)) return -1;
    if (0 != create_command_pool(init, render_data)) return -1;
	if (0 != create_gears(init, render_data)) return -1;
    if (0 != create_vertex_buffer(init, render_data)) return -1;
    if (0 != create_index_buffer(init, render_data)) return -1;
	if (0 != create_uniform_buffer(init, render_data)) return -1;
    if (0 != create_command_buffers(init, render_data)) return -1;
    if (0 != create_sync_objects(init, render_data)) return -1;

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