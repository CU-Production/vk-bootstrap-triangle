#version 460 core
#extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 inPosition;
// layout(location = 1) in vec3 inNormal;
// layout(location = 2) in ivec4 inJointIndices;
// layout(location = 3) in vec4 inJointWeights;
layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform constants
{
	vec4 data;
	mat4 mvp_matrix;
} PushConstants;

struct Vertex {
	vec4 Position;
	vec4 Normal;
	ivec4 JointIndices;
	vec4 JointWeights;
};

layout (std430, binding = 0) readonly buffer VertexBuffer{
	Vertex vertices[];
};

layout (std430, binding = 1) readonly buffer IndexBuffer{
	uint indices[];
};

layout (std430, binding = 2) readonly buffer JointMatrices{
	mat4 jointMat[];
};

// layout (std430, binding = 1) readonly buffer JointMatrices{
// 	mat4 jointMat[];
// };

void main ()
{
	uint vertexId = indices[gl_VertexIndex];
	vec3  inPosition     = vertices[vertexId].Position.xyz;
	vec3  inNormal       = vertices[vertexId].Normal.xyz;
	ivec4 inJointIndices = vertices[vertexId].JointIndices;
	vec4  inJointWeights = vertices[vertexId].JointWeights;

	mat4 skinMat =
		inJointWeights.x * jointMat[int(inJointIndices.x)] +
		inJointWeights.y * jointMat[int(inJointIndices.y)] +
		inJointWeights.z * jointMat[int(inJointIndices.z)] +
		inJointWeights.w * jointMat[int(inJointIndices.w)];
	gl_Position = PushConstants.mvp_matrix * skinMat * vec4(inPosition, 1.0);
	fragColor = inNormal;
}