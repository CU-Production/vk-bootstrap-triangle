#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in ivec4 inJointIndices;
layout(location = 3) in vec4 inJointWeights;
layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform constants
{
	vec4 data;
	mat4 mvp_matrix;
} PushConstants;

layout (std430, binding = 1) readonly buffer JointMatrices{
	mat4 jointMat[];
};

void main ()
{
	mat4 skinMat =
		inJointWeights.x * jointMat[int(inJointIndices.x)] +
		inJointWeights.y * jointMat[int(inJointIndices.y)] +
		inJointWeights.z * jointMat[int(inJointIndices.z)] +
		inJointWeights.w * jointMat[int(inJointIndices.w)];
	gl_Position = PushConstants.mvp_matrix * skinMat * vec4(inPosition, 1.0);
	fragColor = inNormal;
}