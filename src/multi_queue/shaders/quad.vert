#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 0) out vec2 fragUV;

layout(push_constant) uniform constants
{
	vec4 data;
	mat4 mvp_matrix;
} PushConstants;

void main ()
{
	gl_Position = PushConstants.mvp_matrix * vec4(inPosition, 1.0);
	fragUV = inUV;
}