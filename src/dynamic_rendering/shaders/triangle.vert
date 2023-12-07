#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform constants
{
	vec4 data;
	mat4 mvp_matrix;
} PushConstants;

void main ()
{
	gl_Position = PushConstants.mvp_matrix * vec4(inPosition, 1.0);
	fragColor = inColor;
}