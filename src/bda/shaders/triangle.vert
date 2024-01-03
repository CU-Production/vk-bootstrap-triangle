#version 460 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// struct Vertex
// {
//     vec3 position;
//     vec3 color;
// };

layout(buffer_reference, std430, buffer_reference_align = 4) buffer vertex_buffer_type {
    float v;
};

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform constants
{
	vec4 data;
	mat4 mvp_matrix;
	uint64_t vertex_buffer_address;
} PushConstants;

void main ()
{
	vertex_buffer_type vertex_buffer = vertex_buffer_type(PushConstants.vertex_buffer_address);

	float x = vertex_buffer[gl_VertexIndex * 6 + 0].v;
	float y = vertex_buffer[gl_VertexIndex * 6 + 1].v;
	float z = vertex_buffer[gl_VertexIndex * 6 + 2].v;
	vec3 inPosition = vec3(x, y, z);

	float r = vertex_buffer[gl_VertexIndex * 6 + 3].v;
	float g = vertex_buffer[gl_VertexIndex * 6 + 4].v;
	float b = vertex_buffer[gl_VertexIndex * 6 + 5].v;
	vec3 inColor = vec3(r, g, b);

	gl_Position = PushConstants.mvp_matrix * vec4(inPosition, 1.0);
	fragColor = inColor;
}