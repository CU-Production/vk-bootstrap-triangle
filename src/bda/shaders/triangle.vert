#version 460 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct Vertex
{
    vec3 position;
    vec3 color;
};

layout(buffer_reference, std430, buffer_reference_align = 32) buffer vertex_buffer_type {
    Vertex v;
};

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform constants
{
    vec4 data;
    mat4 mvp_matrix;
    // uint64_t vertex_buffer_address;
    vertex_buffer_type vertex_buffer;
} PushConstants;

void main ()
{
    // vertex_buffer_type vertex_buffer = vertex_buffer_type(PushConstants.vertex_buffer_address);
    // vec3 inPosition = vertex_buffer[gl_VertexIndex].v.position;
    // vec3 inColor    = vertex_buffer[gl_VertexIndex].v.color;

    vec3 inPosition = PushConstants.vertex_buffer[gl_VertexIndex].v.position;
    vec3 inColor    = PushConstants.vertex_buffer[gl_VertexIndex].v.color;

    gl_Position = PushConstants.mvp_matrix * vec4(inPosition, 1.0);
    fragColor = inColor;
}