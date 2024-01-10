#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec2 fragUV;

layout (location = 0) out vec4 outColor;

layout (set = 0, binding = 1) uniform sampler2D samplerCsNoiseImg;

void main ()
{
    outColor = vec4(texture(samplerCsNoiseImg, fragUV).xyz, 1.0);
}
