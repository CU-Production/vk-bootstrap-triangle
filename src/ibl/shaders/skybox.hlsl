struct VSOutput
{
    float4 Pos : SV_POSITION;
};

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
static float2 positions[6] =
{
    float2(-1.f, -1.f),
    float2(-1.f,  1.f),
    float2( 1.f, -1.f),
    float2( 1.f,  1.f),
    float2( 1.f, -1.f),
    float2(-1.f,  1.f)
};

VSOutput mainVS(uint vertId : SV_VertexID)
{
    VSOutput output = (VSOutput)0;
    output.Pos = float4(positions[vertId], 0.99, 1.0);

    return output;
}
#endif

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
struct CameraInfoUbo
{
    float4 view;
    float4 right;
    float4 up;
    float2 viewportWidthHeight; // Screen width and height in the unit of pixels.
    float2 nearWidthHeight; // Near plane's width and height in the world.
    float  near; // Pack it with 'up'.
};

[[vk::push_constant]]
CameraInfoUbo i_cameraInfo;

[[vk::binding(1)]] TextureCube i_cubeMapTexture;
[[vk::binding(1)]] SamplerState samplerState;

float4 mainPS(VSOutput fragInput) : SV_TARGET
{
    float4 fragCoord = fragInput.Pos;

    // Map current pixel coordinate to [-1.f, 1.f].
    float vpWidth = i_cameraInfo.viewportWidthHeight[0];
    float vpHeight = i_cameraInfo.viewportWidthHeight[1];

    float x = ((fragCoord[0] / vpWidth) * 2.f) - 1.f;
    float y = ((fragCoord[1] / vpHeight) * 2.f) - 1.f;

    // Generate the pixel world position on the near plane and its world direction.
    float nearWorldWidth = i_cameraInfo.nearWidthHeight[0];
    float nearWorldHeight = i_cameraInfo.nearWidthHeight[1];

    float3 sampleDir = x * (nearWorldWidth / 2.f) * i_cameraInfo.right.xyz +
                       (-y) * (nearWorldHeight / 2.f) * i_cameraInfo.up.xyz +
                       i_cameraInfo.view.xyz * i_cameraInfo.near;

    sampleDir = normalize(sampleDir);

    // Sample the cubemap
    return i_cubeMapTexture.Sample(samplerState, sampleDir);
}
#endif
