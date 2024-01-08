struct VSInput
{
    float3 vPosition : POSITION;
    float3 vNormal   : NORMAL;
    float4 vTangent  : TANGENT;
    float2 vUV       : TEXCOORD;
};

struct VSOutput
{
    float4 Pos       : SV_POSITION;
    float4 WorldPos  : POSITION0;
    float3 Normal    : NORMAL0;
    float3 Tangent   : TANGENT0;
    float3 BiTangent : TANGENT1;
    float2 UV        : TEXCOORD0;
};

struct PushData
{
#if __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
    float4x4 m_matrix;
    float4x4 vp_matrix;
#elif __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
    [[vk::offset(128)]]
    uint4  texIdx;
    uint4  gltfTexIdx;
    float4 lightPositions[4];
    float4 cameraPosition;
    uint4  params; // x: enable light, y: enable ibl diffuse, z: enable ibl diffuse specular, w: mips
#endif
};

[[vk::push_constant]]
PushData pushData;

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
VSOutput mainVS(VSInput vertInput)
{
    VSOutput output = (VSOutput)0;

    const float4x4 modelMat = pushData.m_matrix;

    output.WorldPos = mul(modelMat, float4(vertInput.vPosition, 1.0));
    output.Pos = mul(pushData.vp_matrix, output.WorldPos);
    output.Normal = normalize(mul(modelMat, float4(vertInput.vNormal, 0.0)).xyz);
    output.Tangent = normalize(mul(modelMat, float4(vertInput.vTangent.xyz, 0.0)).xyz);
    output.BiTangent = cross(output.Normal, output.Tangent) * vertInput.vTangent.w;
    output.UV = vertInput.vUV;

    return output;
}
#endif



#if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
#include "GGXModel.hlsli"

[[vk::binding(1)]] Texture2D    i_2dtextures[];
[[vk::binding(1)]] SamplerState i_2dsamplerState;
[[vk::binding(1)]] TextureCube  i_cubeMapTextures[];
[[vk::binding(1)]] SamplerState i_cubesamplerState;

// [[vk::binding(1)]] TextureCube i_diffuseCubeMapTexture;      // texIdx.y
// [[vk::binding(1)]] TextureCube i_prefilterEnvCubeMapTexture; // texIdx.z
// [[vk::binding(1)]] Texture2D   i_envBrdfTexture;             // texIdx.w

float4 mainPS(VSOutput fragInput) : SV_TARGET
{
    const float3 lightColor = float3(24.0, 24.0, 24.0);

    float2 uv = fragInput.UV;
    // float3 Albedo = pushData.albedo_maxPreFilterMips.xyz; // F0
    const float3 Albedo = i_2dtextures[NonUniformResourceIndex(pushData.gltfTexIdx.x)].Sample(i_2dsamplerState, uv).xyz;
    const float3 amr = i_2dtextures[NonUniformResourceIndex(pushData.gltfTexIdx.y)].Sample(i_2dsamplerState, uv).xyz;
    const float ao = amr.x;
    const float metallic = amr.z;
    const float roughness = amr.y;
    const float maxMipLevel = float(pushData.params.w);
    const float3 emissive = i_2dtextures[NonUniformResourceIndex(pushData.gltfTexIdx.w)].Sample(i_2dsamplerState, uv).xyz;

    float3 V = normalize(pushData.cameraPosition.xyz - fragInput.WorldPos.xyz);
    float3 N = normalize(fragInput.Normal);
    float3 R = reflect(-V, N);

    float3 T = normalize(fragInput.Tangent.xyz);
    float3 B = normalize(fragInput.BiTangent.xyz);

    float3 normalSampled = i_2dtextures[NonUniformResourceIndex(pushData.gltfTexIdx.z)].Sample(i_2dsamplerState, uv).xyz;
    normalSampled = normalize(normalSampled * 2.0 - float3(1.0, 1.0, 1.0));
    float3x3 TBN = float3x3(T, B, N);
    N = mul(normalSampled, TBN);

    V = mul(V, TBN);

    float3 F0 = float3(0.04, 0.04, 0.04);
    F0        = lerp(F0, Albedo, float3(metallic, metallic, metallic));

    uint enable_light = pushData.params.x;
    uint enable_ibl_diffuse = pushData.params.y;
    uint enable_ibl_specular = pushData.params.z;

    // point light
    float3 Lo = float3(0.0, 0.0, 0.0); // Output light values to the view direction.
    if (enable_light)
    {
        for(int i = 0; i < 4; i++)
        {
            float3 lightPos = pushData.lightPositions[i].xyz;
            float3 L        = normalize(lightPos - fragInput.WorldPos.xyz);
            float3 H	    = normalize(V + L);
            float distance  = length(lightPos - fragInput.WorldPos.xyz);

            float  attenuation = 1.0 / (distance * distance);
            float3 radiance    = lightColor * attenuation; 

            float lightNormalCosTheta = max(dot(N, L), 0.0);
            float viewNormalCosTheta  = max(dot(N, V), 0.0);

            float  NDF = DistributionGGX(N, H, roughness);
            float  G   = GeometrySmithDirectLight(N, V, L, roughness);
            float3 F   = FresnelSchlick(max(dot(H, V), 0.0), F0);

            float3 NFG = NDF * F * G;
            float denominator = 4.0 * viewNormalCosTheta * lightNormalCosTheta  + 0.0001;
            float3 specular = NFG / denominator;

            float3 kD = float3(1.0, 1.0, 1.0) - F; // The amount of light goes into the material.
            kD *= (1.0 - metallic);

            Lo += (kD * (Albedo / 3.14159265359) + specular) * radiance * lightNormalCosTheta;
        }
    }

    // ibl
    float3 IBLo = float3(0.0, 0.0, 0.0);

    float NoV = max(dot(N, V), 0.0);

    if (enable_ibl_diffuse)
    {
        float3 Ks = fresnelSchlickRoughness(NoV, F0, roughness);
        float3 Kd = float3(1.0, 1.0, 1.0) - Ks;
        Kd *= (1.0 - metallic);

        float3 diffuseIrradiance = i_cubeMapTextures[NonUniformResourceIndex(pushData.texIdx.y)].Sample(i_cubesamplerState, N).xyz;

        float3 diffuse = Kd * diffuseIrradiance * Albedo;

        IBLo += diffuse * ao;
    }

    if (enable_ibl_specular)
    {
        float3 Ks = fresnelSchlickRoughness(NoV, F0, roughness);

        float3 prefilterEnv = i_cubeMapTextures[NonUniformResourceIndex(pushData.texIdx.z)].SampleLevel(i_cubesamplerState,
                                                                       R, roughness * maxMipLevel).xyz;

        float2 envBrdf = i_2dtextures[NonUniformResourceIndex(pushData.texIdx.w)].Sample(i_2dsamplerState, float2(NoV, roughness)).xy;

        float3 specular = prefilterEnv * (Ks * envBrdf.x + envBrdf.y);

        IBLo += specular * computeSpecOcclusion(max(dot(N, V), 0.0), ao, roughness);
    }

    float3 color = IBLo + Lo + emissive * ao;

    // HDR tonemapping
    // color = color / (color + (1.5).xxx);

    // Gamma Correction
    color = pow(color, 1.0 / 2.2);

    return float4(color, 1.0);
    // return float4(emissive + Albedo, 1.0);
    // return float4(normalSampled, 1.0);
}
#endif
