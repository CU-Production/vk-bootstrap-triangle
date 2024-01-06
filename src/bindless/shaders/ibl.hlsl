struct VSInput
{
    float3 vPosition : POSITION;
    float3 vNormal   : NORMAL;
    float4 vTangent  : TANGENT;
    float2 vUV       : TEXCOORD;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float4 WorldPos : POSITION0;
    float4 Normal : NORMAL0;
    float4 Tangent : TANGENT0;
    float2 UV : TEXCOORD0;
    nointerpolation float2 Params : TEXCOORD1; // [Metalic, roughness].
};

struct PushData
{
#if __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
	float4x4 mvp_matrix;
    float4 cumtom_param;
#elif __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
    [[vk::offset(80)]]
    uint4  texIdx;
    float4 lightPositions[4];
	float4 cameraPosition;
    float4 albedo_maxPreFilterMips;
    uint4  params; // x: enable light, y: enable ibl diffuse, z: enable ibl diffuse specular
#endif
};

[[vk::push_constant]]
PushData pushData;

float4x4 PosToModelMat(float3 pos)
{
    // NOTE: HLSL's matrices are column major.
    // But, it is filled column by column in this way. So it's good.
    // As for the UBO mat input, we still need to transpose the row-major matrix.
    float4x4 mat = { float4(1.0, 0.0, 0.0, pos.x),
                     float4(0.0, 1.0, 0.0, pos.y),
                     float4(0.0, 0.0, 1.0, pos.z),
                     float4(0.0, 0.0, 0.0, 1.0) };
                     
    return mat;
}

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
VSOutput mainVS(VSInput vertInput)
{
    VSOutput output = (VSOutput)0;

    float4x4 modelMat = PosToModelMat(float3(3.0, 0, 0));

    float4 worldPos = mul(modelMat, float4(vertInput.vPosition, 1.0));
    float4 worldNormal = mul(modelMat, float4(vertInput.vNormal, 0.0));
    float4 worldTangent = mul(modelMat, vertInput.vTangent);

    output.WorldPos = worldPos;
    output.Normal.xyz = normalize(worldNormal.xyz);
    output.Pos = mul(pushData.mvp_matrix, worldPos);
    output.Tangent = worldTangent;
    output.UV = vertInput.vUV;


    output.Params.xy = pushData.cumtom_param.xy;

    return output;
}
#endif

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
#include "GGXModel.hlsli"

[[vk::binding(1)]] TextureCube  i_cubeMapTextures[];
[[vk::binding(1)]] Texture2D    i_2dtextures[];
[[vk::binding(1)]] SamplerState samplerState;

// [[vk::binding(1)]] TextureCube i_diffuseCubeMapTexture;      // texIdx.y
// [[vk::binding(1)]] TextureCube i_prefilterEnvCubeMapTexture; // texIdx.z
// [[vk::binding(1)]] Texture2D   i_envBrdfTexture;             // texIdx.w

float4 mainPS(VSOutput fragInput) : SV_TARGET
{
    float3 lightColor = float3(24.0, 24.0, 24.0);

	// Gold
	float3 Albedo = pushData.albedo_maxPreFilterMips.xyz; // F0
    float maxMipLevel = pushData.albedo_maxPreFilterMips.w;

	float3 V = normalize(pushData.cameraPosition.xyz -fragInput.WorldPos.xyz);
    float3 N = normalize(fragInput.Normal.xyz);
    float3 R = reflect(-V, N);

	float metallic = fragInput.Params.x;
	float roughness = fragInput.Params.y;

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

        float3 diffuseIrradiance = i_cubeMapTextures[NonUniformResourceIndex(pushData.texIdx.y)].Sample(samplerState, N).xyz;

        float3 diffuse = Kd * diffuseIrradiance * Albedo;

        IBLo += diffuse;
    }

    if (enable_ibl_specular)
    {
        float3 Ks = fresnelSchlickRoughness(NoV, F0, roughness);

        float3 prefilterEnv = i_cubeMapTextures[NonUniformResourceIndex(pushData.texIdx.z)].SampleLevel(samplerState,
                                                                       R, roughness * maxMipLevel).xyz;

        float2 envBrdf = i_2dtextures[NonUniformResourceIndex(pushData.texIdx.w)].Sample(samplerState, float2(NoV, roughness)).xy;

        float3 specular = prefilterEnv * (Ks * envBrdf.x + envBrdf.y);

        IBLo += specular;
    }

    float3 color = IBLo + Lo;
	
    // HDR tonemapping
    color = color / (color + (1.0).xxx);

    // Gamma Correction
    color = pow(color, (1.0/2.2).xxx);

	return float4(color, 1.0);
}
#endif
