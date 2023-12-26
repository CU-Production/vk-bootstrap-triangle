struct VSInput
{
    float3 vPosition : POSITION;
    float3 vNormal : NORMAL;
};

struct VSOutput
{
    float4 Pos : SV_POSITION;
    float4 WorldPos : POSITION0;
    float4 Normal : NORMAL0;
    nointerpolation float2 Params : TEXCOORD0; // [Metalic, roughness].
};

struct PushData
{
#if __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
	float4x4 mvp_matrix;
    float4 cumtom_param;
#elif __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
    [[vk::offset(80)]]
    float4 lightPositions[4];
	float4 cameraPosition;
    float4 albedo_maxPreFilterMips;
    float4 params; // x: enable light, y: enable ibl
#endif
};

[[vk::push_constant]]
PushData pushData;

static const float3 g_sphereWorldPos[15] = {
    float3(15.0,   2.5, -8.0),
    float3(15.0,   2.5, -8.0 + 1.0 * (16.0 / 6.0)),
    float3(15.0,   2.5, -8.0 + 2.0 * (16.0 / 6.0)),
    float3(15.0,   2.5, -8.0 + 3.0 * (16.0 / 6.0)),
    float3(15.0,   2.5, -8.0 + 4.0 * (16.0 / 6.0)),
    float3(15.0,   2.5, -8.0 + 5.0 * (16.0 / 6.0)),
    float3(15.0,   2.5, -8.0 + 6.0 * (16.0 / 6.0)),
    float3(15.0,  -2.5, -8.0),
    float3(15.0,  -2.5, -8.0 + 1.0 * (16.0 / 6.0)),
    float3(15.0,  -2.5, -8.0 + 2.0 * (16.0 / 6.0)),
    float3(15.0,  -2.5, -8.0 + 3.0 * (16.0 / 6.0)),
    float3(15.0,  -2.5, -8.0 + 4.0 * (16.0 / 6.0)),
    float3(15.0,  -2.5, -8.0 + 5.0 * (16.0 / 6.0)),
    float3(15.0,  -2.5, -8.0 + 6.0 * (16.0 / 6.0)),
    float3(15.0,     0,  0)
};

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
VSOutput mainVS(
    VSInput vertInput,
    uint instId : SV_InstanceID)
{
    VSOutput output = (VSOutput)0;

    float4x4 modelMat = PosToModelMat(g_sphereWorldPos[instId]);

    float4 worldPos = mul(modelMat, float4(vertInput.vPosition, 1.0));
    float4 worldNormal = mul(modelMat, float4(vertInput.vNormal, 0.0));

    output.WorldPos = worldPos;
    output.Normal.xyz = normalize(worldNormal.xyz);
    output.Pos = mul(pushData.mvp_matrix, worldPos);

    float roughnessOffset = 1.0 / 7.0;
    int instIdRemap = instId % 7;

    output.Params.x = 1.0;
    if(instId >= 7)
    {
        output.Params.x = 0.0;
    }

    output.Params.y = min(instIdRemap * roughnessOffset + 0.05, 1.0);

    if (instId == 14)
    {
        output.Params.xy = pushData.cumtom_param.xy;
    }

    return output;
}
#endif

#if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
#include "GGXModel.hlsli"

[[vk::binding(2)]] TextureCube i_diffuseCubeMapTexture;
[[vk::binding(2)]] SamplerState i_diffuseCubemapSamplerState;

[[vk::binding(3)]] TextureCube i_prefilterEnvCubeMapTexture;
[[vk::binding(3)]] SamplerState i_prefilterEnvCubeMapSamplerState;

[[vk::binding(4)]] Texture2D    i_envBrdfTexture;
[[vk::binding(4)]] SamplerState i_envBrdfSamplerState;

float4 mainPS(VSOutput fragInput) : SV_TARGET
{
    float3 lightColor = float3(4.0, 4.0, 4.0);

	// Gold
	float3 Albedo = pushData.albedo_maxPreFilterMips.xyz; // F0
    float maxMipLevel = pushData.albedo_maxPreFilterMips.w;

	float3 wo = normalize(pushData.cameraPosition.xyz - fragInput.WorldPos.xyz);
	
	float3 worldNormal = normalize(fragInput.Normal.xyz);

	float viewNormalCosTheta = max(dot(worldNormal, wo), 0.0);

	float metallic = fragInput.Params.x;
	float roughness = fragInput.Params.y;

    // point light
	float3 Lo = float3(0.0, 0.0, 0.0); // Output light values to the view direction.
	for(int i = 0; i < 4; i++)
	{
		float3 lightPos = pushData.lightPositions[i].xyz;
		float3 wi       = normalize(lightPos - fragInput.WorldPos.xyz);
		float3 H	    = normalize(wi + wo);
		float distance  = length(lightPos - fragInput.WorldPos.xyz);

		float  attenuation = 1.0 / (distance * distance);
		float3 radiance    = lightColor * attenuation; 

		float lightNormalCosTheta = max(dot(worldNormal, wi), 0.0);

		float NDF = DistributionGGX(worldNormal, H, roughness);
        float G   = GeometrySmithDirectLight(worldNormal, wo, wi, roughness);

		float3 F0 = float3(0.04, 0.04, 0.04);
        F0        = lerp(F0, Albedo, float3(metallic, metallic, metallic));
        float3 F  = FresnelSchlick(max(dot(H, wo), 0.0), F0);

		float3 NFG = NDF * F * G;

		float denominator = 4.0 * viewNormalCosTheta * lightNormalCosTheta  + 0.0001;
		
		float3 specular = NFG / denominator;

		float3 kD = float3(1.0, 1.0, 1.0) - F; // The amount of light goes into the material.
		kD *= (1.0 - metallic);

		Lo += (kD * (Albedo / 3.14159265359) + specular) * radiance * lightNormalCosTheta;
	}
    Lo *= pushData.params.x;

    // ibl
    float3 IBLo = float3(0.0, 0.0, 0.0);
    {
        float3 V = normalize(fragInput.WorldPos.xyz);
        float3 N = normalize(fragInput.Normal.xyz);
        float NoV = saturate(dot(N, V));
        float3 R = 2 * NoV * N - V;

        float3 F0 = float3(0.04, 0.04, 0.04);
        F0 = lerp(F0, Albedo, float3(metallic, metallic, metallic));

        float3 diffuseIrradiance = i_diffuseCubeMapTexture.Sample(i_diffuseCubemapSamplerState, N).xyz;

        float3 prefilterEnv = i_prefilterEnvCubeMapTexture.SampleLevel(i_prefilterEnvCubeMapSamplerState,
                                                                       R, roughness * maxMipLevel).xyz;

        float2 envBrdf = i_envBrdfTexture.Sample(i_envBrdfSamplerState, float2(NoV, roughness)).xy;

        float3 Ks = fresnelSchlickRoughness(NoV, F0, roughness);
        float3 Kd = float3(1.0, 1.0, 1.0) - Ks;
        Kd *= (1.0 - metallic);

        float3 diffuse = Kd * diffuseIrradiance * Albedo;
        float3 specular = prefilterEnv * (Ks * envBrdf.x + envBrdf.y);

        IBLo = diffuse + specular;
    }
    IBLo *= pushData.params.y;

    float3 color = IBLo + Lo;
	
    // Gamma Correction
    // color = color / (color + float3(1.0, 1.0, 1.0));
    // color = pow(color, float3(1.0/2.2, 1.0/2.2, 1.0/2.2));  

	return float4(color, 1.0);
}
#endif
