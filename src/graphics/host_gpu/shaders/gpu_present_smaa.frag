#version 450
#extension GL_GOOGLE_include_directive : require

// Astro Bot PC (KYTY_PRESENT_AA=smaa) : nuanceur de pixels des trois passes de SMAA 1x
// (3rdparty/smaa/SMAA.hlsl, Jimenez et al., licence MIT). Voir gpu_present_smaa.vert.
//   passe 0 : binding 0 = image du jeu (lineaire)            -> bords (R8G8)
//   passe 1 : binding 0 = bords, 1 = AreaTex, 2 = SearchTex (toutes lineaires, comme
//             l'echantillonneur LinearSampler de la version Direct3D 10 de reference)
//                                                            -> poids de melange (R8G8B8A8)
//   passe 2 : binding 0 = image du jeu (lineaire), 1 = poids (lineaire) -> image lissee

layout(push_constant) uniform SmaaConstants {
	vec4 metrics;
} smaa_constants;

#define SMAA_RT_METRICS smaa_constants.metrics
#define SMAA_GLSL_4
#define SMAA_INCLUDE_VS 0
#define SMAA_INCLUDE_PS 1
#include "SMAA.hlsl"

layout(location = 0) in vec2 texcoord;
#if SMAA_PASS == 0
layout(location = 1) in vec4 offset[3];
layout(binding = 0) uniform sampler2D color_tex;
layout(location = 0) out vec2 out_edges;
#elif SMAA_PASS == 1
layout(location = 1) in vec2 pixcoord;
layout(location = 2) in vec4 offset[3];
layout(binding = 0) uniform sampler2D edges_tex;
layout(binding = 1) uniform sampler2D area_tex;
layout(binding = 2) uniform sampler2D search_tex;
layout(location = 0) out vec4 out_weights;
#else
layout(location = 1) in vec4 offset;
layout(binding = 0) uniform sampler2D color_tex;
layout(binding = 1) uniform sampler2D blend_tex;
layout(location = 0) out vec4 out_color;
#endif

void main() {
#if SMAA_PASS == 0
	// Detection sur la luminance, l'image etant encodee pour l'affichage (gamma), comme le
	// recommande SMAA. Les pixels sans bord sont abandonnes (la cible est effacee a zero).
	out_edges = SMAALumaEdgeDetectionPS(texcoord, offset, color_tex);
#elif SMAA_PASS == 1
	out_weights = SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset, edges_tex, area_tex,
	                                              search_tex, vec4(0.0));
#else
	out_color = SMAANeighborhoodBlendingPS(texcoord, offset, color_tex, blend_tex);
#endif
}
