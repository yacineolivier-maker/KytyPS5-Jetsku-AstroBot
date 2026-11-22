#version 450
#extension GL_GOOGLE_include_directive : require

// Astro Bot PC (KYTY_PRESENT_AA=smaa) : nuanceur de sommets des trois passes de SMAA 1x
// (3rdparty/smaa/SMAA.hlsl, Jimenez et al., licence MIT), sur le triangle plein ecran.
// SMAA_PASS (ligne de commande) : 0 detection des bords, 1 poids de melange, 2 melange du voisinage.
// SMAA_PRESET_HIGH ou SMAA_PRESET_ULTRA vient aussi de la ligne de commande.

layout(push_constant) uniform SmaaConstants {
	vec4 metrics; // (1 / largeur, 1 / hauteur, largeur, hauteur) de l'image du jeu
} smaa_constants;

#define SMAA_RT_METRICS smaa_constants.metrics
#define SMAA_GLSL_4
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 0
#include "SMAA.hlsl"

layout(location = 0) out vec2 texcoord;
#if SMAA_PASS == 0
layout(location = 1) out vec4 offset[3];
#elif SMAA_PASS == 1
layout(location = 1) out vec2 pixcoord;
layout(location = 2) out vec4 offset[3];
#else
layout(location = 1) out vec4 offset;
#endif

void main() {
	// Vulkan : (-1, -1) est le coin haut gauche, comme Direct3D pour lequel SMAA est ecrit ;
	// AreaTex et SearchTex sont donc lues telles quelles, sans retournement.
	const vec2 position = vec2(float((gl_VertexIndex & 1) << 2), float((gl_VertexIndex & 2) << 1));
	gl_Position         = vec4(position - vec2(1.0), 0.0, 1.0);
	texcoord            = position * 0.5;
#if SMAA_PASS == 0
	SMAAEdgeDetectionVS(texcoord, offset);
#elif SMAA_PASS == 1
	SMAABlendingWeightCalculationVS(texcoord, pixcoord, offset);
#else
	SMAANeighborhoodBlendingVS(texcoord, offset);
#endif
}
