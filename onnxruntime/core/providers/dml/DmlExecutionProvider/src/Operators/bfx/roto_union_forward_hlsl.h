#ifndef BFX_HLSL_ROTO_UNION_FORWARD
#define BFX_HLSL_ROTO_UNION_FORWARD

// Thread group width, and the number of polygon points staged per pass. Shared between the shader
// and the C++ that computes the dispatch grid, so the two cannot drift - rle_encode.h duplicates
// its block size with a "// matches shader!" comment, which is the failure this avoids.
//
// kTile bounds groupshared use INDEPENDENTLY OF S: curveSamples is a host parameter and nothing
// caps it, so staging the whole curve would make occupancy scale with it. Same number and same
// reasoning as roto_union_forward.cu.
#define ROTO_UF_BLOCK 256
#define ROTO_UF_TILE  256

#ifdef __cplusplus
namespace roto_union_forward_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int F;   // this object's frame count; curve is (F,S,2) and pos_idx is (F)
    int S;   // curve samples per frame, a CLOSED polygon (segment s runs s -> (s+1) % S)
    int P;   // query points per row
};

#ifdef __cplusplus

#include "GeneratedShaders/roto_union_forward.h"
const static inline ComputeShaderConfig cfg{
    g_roto_union_forward, sizeof(g_roto_union_forward), // bytecode
    6, // n bindings: log1m_in, curve, pts_all, pos_idx, sharpness | log1m_out
    3  // n constants
};

} // roto_union_forward_hlsl
#endif

#endif
