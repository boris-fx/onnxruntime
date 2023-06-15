#ifndef BFX_HLSL_RLE_DECODE_SCATTER
#define BFX_HLSL_RLE_DECODE_SCATTER

#ifdef __cplusplus
namespace rle_decode_scatter_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int n;
};

#ifdef __cplusplus
#include "GeneratedShaders/rle_decode_scatter_int32_int32.h"
const static inline ComputeShaderConfig cfg{
    g_rle_decode_scatter, sizeof(g_rle_decode_scatter), // bytecode
    4, // n bindings
    1  // n constants
};
} // rle_encode_get_diffs
#endif

#endif
