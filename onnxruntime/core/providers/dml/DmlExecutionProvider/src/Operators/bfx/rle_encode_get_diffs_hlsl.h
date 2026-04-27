#ifndef BFX_HLSL_RLE_ENCODE_GET_DIFFS
#define BFX_HLSL_RLE_ENCODE_GET_DIFFS

#ifdef __cplusplus
namespace rle_encode_get_diffs_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int n;
};

#ifdef __cplusplus
#include "GeneratedShaders/rle_encode_get_diffs_int32_int32.h"
const static inline ComputeShaderConfig cfg{
    g_rle_encode_get_diffs, sizeof(g_rle_encode_get_diffs), // bytecode
    3, // n bindings
    1  // n constants
};
}
#endif

#endif
