#ifndef BFX_HLSL_RLE_ENCODE_WRITE_OUT
#define BFX_HLSL_RLE_ENCODE_WRITE_OUT

#ifdef __cplusplus
namespace rle_encode_write_out_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int n;
};

#ifdef __cplusplus
#include "GeneratedShaders/rle_encode_write_out_int32_int32.h"
const static inline ComputeShaderConfig cfg{
    g_rle_encode_write_out, sizeof(g_rle_encode_write_out), // bytecode
    4, // n bindings
    1  // n constants
};
}
#endif

#endif
