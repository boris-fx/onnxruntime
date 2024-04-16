#ifndef BFX_HLSL_SCAN_ADD_BLOCK_SUMS
#define BFX_HLSL_SCAN_ADD_BLOCK_SUMS

#ifdef __cplusplus
namespace scan_add_block_sums_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int32_t n;
};

#ifdef __cplusplus
#include "GeneratedShaders/scan_add_block_sums_int32.h"
const static inline ComputeShaderConfig cfg{
    g_scan_add_block_sums, sizeof(g_scan_add_block_sums), // bytecode
    2, // n bindings
    1  // n constants
};
} // rle_encode_get_diffs
#endif

#endif
