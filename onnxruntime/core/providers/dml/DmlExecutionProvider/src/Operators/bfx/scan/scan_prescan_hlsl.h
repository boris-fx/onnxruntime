#ifndef BFX_HLSL_SCAN_PRESCAN
#define BFX_HLSL_SCAN_PRESCAN

#ifdef __cplusplus
namespace scan_prescan_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int n;
};

#ifdef __cplusplus
#include "GeneratedShaders/scan_prescan_int32.h"
const static inline ComputeShaderConfig cfg{
    g_scan_prescan, sizeof(g_scan_prescan), // bytecode
    3, // n bindings
    1  // n constants
};
} // rle_encode_get_diffs
#endif

#endif
