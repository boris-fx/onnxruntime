#ifndef BFX_HLSL_SCAN_CLEAR_BUFFER
#define BFX_HLSL_SCAN_CLEAR_BUFFER

#ifdef __cplusplus
namespace scan_clear_buffer_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int32_t n;
};

#ifdef __cplusplus

#include "../GeneratedShaders/scan_clear_buffer_int32.h"
const static inline ComputeShaderConfig cfg{
    g_scan_clear_buffer, sizeof(g_scan_clear_buffer), // bytecode
    1, // n bindings
    1  // n constants
};

} // rle_encode_get_diffs
#endif

#endif
