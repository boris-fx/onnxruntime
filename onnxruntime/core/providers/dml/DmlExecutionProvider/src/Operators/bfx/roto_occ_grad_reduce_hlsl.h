#ifndef BFX_HLSL_ROTO_OCC_GRAD_REDUCE
#define BFX_HLSL_ROTO_OCC_GRAD_REDUCE

#define ROTO_OGR_BLOCK 256

#ifdef __cplusplus
namespace roto_occ_grad_reduce_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int F;
    int S;
    int blocksX;   // how many partial rows per frame there are to sum
};

#ifdef __cplusplus

#include "GeneratedShaders/roto_occ_grad_reduce.h"
const static inline ComputeShaderConfig cfg{
    g_roto_occ_grad_reduce, sizeof(g_roto_occ_grad_reduce), // bytecode
    5, // n bindings: partials, d_curve, own_loss, own_weight, own_total
    3  // n constants
};

} // roto_occ_grad_reduce_hlsl
#endif

#endif
