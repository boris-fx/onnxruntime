#ifndef BFX_HLSL_ROTO_OCC_GRAD
#define BFX_HLSL_ROTO_OCC_GRAD

// Thread group width, and polygon points staged per pass. Shared between the shader and the C++
// that computes the grid so the two cannot drift.
#define ROTO_OG_BLOCK 256
#define ROTO_OG_TILE  256

// HOW MANY PARTIAL ACCUMULATORS THIS OP IS ALLOWED, and the reason the whole aggregation scheme
// differs from the CUDA kernel's.
//
// roto_occ_grad.cu dispatches (ceil(P/256), F) and has every block flush its slab with global
// atomicAdd. That works there because CUDA has a HARDWARE float atomic add. HLSL does not - through
// SM 6.x InterlockedAdd is integer only - so the same structure here would mean ~580k
// compare-and-swap retry loops onto 64 addresses per frame, which degrades far worse than a
// hardware atomic under the same contention.
//
// So the grid width is CAPPED and each block grid-strides over P instead. That turns the partial
// count from a function of P into a constant, which makes it affordable to give every block its
// own private row in a temporary and REDUCE, rather than atomically accumulate. The result is
// zero global atomics, and a deterministic op - which the CUDA one is not.
//
// The cost is the temporary: blocksX * F * (S*2 + 1) floats. At the tuned size (S=32, F=64, this
// cap) that is ~1 MB. Re-staging the curve tile once per p-chunk is the other cost, and it is
// nothing: S/256 tiles of S+1 loads against 256 threads x S segments of real work.
//
// 64 is a starting point, not a measured optimum - it is one of the two numbers worth sweeping
// once this runs (the other is whether the wave aggregation below is pulling its weight).
#define ROTO_OG_MAX_BLOCKS_X 64

// Groupshared slab capacity, in floats. D3D12 caps TOTAL groupshared at 32 KB
// (D3D12_CS_TGSM_REGISTER_COUNT = 8192 32-bit values), and this shader also needs
// tileX/tileY (257*2 floats = 2056 B) and red (256 floats = 1024 B). So the ceiling is 7422
// floats - but CUDA's kMaxSlabFloats of 8192 would not have fit anyway (it assumes a 48 KB
// budget), and anything near the ceiling would wreck occupancy for no gain: the tuned configs run
// S = 32..128, i.e. 64..256 slots. 2048 floats covers S <= 1024 at 8 KB, leaving total
// groupshared at ~11 KB.
//
// Above it, `useSlab` is 0 and the block CASes straight into its own private partial row - still
// contended only within one group of 256 threads, never across the grid.
#define ROTO_OG_MAX_SLAB_FLOATS 2048

// Wave-level aggregation before anything is written. Lanes that picked the same nearest segment
// combine first, so one lane emits their sum instead of each lane emitting its own.
//
// THIS IS WORTH IT BECAUSE THE QUERY POINTS ARE SPATIALLY COHERENT. query.build_adaptive_points
// ends in np.nonzero(), which returns raster order, and the finest tier is stride 1 within 3 px of
// the boundary - so lane-adjacent threads are horizontally adjacent pixels, and with S = 32
// samples around a whole shape each segment owns a long run of boundary. Adjacent lanes therefore
// share a bestIdx nearly always, and the loop below collapses a whole wave into ~1-3 emits.
//
// Set to 0 to compile the aggregation out - it is the first thing to A/B, and the fallback if a
// device turns out not to report D3D12_FEATURE_DATA_D3D12_OPTIONS1::WaveOps. Wave intrinsics are
// SM 6.0, so they are already inside the cs_6_2 profile every bfx shader is built with; the
// device capability is the only new requirement this port introduces.
#ifndef ROTO_OG_WAVE
#define ROTO_OG_WAVE 1
#endif

#ifdef __cplusplus
namespace roto_occ_grad_hlsl {
struct constants
#else
cbuffer Constants
#endif

{
    int F;        // this object's frames; curve is (F,S,2), own is (F,P), pos_idx is (F)
    int S;        // curve samples per frame, a CLOSED polygon
    int P;        // query points per row
    int ext;      // 1 = fit the union against the external matte, 0 = fit own masks. NEVER both.
    int blocksX;  // grid width actually dispatched; the p-stride is blocksX * ROTO_OG_BLOCK
    int useSlab;  // 1 = aggregate in groupshared, 0 = CAS into this block's own partial row
};

#ifdef __cplusplus

#include "GeneratedShaders/roto_occ_grad.h"
const static inline ComputeShaderConfig cfg{
    g_roto_occ_grad, sizeof(g_roto_occ_grad), // bytecode
    14, // n bindings: 13 inputs + the partials temporary
    6   // n constants
};

} // roto_occ_grad_hlsl
#endif

#endif
