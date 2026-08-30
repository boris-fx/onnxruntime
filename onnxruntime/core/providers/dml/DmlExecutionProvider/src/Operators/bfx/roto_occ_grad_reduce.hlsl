// bfx::roto_OccGrad, pass 2 of 2: sum the per-block partial rows into the real outputs.
//
// This pass is the whole reason roto_occ_grad.hlsl contains no global atomic. Each block there
// owned a private row of `partials`, laid out (blocksX, F, S*2 + 1) - the trailing slot of each
// row being that block's own-term partial - and wrote it with plain stores. All that is left is a
// sum over the blocksX rows belonging to each frame.
//
// It also makes the op DETERMINISTIC, which the CUDA kernel is not: the summation order here is
// fixed by the loop rather than by whichever block reached the atomic first. That is a real
// property to have when the thing being compared against is a stored reference.
//
// The outputs are WRITTEN, not accumulated, so nothing needs to be zeroed first - which is what
// removes the cudaMemsetAsync the CUDA launcher has to do (and the descriptor-heap dance
// ClearUnorderedAccessViewFloat would have needed here).

RWByteAddressBuffer       partials  : register(u0); // (blocksX, F, S*2 + 1)
RWStructuredBuffer<float> dCurve    : register(u1); // (F,S,2)
RWStructuredBuffer<float> ownLoss   : register(u2); // (1)
RWStructuredBuffer<float> ownWtBuf  : register(u3); // rank-0
RWStructuredBuffer<float> ownTotBuf : register(u4); // rank-0

#include "roto_occ_grad_reduce_hlsl.h"

[numthreads(ROTO_OGR_BLOCK, 1, 1)]
void roto_occ_grad_reduce(uint3 dtid : SV_DispatchThreadID)
{
    const uint slots = (uint)S * 2u;
    const uint rowStride = slots + 1u;
    const uint total = (uint)F * slots;

    const uint idx = dtid.x;

    if (idx < total) {
        // idx enumerates dCurve's (F, S*2). Each output sums the same slot across blocksX rows.
        const uint f = idx / slots;
        const uint slot = idx - f * slots;

        float acc = 0.0f;
        for (uint b = 0; b < (uint)blocksX; b++) {
            const uint at = ((b * (uint)F + f) * rowStride + slot) * 4u;
            acc += asfloat(partials.Load(at));
        }
        dCurve[idx] = acc;
        return;
    }

    // ---- the own term's value ----------------------------------------------------------------
    // One thread, summing blocksX * F partials - a few thousand loads at the working size, which
    // is microseconds and not worth a second dispatch to parallelise.
    //
    // THE SCALE IS APPLIED ONCE, HERE. roto_occ_grad.cu multiplies by ownWeight/ownTotal per block
    // because it has no second pass to do it in; having one, this matches _own_term's own
    // sum-then-scale order instead. In ext mode every partial is zero, so own_loss comes out zero
    // with no branch and no stale value.
    if (idx == total) {
        float acc = 0.0f;
        for (uint b = 0; b < (uint)blocksX; b++) {
            for (uint f = 0; f < (uint)F; f++) {
                const uint at = ((b * (uint)F + f) * rowStride + slots) * 4u;
                acc += asfloat(partials.Load(at));
            }
        }
        // Guarded exactly as roto_occ_grad.cu guards its atomic (`if (red[0] != 0.0f)`), and for a
        // reason that only shows up here: in EXT MODE own_weight and own_total are dummy bindings
        // whose contents are whatever the caller had lying around - the kernel is entitled not to
        // read them. acc is zero there, so without this an own_total of zero would turn a correct
        // 0 into 0/0 = NaN and poison the graph's own_loss output.
        ownLoss[0] = (acc != 0.0f) ? (acc * ownWtBuf[0] / ownTotBuf[0]) : 0.0f;
    }
}
