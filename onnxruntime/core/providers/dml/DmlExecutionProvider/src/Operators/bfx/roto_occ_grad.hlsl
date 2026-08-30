// bfx::roto_OccGrad - the whole of refine's pass 2 downstream of the curve, as ONE shader: the
// nearest-segment scan, soft occupancy, ONE of the two fit terms, the rasterizer's entire backward,
// and the gradient scatter, in O(1) state per query point.
//
// The arithmetic is a transcription of src/Backend/customops/cuda/roto_occ_grad.cu in the ml_sdk
// repo, and that file's header carries the closed form and the reasoning. THE SCATTER IS NOT a
// transcription - see roto_occ_grad_hlsl.h for why HLSL's lack of a float atomic forces a
// different structure, and what this does instead.
//
// The three-level funnel, widest first:
//
//   1. WAVE. Lanes that picked the same nearest segment combine via WaveActiveSum before anything
//      is written. Query points are raster-ordered and 1 px apart near the boundary, so a whole
//      wave usually collapses to one or two emits.
//   2. GROUPSHARED. What survives goes into a per-block slab by compare-and-swap - cheap, because
//      it is groupshared latency and the contention is one group of 256 threads onto S*2 slots.
//   3. PARTIALS. Each block owns a private row of a temporary and writes it with PLAIN STORES.
//      There is no global atomic anywhere in this shader. roto_occ_grad_reduce.hlsl sums the rows.
//
// WHY THIS IS SAFE TO WRITE BY HAND, given that it IS a gradient: the op is never differentiated.
// It is called to PRODUCE a gradient, which the graph then pushes through torch.func.vjp of the
// cheap upstream half. There is no backward-of-a-backward to get wrong.
//
// PRECISION: see the note in roto_union_forward.hlsl. -Gis is the lever if the numbers drift.

RWStructuredBuffer<float> curve          : register(u0);  // (F,S,2)
RWStructuredBuffer<float> ptsAll         : register(u1);  // (Q,P,2)
RWStructuredBuffer<uint>  own            : register(u2);  // (F,P)   uint8, packed 4/word
RWStructuredBuffer<uint>  present        : register(u3);  // (F)     uint8, packed 4/word
RWStructuredBuffer<uint>  tgtAll         : register(u4);  // (Q,P)   uint8, packed 4/word
RWStructuredBuffer<uint>  wgtAll         : register(u5);  // (Q,P)   float16, packed 2/word
RWStructuredBuffer<float> log1mAll       : register(u6);  // (Q,P)
RWStructuredBuffer<float> dlduAll        : register(u7);  // (Q,P)
RWStructuredBuffer<uint2> posIdx         : register(u8);  // (F) int64, low word read
RWStructuredBuffer<float> sharpnessBuf   : register(u9);  // rank-0
RWStructuredBuffer<float> neighborWtBuf  : register(u10); // rank-0
RWStructuredBuffer<float> ownWtBuf       : register(u11); // rank-0
RWStructuredBuffer<float> ownTotalBuf    : register(u12); // rank-0
RWByteAddressBuffer       partials       : register(u13); // (blocksX, F, S*2 + 1)

#include "roto_occ_grad_hlsl.h"
#include "roto_shader_util.h"

// tile[k] is polygon point s0 + k, with ONE EXTRA staged: a tile of n points describes only n-1
// complete segments, and the polygon is closed, so the last tile's halo wraps to point 0.
groupshared float tileX[ROTO_OG_TILE + 1];
groupshared float tileY[ROTO_OG_TILE + 1];

// The gradient slab, as float BIT PATTERNS - HLSL's InterlockedCompareExchange is integer-typed,
// so the accumulator has to be uint and the add happens through asfloat/asuint.
groupshared uint  slab[ROTO_OG_MAX_SLAB_FLOATS];

// the own term's per-block partial, tree-reduced at the end
groupshared float red[ROTO_OG_BLOCK];

// Float atomic add into the groupshared slab, by compare-and-swap. Cheap as these go: groupshared
// latency, and contention limited to one group of 256 threads (already thinned by the wave
// aggregation) over S*2 slots.
void roto_og_slab_add(uint idx, float value)
{
    uint expected = slab[idx];
    [loop]
    while (true) {
        const uint desired = asuint(asfloat(expected) + value);
        uint prev;
        InterlockedCompareExchange(slab[idx], expected, desired, prev);
        if (prev == expected) { break; }
        expected = prev;
    }
}

// The same, into this block's PRIVATE row of the partials temporary. Only reached when S is too
// large for the slab; still contended within one group only, never across the grid. Written
// against the global rather than taking the buffer as a parameter - see roto_shader_util.h.
void roto_og_partial_add(uint idx, float value)
{
    const uint at = idx * 4u;
    uint expected = partials.Load(at);
    [allow_uav_condition] [loop]
    while (true) {
        const uint desired = asuint(asfloat(expected) + value);
        uint prev;
        partials.InterlockedCompareExchange(at, expected, desired, prev);
        if (prev == expected) { break; }
        expected = prev;
    }
}

[numthreads(ROTO_OG_BLOCK, 1, 1)]
void roto_occ_grad(
    uint3 gid  : SV_GroupID,
    uint3 gtid : SV_GroupThreadID,
    uint3 dtid : SV_DispatchThreadID)
{
    const uint f = gid.y;                      // one frame per block row
    const uint slots = (uint)S * 2u;           // dCurve slots for this frame
    const uint rowStride = slots + 1u;         // + the block's own-loss partial
    const uint rowBase = (gid.x * (uint)F + f) * rowStride;   // this block's private row

    // ---- zero this block's accumulator ------------------------------------------------------
    // Either the slab or, when S is too large for it, the block's own partial row directly. The
    // row is private to this block, so even the no-slab path never contends across the grid.
    if (useSlab != 0) {
        for (uint i = gtid.x; i < slots; i += ROTO_OG_BLOCK) { slab[i] = 0u; }
    } else {
        for (uint i = gtid.x; i < slots; i += ROTO_OG_BLOCK) {
            partials.Store((rowBase + i) * 4u, 0u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint qRow = posIdx[f].x;             // row of the (Q,P) stacks this frame reads
    const float sharpness = sharpnessBuf[0];
    const uint curveF = f * (uint)S * 2u;

    float lossAcc = 0.0f;

    // ---- grid-stride over the query points ---------------------------------------------------
    // pBase depends only on gid.x, S, P and blocksX, never on gtid, so the trip count is uniform
    // across the group and the barriers inside are reached by every thread.
    const uint pStride = (uint)blocksX * ROTO_OG_BLOCK;
    for (uint pBase = gid.x * ROTO_OG_BLOCK; pBase < (uint)P; pBase += pStride) {

        const uint p = pBase + gtid.x;
        const bool active = p < (uint)P;

        const uint oq = qRow * (uint)P + p;    // the gather that used to be five separate nodes
        const uint o  = f * (uint)P + p;       // own is (F,P), indexed by the frame itself

        float px = 0.0f;
        float py = 0.0f;
        if (active) {
            px = ptsAll[oq * 2u];
            py = ptsAll[oq * 2u + 1u];
        }

        // ---- the scan: nearest segment and inside/outside, fused, one pass over the curve ----
        float bestD2 = ROTO_FLT_MAX;
        int bestIdx = 0;
        int crossings = 0;

        for (int t0 = 0; t0 < S; t0 += ROTO_OG_TILE) {
            const int n = min(ROTO_OG_TILE, S - t0);

            for (uint k = gtid.x; k <= (uint)n; k += ROTO_OG_BLOCK) {
                int src = t0 + (int)k;
                if (src >= S) { src -= S; }    // the closing wrap
                tileX[k] = curve[curveF + (uint)src * 2u];
                tileY[k] = curve[curveF + (uint)src * 2u + 1u];
            }
            GroupMemoryBarrierWithGroupSync();

            if (active) {
                for (int k = 0; k < n; k++) {
                    const float ax = tileX[k];
                    const float ay = tileY[k];
                    const float bx = tileX[k + 1];
                    const float by = tileY[k + 1];

                    const float ex = bx - ax;
                    const float ey = by - ay;
                    float len2 = ex * ex + ey * ey;
                    if (len2 < ROTO_SEG_LEN2_MIN) { len2 = ROTO_SEG_LEN2_MIN; }

                    float tt = ((px - ax) * ex + (py - ay) * ey) / len2;
                    tt = clamp(tt, 0.0f, 1.0f);

                    const float dx = px - (ax + tt * ex);
                    const float dy = py - (ay + tt * ey);
                    const float d2 = dx * dx + dy * dy;

                    // STRICTLY less, scanning s ascending, so the FIRST minimum wins - which is
                    // what torch's argmin returns. Ties are not cosmetic: two segments equidistant
                    // from a point send the gradient to DIFFERENT control points.
                    if (d2 < bestD2) {
                        bestD2 = d2;
                        bestIdx = t0 + k;
                    }

                    if ((ay > py) != (by > py)) {
                        const float xc = ax + (py - ay) * (bx - ax) / ((by - ay) + ROTO_CROSS_EPS);
                        if (px < xc) { crossings++; }
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // ---- the closed form ------------------------------------------------------------------
        int s0 = 0;
        int s1 = 0;
        float4 dp = float4(0.0f, 0.0f, 0.0f, 0.0f);   // dp0x, dp0y, dp1x, dp1y
        bool emit = false;

        if (active) {
            // Re-read the winning segment rather than carrying it out of the scan. Four floats out
            // of cache, and it keeps this block textually identical to the gathered formulation in
            // refine_graph.occupancy - which is the thing it was verified against.
            s0 = bestIdx;
            s1 = s0 + 1;
            if (s1 >= S) { s1 = 0; }

            const float ax = curve[curveF + (uint)s0 * 2u];
            const float ay = curve[curveF + (uint)s0 * 2u + 1u];
            const float bx = curve[curveF + (uint)s1 * 2u];
            const float by = curve[curveF + (uint)s1 * 2u + 1u];

            const float ex = bx - ax;
            const float ey = by - ay;
            const float segLen2 = ex * ex + ey * ey;
            const float len2 = segLen2 < ROTO_SEG_LEN2_MIN ? ROTO_SEG_LEN2_MIN : segLen2;
            const float wx = px - ax;
            const float wy = py - ay;
            const float tRaw = (wx * ex + wy * ey) / len2;
            const float tc = clamp(tRaw, 0.0f, 1.0f);
            const float rx = px - (ax + tc * ex);
            const float ry = py - (ay + tc * ey);
            const float d2 = rx * rx + ry * ry;
            const float dist = sqrt(d2 < ROTO_DIST_EPS ? ROTO_DIST_EPS : d2);
            const bool ins = (crossings & 1) != 0;
            const float occ = 1.0f / (1.0f + exp(sharpness * (ins ? -dist : dist)));

            // ---- dL/docc: ONE of the two fit terms ----
            float dOcc;
            if (ext != 0) {
                // log(1 - x), NOT log1p(-x). This cancels against a log1m_tot produced by pass 1,
                // and bfx::roto_UnionForward spells it the same way. A different spelling here
                // leaves behind exactly the residue the subtraction exists to remove.
                const float occC = occ > ROTO_OCC_MAX ? ROTO_OCC_MAX : occ;
                dOcc = dlduAll[oq] * exp(log1mAll[oq] - log(1.0f - occC));
                // lossAcc is untouched, so ext mode contributes nothing to own_loss - no branch
                // needed for it, and no stale value either.
            } else {
                const uint ownV = roto_load_u8(own, o);
                // `wgt > 0` is the ONLY thing the loss ever asks of the weight, so this never
                // converts a half to float - see roto_shader_util.h.
                const bool wpos = roto_f16_positive(roto_load_f16_bits(wgtAll, oq));

                float ow = (ownV == 0u && roto_load_u8(tgtAll, oq) > 0u) ? neighborWtBuf[0] : 1.0f;
                if (!wpos || roto_load_u8(present, f) == 0u) { ow = 0.0f; }

                // the same residual the gradient is built from, squared - _own_term's summand
                const float resid = occ - (float)ownV;
                lossAcc += ow * resid * resid;
                dOcc = ow * 2.0f * resid * ownWtBuf[0] / ownTotalBuf[0];
            }

            // ---- backward through the rasterizer ----
            const float dSigned = dOcc * occ * (1.0f - occ) * (-sharpness);
            const float dDist = ins ? -dSigned : dSigned;
            // clamp_min's backward, INCLUSIVE at equality - torch passes gradient where
            // input >= min, and a strict > here would differ exactly where a query point lands on
            // the curve
            const float dD2 = (d2 >= ROTO_DIST_EPS) ? dDist * 0.5f / dist : 0.0f;
            const float dprojx = -2.0f * rx * dD2;
            const float dprojy = -2.0f * ry * dD2;

            // clamp(0,1)'s backward, inclusive at BOTH ends for the same reason
            float gT = dprojx * ex + dprojy * ey;
            if (tRaw < 0.0f || tRaw > 1.0f) { gT = 0.0f; }

            const float kk = gT / len2;
            const float maskL = (segLen2 >= ROTO_SEG_LEN2_MIN) ? 1.0f : 0.0f;
            const float dwx = kk * ex;
            const float dwy = kk * ey;
            // t_raw = (w.seg)/L, and seg appears in the numerator AND, through |seg|^2, in L
            const float dsegx = kk * (wx - 2.0f * tRaw * ex * maskL);
            const float dsegy = kk * (wy - 2.0f * tRaw * ey * maskL);

            // w = pts - p0 and seg = p1 - p0, so both subtract from p0
            dp.x = (1.0f - tc) * dprojx - dwx - dsegx;
            dp.y = (1.0f - tc) * dprojy - dwy - dsegy;
            dp.z = tc * dprojx + dsegx;
            dp.w = tc * dprojy + dsegy;
            emit = true;
        }

        // ---- level 1: wave aggregation --------------------------------------------------------
        // Called OUTSIDE the `if (active)` above, on purpose: the loop's conditions are wave-wide
        // reductions, so every lane of the wave has to reach them. Inactive lanes enter with
        // emit == false and drop out on the first iteration without contributing.
#if ROTO_OG_WAVE
        {
            bool done = !emit;
            [loop]
            while (WaveActiveAnyTrue(!done)) {
                // WaveActiveMin over a sentinel picks a key that some not-yet-done lane holds.
                // WaveReadLaneFirst cannot be used for this - it reads the lowest lane active in
                // CONTROL FLOW, which here includes the lanes that are already done.
                const int key = WaveActiveMin(done ? 0x7FFFFFFF : s0);
                const bool match = !done && (s0 == key);

                const float4 sum = float4(
                    WaveActiveSum(match ? dp.x : 0.0f),
                    WaveActiveSum(match ? dp.y : 0.0f),
                    WaveActiveSum(match ? dp.z : 0.0f),
                    WaveActiveSum(match ? dp.w : 0.0f));

                // exactly one lane of the matching set writes
                if (match && WavePrefixCountBits(match) == 0u) {
                    int k1 = key + 1;
                    if (k1 >= S) { k1 = 0; }
                    if (useSlab != 0) {
                        roto_og_slab_add((uint)key * 2u,      sum.x);
                        roto_og_slab_add((uint)key * 2u + 1u, sum.y);
                        roto_og_slab_add((uint)k1  * 2u,      sum.z);
                        roto_og_slab_add((uint)k1  * 2u + 1u, sum.w);
                    } else {
                        roto_og_partial_add(rowBase + (uint)key * 2u,      sum.x);
                        roto_og_partial_add(rowBase + (uint)key * 2u + 1u, sum.y);
                        roto_og_partial_add(rowBase + (uint)k1  * 2u,      sum.z);
                        roto_og_partial_add(rowBase + (uint)k1  * 2u + 1u, sum.w);
                    }
                }
                done = done || match;
            }
        }
#else
        if (emit) {
            if (useSlab != 0) {
                roto_og_slab_add((uint)s0 * 2u,      dp.x);
                roto_og_slab_add((uint)s0 * 2u + 1u, dp.y);
                roto_og_slab_add((uint)s1 * 2u,      dp.z);
                roto_og_slab_add((uint)s1 * 2u + 1u, dp.w);
            } else {
                roto_og_partial_add(rowBase + (uint)s0 * 2u,      dp.x);
                roto_og_partial_add(rowBase + (uint)s0 * 2u + 1u, dp.y);
                roto_og_partial_add(rowBase + (uint)s1 * 2u,      dp.z);
                roto_og_partial_add(rowBase + (uint)s1 * 2u + 1u, dp.w);
            }
        }
#endif

        // the tile staging at the top of the next p-chunk overwrites tileX/tileY, so nothing may
        // still be reading them
        GroupMemoryBarrierWithGroupSync();
    }

    // ---- level 3: flush this block's private partial row -------------------------------------
    // PLAIN STORES, not atomics: the row belongs to this block alone. Zeros are written too, so
    // the reduce can sum unconditionally and nothing has to be cleared beforehand.
    if (useSlab != 0) {
        GroupMemoryBarrierWithGroupSync();
        for (uint i = gtid.x; i < slots; i += ROTO_OG_BLOCK) {
            partials.Store((rowBase + i) * 4u, slab[i]);
        }
    }

    // ---- the own term's value: block tree reduce, then one store ------------------------------
    // NOT scaled by ownWeight/ownTotal here. roto_occ_grad.cu applies the scale per block because
    // it has no second pass to apply it in; this does, so the scale happens once at the end in the
    // reduce - which is _own_term's own sum-then-scale order, and therefore very slightly closer
    // to the reference than the CUDA kernel is.
    GroupMemoryBarrierWithGroupSync();
    red[gtid.x] = lossAcc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = ROTO_OG_BLOCK / 2u; s > 0u; s >>= 1) {
        if (gtid.x < s) { red[gtid.x] += red[gtid.x + s]; }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gtid.x == 0) {
        partials.Store((rowBase + slots) * 4u, asuint(red[0]));
    }
}
