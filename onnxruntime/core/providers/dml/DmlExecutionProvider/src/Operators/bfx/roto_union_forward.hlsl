// bfx::roto_UnionForward - refine's pass 1 for one object, as ONE shader.
//
// A transcription of src/Backend/customops/cuda/roto_union_forward.cu in the ml_sdk repo; that
// file's header carries the reasoning for why this is a kernel rather than a chain of tensor ops,
// and everything it says about fusing the nearest-segment scan, the distance and the query gather
// applies here unchanged. Only the mechanics differ, and only where HLSL has no analogue:
//
//   __shared__           -> groupshared
//   __syncthreads()      -> GroupMemoryBarrierWithGroupSync()
//   blockIdx / threadIdx -> SV_GroupID / SV_GroupThreadID / SV_DispatchThreadID
//   int64 pos_idx        -> bound as uint2, low word read (indices are small and non-negative,
//                           and HLSL has no 64-bit integer without SM 6.0 Int64Ops support)
//   expf / logf / sqrtf  -> exp / log / sqrt   (SEE THE PRECISION NOTE BELOW)
//
// There are NO atomics here and nothing is accumulated across threads - each (q,p) writes exactly
// one output element. That is what makes this the right op to port first: it exercises the schema,
// the registration, the int64 read, the tile loop and the arithmetic, with none of the scatter
// machinery roto_OccGrad needs.
//
// PRECISION. DXC lowers exp(x) to exp2(x * 1.44269504) and may contract a divide into a
// reciprocal-multiply, so this will not be bit-identical to the CUDA kernel. It does not need to
// be - the shipped test cases allow 1e-4 relative, and log1m_out specifically allows 3e-3 because
// log(1-occ) is ill-conditioned as occ -> 1 (see refine_graph._ext_surrogate). If the numbers do
// drift past that, compile with -Gis (force IEEE strictness) BEFORE touching the arithmetic.
//
// Do NOT use shader_util.h::sigmoidf here: it is pow(EULER_NUMBER, -n), which is materially worse
// than exp() on exactly the quantity this shader is most sensitive to.

RWStructuredBuffer<float> log1mIn   : register(u0); // (Q,P) accumulator over the WHOLE range
RWStructuredBuffer<float> curve     : register(u1); // (F,S,2) closed polygon per frame
RWStructuredBuffer<float> ptsAll    : register(u2); // (Q,P,2) query for the whole range
RWStructuredBuffer<uint2> posIdx    : register(u3); // (F) int64, STRICTLY INCREASING
RWStructuredBuffer<float> sharpness : register(u4); // rank-0
RWStructuredBuffer<float> log1mOut  : register(u5); // (Q,P)

#include "roto_union_forward_hlsl.h"
#include "roto_shader_util.h"

// tile[k] is polygon point s0 + k, with ONE EXTRA staged: a tile of n points describes only n-1
// complete segments, and the polygon is closed, so the last tile's halo wraps to point 0.
groupshared float tileX[ROTO_UF_TILE + 1];
groupshared float tileY[ROTO_UF_TILE + 1];
groupshared int   sFrame;

// Which of this object's frames renders into range row q, or -1 if it does not cover it.
// Binary search, which is what requires pos_idx to be strictly increasing - true by construction
// on both sides. Reads the low word of each int64; see the header note.
int roto_uf_find_frame(uint q)
{
    int lo = 0;
    int hi = F - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint v = posIdx[mid].x;
        if (v == q) { return mid; }
        if (v < q) { lo = mid + 1; } else { hi = mid - 1; }
    }
    return -1;
}

[numthreads(ROTO_UF_BLOCK, 1, 1)]
void roto_union_forward(
    uint3 gid  : SV_GroupID,
    uint3 gtid : SV_GroupThreadID,
    uint3 dtid : SV_DispatchThreadID)
{
    // The grid is over the OUTPUT (Q,P), not over this object's (F,P). That is what makes this one
    // pass instead of copy-then-accumulate: each group resolves its row q back to a frame and then
    // either renders into it or copies it through, so the accumulator is read once and written
    // once.
    const uint q = gid.y;

    if (gtid.x == 0) { sFrame = roto_uf_find_frame(q); }
    GroupMemoryBarrierWithGroupSync();
    const int f = sFrame;   // uniform across the group, so every branch below is

    const uint p = dtid.x;
    const bool active = p < (uint)P;
    const uint o = q * (uint)P + p;

    // f depends only on gid.y, so this branch is GROUP-UNIFORM and the barriers inside the else
    // are reached by every thread of every group that takes it. HLSL's rule, like CUDA's, is per
    // group; this is the same structure roto_union_forward.cu has.
    if (f < 0) {
        // THE COPY-THROUGH IS LOAD-BEARING. Most objects cover a subrange of the clip, and a
        // shader that zeroed or skipped the rows it does not cover would corrupt the union for
        // every OTHER object - which reads as a fit going wrong rather than a kernel going wrong.
        // trace_refine.verify_union_forward pins it.
        if (active) { log1mOut[o] = log1mIn[o]; }
    } else {
        float px = 0.0f;
        float py = 0.0f;
        if (active) {
            px = ptsAll[o * 2u];
            py = ptsAll[o * 2u + 1u];
        }

        const uint curveF = (uint)f * (uint)S * 2u;

        float bestD2 = ROTO_FLT_MAX;
        int crossings = 0;

        for (int s0 = 0; s0 < S; s0 += ROTO_UF_TILE) {
            const int n = min(ROTO_UF_TILE, S - s0);

            for (uint k = gtid.x; k <= (uint)n; k += ROTO_UF_BLOCK) {
                int src = s0 + (int)k;
                if (src >= S) { src -= S; }     // the closing wrap
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

                    // ---- distance to this segment ----
                    const float ex = bx - ax;
                    const float ey = by - ay;
                    float len2 = ex * ex + ey * ey;
                    if (len2 < ROTO_SEG_LEN2_MIN) { len2 = ROTO_SEG_LEN2_MIN; }

                    float t = ((px - ax) * ex + (py - ay) * ey) / len2;
                    t = clamp(t, 0.0f, 1.0f);

                    const float dx = px - (ax + t * ex);
                    const float dy = py - (ay + t * ey);
                    const float d2 = dx * dx + dy * dy;

                    // Strictly less, scanning s ascending - the FIRST minimum, matching torch's
                    // argmin. No index is returned here, but keeping the rule identical is what
                    // lets this be checked against refine_graph's seg_nearest_native + occupancy
                    // rather than only against itself.
                    if (d2 < bestD2) { bestD2 = d2; }

                    // ---- crossing-number parity, same pass ----
                    if ((ay > py) != (by > py)) {
                        const float xc = ax + (py - ay) * (bx - ax) / ((by - ay) + ROTO_CROSS_EPS);
                        if (px < xc) { crossings++; }
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (active) {
            // bestD2 IS occupancy()'s d2 - the same expression on the same two endpoints - so the
            // gathered recompute the tensor version needs is simply not here.
            const float dist = sqrt(bestD2 < ROTO_DIST_EPS ? ROTO_DIST_EPS : bestD2);
            const float sh = sharpness[0];
            const bool inside = (crossings & 1) != 0;
            const float occ = 1.0f / (1.0f + exp(sh * (inside ? -dist : dist)));

            // log(1 - x), NOT log1p(-x). ONNX has no Log1p - torch.log1p lowers to Add(-occ, 1)
            // then Log - so this is what the graph being replaced actually computed, and it is
            // also what bfx::roto_OccGrad divides back out of this value in pass 2. The
            // cancellation _ext_surrogate depends on is only exact while the two are the same
            // expression.
            const float occC = occ > ROTO_OCC_MAX ? ROTO_OCC_MAX : occ;
            log1mOut[o] = log1mIn[o] + log(1.0f - occC);
        }
    }
}
