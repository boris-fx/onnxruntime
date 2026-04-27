#if !defined(scalar_t)
#define scalar_t float
#endif

RWStructuredBuffer<scalar_t> feats      : register(u0); // n, n_ch_feats, h, w
RWStructuredBuffer<scalar_t> flow_1     : register(u1); // n, 2, h, w
RWStructuredBuffer<scalar_t> flow_2     : register(u2); // n, 2, h, w
RWStructuredBuffer<scalar_t> out_offset : register(u3); // n, 2 * n_ch_feats // 3, h, w
RWStructuredBuffer<scalar_t> out_mask   : register(u4); // n, n_ch_feats // 3, h, w

#include "second_order_deform_offset_mask_shader_constants.h"

#include "shader_util.h"

[numthreads(16, 16, 1)]
void second_order_deform_offset_mask(uint3 dtid : SV_DispatchThreadId)
{
    const int x = dtid.x;
    const int y = dtid.y;
    const int b_and_ch = dtid.z;
    const int b = b_and_ch / n_ch_feats;
    const int ch = b_and_ch % n_ch_feats;

    if (x >= w || y >= h || ch >= n_ch_feats || b >= n) return;

    const int n_ch_mask = n_ch_feats / 3;
    const int n_ch_offset = n_ch_mask * 2;

    if (ch < n_ch_offset) {

        const int flow_idx =
            (b * 2 * h * w) +
            (((ch + 1) % 2) * h * w) +
            (y * w) +
            x;
        scalar_t flow = (ch < (n_ch_offset / 2)) ? flow_1[flow_idx] : flow_2[flow_idx];

        const float offset_feat = (float)feats[
            (b * n_ch_feats * h * w) +
            (ch * h * w) +
            (y * w) +
            x];

        out_offset[
            (b * n_ch_offset * h * w) +
            (ch * h * w) +
            (y * w) +
            x] = flow + (scalar_t)(max_residue_magnitude * tanh(offset_feat));

    } else {

        const int ch_mask = ch - n_ch_offset;
        // calculate mask for given output element!

        scalar_t v = feats[
            (b * n_ch_feats * h * w) +
            (ch * h * w) +
            (y * w) +
            x];

        out_mask[
            (b * n_ch_mask * h * w) +
            (ch_mask * h * w) +
            (y * w) +
            x] = (scalar_t)sigmoidf((float)v);
    }

    return;
}
