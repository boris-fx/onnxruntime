#if !defined(scalar_t)
#define scalar_t float
#endif

RWStructuredBuffer<scalar_t> input  : register(u0); // n, n_ch, h, w
RWStructuredBuffer<scalar_t> flow   : register(u1); // n, 2, h, w
RWStructuredBuffer<scalar_t> output : register(u2); // n, n_ch, h, w

#include "warp_flow_shader_constants.h"

#include "shader_util.h"

[numthreads(16, 16, 1)]
void warp_flow(uint3 dtid : SV_DispatchThreadId)
{
    const int x = dtid.x;
    const int y = dtid.y;
    const int b_and_ch = dtid.z;
    const int b = b_and_ch / n_ch;
    const int ch = b_and_ch % n_ch;

    if (x >= w || y >= h || ch >= n_ch || b >= n) return;

    // otherwise sample!
    // force coordinate logic to float for precision!
    float fy = (float)flow[(b * 2 * h * w) + (h * w) + (y * w) + x];
    float fx = (float)flow[(b * 2 * h * w) + (y * w) + x ];
    float sample_y = (float)y + fy;
    float sample_x = (float)x + fx;

    if (!align_corners) {
        sample_y = ((sample_y * h) / (h-1)) - 0.5;
        sample_x = ((sample_x * w) / (w-1)) - 0.5;
    }

    const int start_idx = (b * n_ch * h * w) + (ch * h * w);

    scalar_t val;
    if (padding_mode == 0) {
        val = bilinear_interpolate_zeros(input, start_idx, (int)h, (int)w, sample_y, sample_x);
    } else {
        val = bilinear_interpolate_border(input, start_idx, (int)h, (int)w, sample_y, sample_x);
    }

    output[start_idx + (y * w) + x] = val;

    return;
}
