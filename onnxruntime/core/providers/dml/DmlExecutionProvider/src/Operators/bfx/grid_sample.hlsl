#if !defined(scalar_t)
#define scalar_t float
#endif

RWStructuredBuffer<scalar_t> input  : register(u0); // n, n_ch, in_h, in_w
RWStructuredBuffer<float>    grid   : register(u1); // n, 2, out_h, out_w
RWStructuredBuffer<scalar_t> output : register(u2); // n, n_ch, out_h, out_w

#include "grid_sample_shader_constants.h"

#include "shader_util.h"

[numthreads(16, 16, 1)]
void grid_sample(uint3 dtid : SV_DispatchThreadId)
{
    const int out_x = dtid.x;
    const int out_y = dtid.y;
    const int b_and_ch = dtid.z;
    const int b = b_and_ch / n_ch;
    const int ch = b_and_ch % n_ch;

    if (out_x >= out_w || out_y >= out_h || ch >= n_ch || b >= n) return;

    // otherwise sample!
    // force coordinate logic to float for precision!
    float sample_x_ndc = (float)grid[(b * 2 * out_h * out_w) +                   (out_y * out_w) + out_x ];
    float sample_y_ndc = (float)grid[(b * 2 * out_h * out_w) + (out_h * out_w) + (out_y * out_w) + out_x];

    float sample_x = (sample_x_ndc + 1) / 2;
    float sample_y = (sample_y_ndc + 1) / 2;
    if (!align_corners) {
        sample_x = (sample_x * in_w) - 0.5f;
        sample_y = (sample_y * in_h) - 0.5f;
    } else {
        sample_x = (sample_x * (in_w-1));
        sample_y = (sample_y * (in_h-1));
    }

    const int in_start_idx =  (b * n_ch * in_h  * in_w) +  (ch * in_h  * in_w);
    const int out_start_idx = (b * n_ch * out_h * out_w) + (ch * out_h * out_w);

    scalar_t val;
    if (padding_mode == 0) {
        val = bilinear_interpolate_zeros(input, in_start_idx, (int)in_h, (int)in_w, sample_y, sample_x);
    } else {
        val = bilinear_interpolate_border(input, in_start_idx, (int)in_h, (int)in_w, sample_y, sample_x);
    }

    output[out_start_idx + (out_y * out_w) + out_x] = val;

    return;
}
