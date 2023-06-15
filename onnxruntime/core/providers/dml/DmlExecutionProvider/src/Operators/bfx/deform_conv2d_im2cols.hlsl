#if !defined(scalar_t)
#define scalar_t float
#endif

RWStructuredBuffer<scalar_t> input  : register(u0); // n, n_in_channels, in_h, in_w
RWStructuredBuffer<scalar_t> offset : register(u1); // n, (n_offset_grps * kernel_h * kernel_w * 2), out_h, out_w
RWStructuredBuffer<scalar_t> mask   : register(u2); // n, (n_offset_grps * kernel_h * kernel_w), out_h, out_w
RWStructuredBuffer<scalar_t> output : register(u3); // (n_in_channels * kernel_h * kernel_w), (out_h, out_w)

#include "deform_conv2d_im2cols_shader_constants.h"

#include "shader_util.h"

[numthreads(256, 1, 1)]
void deform_conv2d_im2cols(uint3 dtid : SV_DispatchThreadId)
{
    const int idx = dtid.x;
    if (idx >= n_kernels) return;

    const int out_x = idx % out_w;
    const int out_y = (idx / out_w) % out_h;
    const int out_b = (idx / (out_w * out_h)) % n;
    const int in_c = idx / (out_w * out_h * n);
    const int out_c = in_c * kernel_h * kernel_w;

    const int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    int output_idx = (out_c * (n * out_h * out_w) + out_b * (out_h * out_w) + out_y * out_w + out_x);
    int input_idx = (out_b * (n_in_channels * in_h * in_w) + in_c * (in_h * in_w));
    int offset_idx = (out_b * n_offset_grps + grp_idx) * 2 * kernel_h * kernel_w * out_h * out_w;
    int mask_idx = (out_b * n_offset_grps + grp_idx) * kernel_h * kernel_w * out_h * out_w;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int mask_idx_ = i * kernel_w + j;
        const int offset_idx_ = 2 * mask_idx_;

        scalar_t mask_value = 1;
        if (use_mask) {
          mask_value = mask[mask_idx + (mask_idx_ * (out_h * out_w) + out_y * out_w + out_x)];
        }

        const scalar_t offset_h =
            offset[offset_idx + (offset_idx_ * (out_h * out_w) + out_y * out_w + out_x)];
        const scalar_t offset_w =
            offset[offset_idx + ((offset_idx_ + 1) * (out_h * out_w) + out_y * out_w + out_x)];
        const scalar_t y =
            scalar_t((out_y * stride_h - pad_h) + i * dil_h) + offset_h;
        const scalar_t x =
            scalar_t((out_x * stride_w - pad_w) + j * dil_w) + offset_w;
        output[output_idx] =
            mask_value * bilinear_interpolate_zeros(input, (int)input_idx, (int)in_h, (int)in_w, y, x);
        output_idx += n * out_h * out_w;
      }
    }
}
