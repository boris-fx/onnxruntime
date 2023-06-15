#if !defined(scalar_t)
#define scalar_t float
#endif

RWStructuredBuffer<scalar_t> input  : register(u0); // n, n_in_channels, in_h, in_w
RWStructuredBuffer<scalar_t> offset : register(u1); // n, (n_offset_grps * kernel_h * kernel_w * 2), out_h, out_w
RWStructuredBuffer<scalar_t> mask   : register(u2); // n, (n_offset_grps * kernel_h * kernel_w), out_h, out_w
RWStructuredBuffer<scalar_t> output : register(u3); // (n_in_channels * kernel_h * kernel_w), (out_h, out_w)

cbuffer Constants
{
    int n_kernels; // total number of elements in output!
    int n; // batch size
    int n_in_channels;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int dil_h;
    int dil_w;
    int stride_h;
    int stride_w;
    int n_offset_grps;
    int use_mask;
};

// this copied verbatim from:
// https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp#L80
// template <typename T>
scalar_t bilinear_interpolate(
    // const T* in,
    int input_idx,
    int height,
    int width,
    scalar_t h,
    scalar_t w) {

  if (h <= (scalar_t)-1 || (scalar_t)height <= h || w <= (scalar_t)-1 || (scalar_t)width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - (scalar_t)h_low;
  scalar_t lw = w - (scalar_t)w_low;
  scalar_t hh = (scalar_t)1 - lh;
  scalar_t hw = (scalar_t)1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = input[input_idx + (h_low * width + w_low)];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[input_idx + (h_low * width + w_high)];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[input_idx + (h_high * width + w_low)];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[input_idx + (h_high * width + w_high)];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


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
        // *columns_ptr =
        output[output_idx] =
            mask_value * bilinear_interpolate((int)input_idx, (int)in_h, (int)in_w, y, x);
        output_idx += n * out_h * out_w;
      }
    }



    // if (idx == 0) {
        // printf("hello from HLSL?\n");
    // }

    // output[idx] = (scalar_t)(69.f);

}
