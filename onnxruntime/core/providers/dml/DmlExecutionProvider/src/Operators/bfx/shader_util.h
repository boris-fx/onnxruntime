#pragma once

// utility functions to be called from hlsl shaders..

#define EULER_NUMBER 2.71828182846

inline float sigmoidf(float n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

// this copied verbatim from:
// https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp#L80
// template <typename T>
inline scalar_t bilinear_interpolate_zeros(
    RWStructuredBuffer<scalar_t> img,
    int input_idx,
    int height,
    int width,
    float h,
    float w) {

  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - (float)h_low;
  float lw = w - (float)w_low;
  float hh = (float)1 - lh;
  float hw = (float)1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = img[input_idx + (h_low * width + w_low)];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = img[input_idx + (h_low * width + w_high)];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = img[input_idx + (h_high * width + w_low)];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = img[input_idx + (h_high * width + w_high)];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return (scalar_t)val;
}



// template <typename T>
inline scalar_t bilinear_interpolate_border(
    RWStructuredBuffer<scalar_t> img,
    int input_idx,
    int height,
    int width,
    float h,
    float w) {

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1 - lh;
  const float hw = 1 - lw;

  if (h_low < 0) h_low = 0;
  if (h_low >= height) h_low = height - 1;
  if (w_low < 0) w_low = 0;
  if (w_low >= width) w_low = width - 1;
  if (h_high < 0) h_high = 0;
  if (h_high >= height) h_high = height - 1;
  if (w_high < 0) w_high = 0;
  if (w_high >= width) w_high = width - 1;

  float v1 = (float)img[input_idx + h_low * width + w_low];
  float v2 = (float)img[input_idx + h_low * width + w_high];
  float v3 = (float)img[input_idx + h_high * width + w_low];
  float v4 = (float)img[input_idx + h_high * width + w_high];

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return (scalar_t)val;
}
