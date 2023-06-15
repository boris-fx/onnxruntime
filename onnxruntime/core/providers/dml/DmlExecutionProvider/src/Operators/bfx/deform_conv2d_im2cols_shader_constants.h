#pragma once

#ifdef __cplusplus
struct shader_constants
#else
cbuffer Constants
#endif

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

#ifdef __cplusplus
const static inline int32_t num_els_constants = 17; // TODO: compute based on sizeof..
#endif
