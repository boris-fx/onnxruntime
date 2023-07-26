#pragma once

#ifdef __cplusplus
struct shader_constants
#else
cbuffer Constants
#endif

{
    int interpolation_mode;
    int padding_mode;
    int align_corners;

    int n;
    int n_ch;
    int in_h;
    int in_w;
    int out_h;
    int out_w;
};

#ifdef __cplusplus
const static inline int32_t num_els_constants = 9; // TODO: compute based on sizeof..
#endif
