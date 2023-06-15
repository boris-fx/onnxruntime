#pragma once

#ifdef __cplusplus
struct shader_constants
#else
cbuffer Constants
#endif

{
    float max_residue_magnitude;

    int n;
    int n_ch_feats;
    int h;
    int w;
};

#ifdef __cplusplus
const static inline int32_t num_els_constants = 5; // TODO: compute based on sizeof..
#endif
