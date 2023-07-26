#pragma once

#ifdef __cplusplus
struct shader_constants
#else
cbuffer Constants
#endif

{
    int n;
    int tile_width;
    int tile_height;
};

#ifdef __cplusplus
const static inline int32_t num_els_constants = 3; // TODO: compute based on sizeof..
#endif
