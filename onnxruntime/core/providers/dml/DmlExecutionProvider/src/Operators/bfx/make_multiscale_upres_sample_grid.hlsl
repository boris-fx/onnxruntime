#if !defined(scalar_t)
#define scalar_t float
#endif

RWStructuredBuffer<float> exec_config  : register(u0); // 5
RWStructuredBuffer<float> grid_out     : register(u1); // n, 2, tile_height, tile_width

#include "make_multiscale_upres_sample_grid_shader_constants.h"

#include "shader_util.h"

[numthreads(16, 16, 1)]
void make_multiscale_upres_sample_grid(uint3 dtid : SV_DispatchThreadId)
{
    const int x_tile = dtid.x;
    const int y_tile = dtid.y;

    if (x_tile >= tile_width || y_tile >= tile_height) return;

    const float x_start_hq = exec_config[0];
    const float y_start_hq = exec_config[1];
    const float x_start_lq = exec_config[2];
    const float y_start_lq = exec_config[3];
    const float scale =      exec_config[4];

    const float x_norm = (((x_tile + x_start_hq + 0.5f) / scale) - x_start_lq) / tile_width;
    const float y_norm = (((y_tile + y_start_hq + 0.5f) / scale) - y_start_lq) / tile_height;

    float x_ndc = (x_norm * 2.f) - 1.f;
    float y_ndc = (y_norm * 2.f) - 1.f;

    for (int b = 0; b < n; b++) {
        grid_out[(b * 2 * tile_width * tile_height) +                              (y_tile * tile_width) + x_tile] = x_ndc;
        grid_out[(b * 2 * tile_width * tile_height) + (tile_width * tile_height) + (y_tile * tile_width) + x_tile] = y_ndc;
    }
}
