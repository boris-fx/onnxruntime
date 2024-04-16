#if !defined(T_vals)
#define T_vals int32_t
#endif

#if !defined(T_idxs)
#define T_idxs int32_t
#endif

RWStructuredBuffer<T_idxs>  n_vals : register(u0);
RWStructuredBuffer<T_vals>  idxs   : register(u1);
RWStructuredBuffer<T_idxs>  vals   : register(u2);
RWStructuredBuffer<T_vals>  output : register(u3);

#include "rle_decode_scatter_hlsl.h"

[numthreads(256, 1, 1)]
void rle_decode_scatter(uint3 dtid : SV_DispatchThreadId)
{
    const int i = dtid.x;
    // if bigger than output idx..
    // if (i >= n) return;
    // if bigger than num vals, which could be a constant, but that would require a device-to-host sync/copy
    if (i >= n_vals[0]) return;

    output[idxs[i]] = vals[i];
}
