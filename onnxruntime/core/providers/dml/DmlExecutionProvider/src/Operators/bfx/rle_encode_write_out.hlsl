#if !defined(T_diffs)
#define T_diffs int32_t
#endif

#if !defined(T_idxs)
#define T_idxs int32_t
#endif

RWStructuredBuffer<T_idxs>  nonzero_mask_pfx_sum : register(u0); // n+1,
RWStructuredBuffer<T_idxs>  idxes_out            : register(u1); // n,
RWStructuredBuffer<T_diffs> diffs_in             : register(u2); // n,
RWStructuredBuffer<T_diffs> diffs_out            : register(u3); // n,

#include "rle_encode_write_out_shader_constants.h"

[numthreads(256, 1, 1)]
void rle_encode_write_out(uint3 dtid : SV_DispatchThreadId)
{
    const int i = dtid.x;
    if (i >= n) return;
    const T_idxs a = nonzero_mask_pfx_sum[i + 1];
    const T_idxs b = (i == 0) ? 0 : nonzero_mask_pfx_sum[i];

    if (a - b == 1) {
        idxes_out[a-1] = (T_idxs)i;
        diffs_out[a-1] = diffs_in[i];
    }
}
