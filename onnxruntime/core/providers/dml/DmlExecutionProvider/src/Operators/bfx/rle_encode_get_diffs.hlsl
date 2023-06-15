#if !defined(T_diffs)
#define T_diffs int32_t
#endif

#if !defined(T_idxs)
#define T_idxs int32_t
#endif

RWStructuredBuffer<T_diffs> els          : register(u0); // n,
RWStructuredBuffer<T_idxs>  nonzero_mask : register(u1); // n,
RWStructuredBuffer<T_diffs> diffs        : register(u2); // n,

#include "rle_encode_get_diffs_shader_constants.h"

[numthreads(256, 1, 1)]
void rle_encode_get_diffs(uint3 dtid : SV_DispatchThreadId)
{
    const int i = dtid.x;
    if (i >= n) return;

    // first difference element is itself, later elements are differences between element and previous element
    const T_diffs el_i = els[i];
    T_diffs el_diff = el_i;
    if (i > 0) { el_diff = el_i - els[i-1]; }

    diffs[i] = el_diff;
    nonzero_mask[i] = el_diff == 0 ? 0 : 1;
}
