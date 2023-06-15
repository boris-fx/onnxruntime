#if !defined(T)
#define T int32_t
#endif

RWStructuredBuffer<T> els : register(u0); // n,

#include "scan_clear_buffer_hlsl.h"

[numthreads(256, 1, 1)]
void scan_clear_buffer(uint3 dtid : SV_DispatchThreadId)
{
    const int i = dtid.x;
    if (i >= n) return;
    // why can't this just be a 'memset' directly to the command list??
    els[i] = 0;
}
