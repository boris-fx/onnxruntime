#include "scan_defs.h"

#if !defined(T)
#define T int32_t
#endif


RWStructuredBuffer<T> d            : register(u0);
RWStructuredBuffer<T> d_block_sums : register(u1);

#include "scan_add_block_sums_hlsl.h"

[numthreads(MAX_BLOCK_SZ / 2, 1, 1)]
void scan_add_block_sums(uint3 dtid : SV_DispatchThreadId, uint3 gid : SV_GroupID)
{
    const int blockDim_x = MAX_BLOCK_SZ / 2; // matches above
    const int blockIdx_x = gid.x;
    const int threadIdx_x = dtid.x - (blockDim_x * blockIdx_x);

	const T d_block_sum_val = d_block_sums[blockIdx_x];

	// Simple implementation's performance is not significantly (if at all)
	//  better than previous verbose implementation
	const unsigned int cpy_idx = 2 * blockIdx_x * blockDim_x + threadIdx_x;
	if (cpy_idx < n)
	{
		d[cpy_idx] = d[cpy_idx] + d_block_sum_val;
		if (cpy_idx + blockDim_x < n)
			d[cpy_idx + blockDim_x] = d[cpy_idx + blockDim_x] + d_block_sum_val;
	}
}
