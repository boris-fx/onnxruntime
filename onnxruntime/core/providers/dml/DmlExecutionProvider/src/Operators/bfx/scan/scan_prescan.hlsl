#include "scan_defs.h"

#if !defined(T)
#define T int32_t
#endif

RWStructuredBuffer<T> d_out        : register(u0);
RWStructuredBuffer<T> d_in         : register(u1);
RWStructuredBuffer<T> d_block_sums : register(u2);

groupshared T s_out[SHMEM_SZ];

#include "scan_prescan_hlsl.h"

[numthreads(BLOCK_SZ, 1, 1)]
void scan_prescan(uint3 dtid : SV_DispatchThreadId, uint3 gid : SV_GroupID)
{
    const int blockDim_x = BLOCK_SZ; // matches above
    const int blockIdx_x = gid.x;
    const int threadIdx_x = dtid.x - (blockDim_x * blockIdx_x);

    const int thid = threadIdx_x;
	int ai = thid;
	int bi = thid + blockDim_x;

    // Zero out the shared memory
	// Helpful especially when input size is not power of two
	s_out[thid] = 0;
	s_out[thid + blockDim_x] = 0;
	if (thid + MAX_BLOCK_SZ < SHMEM_SZ)
		s_out[thid + MAX_BLOCK_SZ] = 0;

    GroupMemoryBarrierWithGroupSync();

    // Copy d_in to shared memory
	// Note that d_in's elements are scattered into shared memory
	//  in light of avoiding bank conflicts
	unsigned int cpy_idx = MAX_BLOCK_SZ * blockIdx_x + threadIdx_x;
	if (cpy_idx < n)
	{
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
		if (cpy_idx + blockDim_x < n)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim_x];
	}


	// For both upsweep and downsweep:
	// Sequential indices with conflict free padding
	//  Amount of padding = target index / num banks
	//  This "shifts" the target indices by one every multiple
	//   of the num banks
	// offset controls the stride and starting index of
	//  target elems at every iteration
	// d just controls which threads are active
	// Sweeps are pivoted on the last element of shared memory

	// Upsweep/Reduce step
	int offset = 1;
	for (int d = MAX_BLOCK_SZ >> 1; d > 0; d >>= 1)
	{
		GroupMemoryBarrierWithGroupSync();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_out[bi] += s_out[ai];
		}
		offset <<= 1;
	}

	// Save the total sum on the global block sums array
	// Then clear the last element on the shared memory
	if (thid == 0)
	{
		d_block_sums[blockIdx_x] = s_out[MAX_BLOCK_SZ - 1
			+ CONFLICT_FREE_OFFSET(MAX_BLOCK_SZ - 1)];
		s_out[MAX_BLOCK_SZ - 1
			+ CONFLICT_FREE_OFFSET(MAX_BLOCK_SZ - 1)] = 0;
	}

	// Downsweep step
	for (d = 1; d < MAX_BLOCK_SZ; d <<= 1)
	{
		offset >>= 1;
		GroupMemoryBarrierWithGroupSync();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T temp = s_out[ai];
			s_out[ai] = s_out[bi];
			s_out[bi] += temp;
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// Copy contents of shared memory to global memory
	if (cpy_idx < n)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
		if (cpy_idx + blockDim_x < n)
			d_out[cpy_idx + blockDim_x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
	}
}
