#pragma once

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

#define BLOCK_SZ (MAX_BLOCK_SZ / 2)
#define SHMEM_SZ (MAX_BLOCK_SZ + ((MAX_BLOCK_SZ - 1) >> LOG_NUM_BANKS))
