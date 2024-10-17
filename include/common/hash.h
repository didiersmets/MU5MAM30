#pragma once
/******************************************************************************
 * Hash  : A couple favourite hash functions.  They hash 4 (resp. 8) bytes at
 *         a time (the output of the last call should be given as a first 
 *         argument of the next call when applying them to data with multiple
 *         4 (resp. 8) bytes words. On first call 0 can be safely used as the 
 *         first hash argument.
 *****************************************************************************/

#include <stdio.h>
#include <stdint.h>

/* Murmur2 32*/
inline uint32_t murmur2_32(uint32_t hash, uint32_t key)
{
	const uint32_t m = 0x5bd1e995;
	const int r = 24;

	key *= m;
	key ^= key >> r;
	key *= m;

	hash *= m;
	hash ^= key;

	return hash;
}

/* Murmur2 64*/
inline uint64_t murmur2_64(uint64_t hash, uint64_t key)
{
	const uint64_t m = 0xc6a4a7935bd1e995llu;
	const int r = 47;

	key *= m;
	key ^= key >> r;
	key *= m;

	hash *= m;
	hash ^= key;

	return hash;
};
