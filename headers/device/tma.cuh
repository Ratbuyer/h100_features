// TMA api wrappers

#include <stdint.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

enum Cache_Policy
{
  evict_normal,
  evict_first,
  evict_last,
  evict_unchanged,
  no_allocate,
};

/*
// global -> shared::cluster:
cp.async.bulk.prefetch.tensor.dim.L2.src{.load_mode}{.level::cache_hint} [tensorMap, tensorCoords]
                                                             {, im2colOffsets } {, cache-policy}

.src =                { .global }
.dim =                { .1d, .2d, .3d, .4d, .5d }
.load_mode =          { .tile, .im2col }
.level::cache_hint =  { .L2::cache_hint }
*/

// 1d prefetch
// src align to 16, size multiple of 16
__device__ void copy_async_1d_prefetch(const CUtensorMap *__tensor_map, int coordinate)
{
  asm volatile(
      "cp.async.bulk.prefetch.tensor.dim.L2.src.global.tile"
      " [%0, {%1}];"
      :
      : "l"(__tensor_map),
        "r"(coordinate)
      : "memory");
}

// 2d prefetch
// src align to 16, size multiple of 16
__device__ void copy_async_2d_prefetch(const CUtensorMap *__tensor_map, int coordinate1, int coordinate2)
{
  asm volatile(
      "cp.async.bulk.prefetch.tensor.2d.L2.global.tile"
      " [%0, {%1, %2}];"
      :
      : "l"(__tensor_map),
        "r"(coordinate1),
        "r"(coordinate2)
      : "memory");
}