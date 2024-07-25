// TMA api wrappers

#pragma once

// Suppress warning about barrier in shared memory
#pragma nv_diag_suppress static_var_with_dynamic_init

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

// inline _LIBCUDACXX_DEVICE void cp_async_bulk_tensor_1d_shared_to_global_multicast(
//     void *__dest,
//     const CUtensorMap *__tensor_map,
//     int __c0,
//     ::cuda::barrier<::cuda::thread_scope_block> &__bar,
//     uint16_t __ctaMask)
// {
//   asm volatile(
//       "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
//       "[%0], [%1, {%2}], [%3], %4;\n"
//       :
//       : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__dest))),
//         "l"(&__tensor_map),
//         "r"(__c0),
//         "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(::cuda::device::barrier_native_handle(__bar)))),
//         "h"(__ctaMask)
//       : "memory");
// }