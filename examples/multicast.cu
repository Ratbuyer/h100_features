/*
This code uses TMA's 1d tensor load to load
a portion of an array to shared memory and then
change the value in the shared memory and uses TMA's store
to store the portion back to global memory. We print the result
to show the changes are done.
*/

// supress warning about barrier in shared memory on line 32
#pragma nv_diag_suppress static_var_with_dynamic_init

#include <cuda/barrier>
#include <iostream>
#include <cooperative_groups.h>

#include "tma_tensor_map.cuh"
#include "matrix_utilities.cuh"
#include "profile_utilities.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

namespace cg = cooperative_groups;

const int array_size = 128;
const int tile_size = 16;

__global__ void __cluster_dims__(4, 1, 1) kernel(const __grid_constant__ CUtensorMap tensor_map, int coordinate)
{
  // cluster metadata
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();

  __shared__ alignas(16) int tile_shared[tile_size];

  // we let the first block in the cluster to load a
  // tile to the shared memory of both blocks
  if (clusterBlockRank == 0)
  {
    __shared__ barrier bar;

    if (threadIdx.x == 0)
    {
      init(&bar, blockDim.x);
      cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier::arrival_token token;
    if (threadIdx.x == 0)
    {
      cde::cp_async_bulk_tensor_1d_global_to_shared(tile_shared, &tensor_map, coordinate, bar);

      uint16_t ctaMask = 3;
      asm volatile(
          "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
          "[%0], [%1, {%2}], [%3], %4;\n"
          :
          : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(tile_shared))),
            "l"(&tensor_map),
            "r"(coordinate),
            "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(::cuda::device::barrier_native_handle(bar)))),
            "h"(ctaMask)
          : "memory");

      token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile_shared));
    }
    else
    {
      token = bar.arrive();
    }

    bar.wait(std::move(token));
  }

  // cluster 1 needs to wait for cluster 0 to load the data
  __threadfence();
  __syncthreads();
  cluster.sync();

  // verify block 1 recieved the data
  if (clusterBlockRank == 0 && threadIdx.x == 0)
  {
    printf("clusterBlockRank: %d, threadIdx.x: %d\n", clusterBlockRank, threadIdx.x);
    for (int i = 0; i < tile_size; ++i)
    {
      printf("%d|", tile_shared[i]);
    }
    printf("\n");
  }

  __threadfence();
  __syncthreads();
  cluster.sync();

  if (clusterBlockRank == 1 && threadIdx.x == 0)
  {
    printf("clusterBlockRank: %d, threadIdx.x: %d\n", clusterBlockRank, threadIdx.x);
    for (int i = 0; i < tile_size; ++i)
    {
      printf("%d|", tile_shared[i]);
    }
    printf("\n");
  }

  __threadfence();
  __syncthreads();
  cluster.sync();

  if (clusterBlockRank == 2 && threadIdx.x == 0)
  {
    printf("clusterBlockRank: %d, threadIdx.x: %d\n", clusterBlockRank, threadIdx.x);
    for (int i = 0; i < tile_size; ++i)
    {
      printf("%d|", tile_shared[i]);
    }
    printf("\n");
  }

  __threadfence();
  __syncthreads();
  cluster.sync();

  if (clusterBlockRank == 3 && threadIdx.x == 0)
  {
    printf("clusterBlockRank: %d, threadIdx.x: %d\n", clusterBlockRank, threadIdx.x);
    for (int i = 0; i < tile_size; ++i)
    {
      printf("%d|", tile_shared[i]);
    }
    printf("\n");
  }

  __threadfence();
  __syncthreads();
  cluster.sync();
}

int main()
{
  // initialize array and fill it with values
  int h_data[array_size];
  for (size_t i = 0; i < array_size; ++i)
  {
    h_data[i] = i;
  }

  // print the array before the kernel
  // one tile per line
  print_matrix(h_data, array_size / tile_size, tile_size);

  // transfer array to device
  int *d_data = nullptr;
  cudaMalloc(&d_data, array_size * sizeof(int));
  cudaMemcpy(d_data, h_data, array_size * sizeof(int), cudaMemcpyHostToDevice);

  // create tensor map
  CUtensorMap tensor_map = create_1d_tensor_map(array_size, tile_size, d_data);

  size_t offset = tile_size * 3; // select the second tile of the array to change
  kernel<<<4, 128>>>(tensor_map, offset);

  cuda_check_error();

  cudaFree(d_data);

  return 0;
}