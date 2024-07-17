#include <cooperative_groups.h>
#include <stdio.h>

__global__ void __cluster_dims__(2, 1, 1) cluster_kernel()
{
  // printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);

  __shared__ int smem[32];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  smem[tid] = blockIdx.x + threadIdx.x;

  cluster.sync();

  int *other_block_smem = cluster.map_shared_rank(smem, 1 - clusterBlockRank);

  int value = other_block_smem[tid];

  // print the value
  printf("blockIdx.x: %d, threadIdx.x: %d, value: %d\n", blockIdx.x, threadIdx.x, value);
}

int main()
{

  // two blocks in a cluster
  cluster_kernel<<<2, 32>>>();

  cudaDeviceSynchronize();

  // check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}
