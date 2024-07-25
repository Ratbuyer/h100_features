/*
This code demonstrates how to use the sparse wgmma instructions
to perform matrix multiplication

Sparse means matrix A follows a 2:4 format
*/

#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>

#include "matrix_utilities.cuh"
#include "wgmma.cuh"
#include "profile_utilities.cuh"

const int M = 64;
const int N = 8;
const int K = 32;

// 2:4 format
const int K2 = 16;

__global__ void kernel(half *A, half *B, half *C, u_int32_t *metadata_array)
{
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int group_id = lane_id >> 2;
  const int lane_in_group = lane_id & 3;
  const int lane_in_work_group = lane_in_group % 2;

  __align__(16) __shared__ half A_shared[M * K2];
  __align__(16) __shared__ half B_shared[K * N];

  // use one thread to load so it's easier to tell the layout
  // refer to the ptx menu for the layout of the shared memory
  if (tid == 0)
  {
    for (int i = 0; i < M; i++)
    {
      for (int j = 0; j < K2; j++)
      {
        int block_x = i / 8;
        int block_row = i % 8;
        int block_y = j / 8;
        int block_col = j % 8;
        int block_id = block_x * 2 + block_y;
        int offset = block_id * 64 + block_row * 8 + block_col;
        A_shared[offset] = A[i * K2 + j];
      }
    }

    for (int i = 0; i < K; i++)
    {
      for (int j = 0; j < N; j++)
      {
        int block_x = i / 8;
        int block_row = i % 8;
        int block_y = j / 8;
        int block_col = j % 8;
        int block_id = block_x * 1 + block_y;
        int offset = block_id * 64 + block_row * 8 + block_col;
        B_shared[offset] = B[i * N + j];
      }
    }
  }

  __syncthreads();

  // load metadata
  u_int32_t metadata;
  uint metadata_offset = warp_id * 16 + lane_in_work_group * 8 + group_id;
  metadata = metadata_array[metadata_offset];

  __syncthreads();

  // create descriptors
  GmmaDescriptor desc_a = make_desc_a(A_shared);
  GmmaDescriptor desc_b = make_desc_b(B_shared);

  // accumulator
  uint32_t c[2] = {};

  warpgroup_arrive();

  asm volatile("wgmma.mma_async.sp.sync.aligned.m64n8k32.f16.f16.f16 "
               "{%0, %1}, " // c
               "%2, %3, "   // desc A, B
               "%4, "       // meta
               "0, "        // thread selection
               "1, "        // scale D
               "%7, %8, "   // +/- scale A, B
               "%9, %10;"   // transpose A, B
               : "+r"(c[0]), "+r"(c[1])
               : "l"(desc_a), "l"(desc_b),
                 "r"(metadata),   // metadata
                 "r"(0),          // thread selection
                 "r"(1),          // scale D
                 "n"(1), "n"(1),  // +- scale A, B
                 "n"(0), "n"(1)); // transpose A, B

  // commit, start the computation
  warpgroup_commit_batch();

  // wait for the previous commit to finish
  warpgroup_wait<0>();

  // thread fence needed for async operations
  __threadfence();

  warpgroup_arrive();

  // store the result
  uint32_t *C_ptr = reinterpret_cast<uint32_t *>(C);

  int offset1 = warp_id * 16 * 4 + group_id * 4 + lane_in_group;
  int offset2 = warp_id * 16 * 4 + (group_id + 8) * 4 + lane_in_group;

  C_ptr[offset1] = c[0];
  C_ptr[offset2] = c[1];
}

int main()
{

  half *d_C;
  half h_C[M * N];
  half h_CPU[M * N];
  half h_A[M * K];
  half h_A2[M * K2];
  half h_B[K * N];

  fill_24(h_A, M, K);
  fill_random(h_B, K, N);

  // print_matrix(h_A, M, K);

  // extract the non-zeros in each 2:4 tile to a compressed matrix A2
  compress24(h_A, h_A2, M, K);

  // print_matrix(h_A2, M, K2);

  half *d_A, *d_B;

  cudaMalloc((void **)&d_A, M * K2 * sizeof(half));
  cudaMalloc((void **)&d_B, K * N * sizeof(half));
  cudaMalloc((void **)&d_C, M * N * sizeof(half));

  cudaMemcpy(d_A, h_A2, M * K2 * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

  int metadata_size = (M / 16) * (K / 16) * 8;

  u_int32_t *metadata_array = new u_int32_t[metadata_size];
  inspect_metadata(h_A, metadata_array, M, K);

  u_int32_t *d_metadata;
  cudaMalloc((void **)&d_metadata, metadata_size * sizeof(u_int32_t));
  cudaMemcpy(d_metadata, metadata_array, metadata_size * sizeof(u_int32_t), cudaMemcpyHostToDevice);

  kernel<<<1, 128>>>(d_A, d_B, d_C, d_metadata);

  cuda_check_error();

  cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

  // print_matrix(h_C, M, N);

  CPU_gemm(h_A, h_B, h_CPU, M, N, K);

  compare_matrices(h_CPU, h_C, M, N);

  // print_differnce(h_CPU, h_C, M, N, 0);

  return 0;
}