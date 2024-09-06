// This code uses TMA's 1d load to load an array to
// shared memory and then change the value in the
// shared memory and uses TMA's store to store the
// array back to global memory.

#include <cuda/barrier>
#include <iostream>


using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

static constexpr size_t buf_len = 1024;

__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffers for x and y. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                // a)
    cde::fence_proxy_async_shared_cta();   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_global_to_shared(smem_data, data + offset, sizeof(smem_data), bar);
    // 3a. Arrive on the barrier and tell how many bytes are expected to come in (the transaction count)
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_data));
  } else {
    // 3b. Rest of threads just arrive
    token = bar.arrive();
  }

  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_shared_to_global(data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }
}

int main()
{
    int* d_data;
    int h_data[buf_len];

    for (size_t i = 0; i < buf_len; ++i) {
        h_data[i] = i;
    }

    std::cout << "Array before kernel execution:\n";
    for (size_t i = 0; i < buf_len; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    cudaMalloc(&d_data, buf_len * sizeof(int));
    cudaMemcpy(d_data, h_data, buf_len * sizeof(int), cudaMemcpyHostToDevice);

    size_t offset = 0;
    add_one_kernel<<<1, 256>>>(d_data, offset);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, buf_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    std::cout << "Array after kernel execution:\n";
    for (size_t i = 0; i < buf_len; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}