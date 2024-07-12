#include <cuda.h> // CUtensormap
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

const int GMEM_WIDTH = 1024;
const int GMEM_HEIGHT = 1024;

const int SMEM_WIDTH = 128;
const int SMEM_HEIGHT = 128;

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
{
  // Get pointer to cuGetProcAddress
  cudaDriverEntryPointQueryResult driver_status;
  void *cuGetProcAddress_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPoint("cuGetProcAddress", &cuGetProcAddress_ptr, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);
  PFN_cuGetProcAddress_v12000 cuGetProcAddress = reinterpret_cast<PFN_cuGetProcAddress_v12000>(cuGetProcAddress_ptr);

  // Use cuGetProcAddress to get a pointer to the CTK 12.0 version of cuTensorMapEncodeTiled
  CUdriverProcAddressQueryResult symbol_status;
  void *cuTensorMapEncodeTiled_ptr = nullptr;
  CUresult res = cuGetProcAddress("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
  assert(res == CUDA_SUCCESS && symbol_status == CU_GET_PROC_ADDRESS_SUCCESS);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y)
{
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0)
  {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0)
  {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  }
  else
  {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0)
  {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0)
  {
    (&bar)->~barrier();
  }
}

int main()
{

  void *tensor_ptr = nullptr;

  tensor_ptr = malloc(GMEM_WIDTH * GMEM_HEIGHT * sizeof(int));
  
  CUtensorMap tensor_map{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map, // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
      rank,        // cuuint32_t tensorRank,
      tensor_ptr,  // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      box_size,    // const cuuint32_t *boxDim,
      elem_stride, // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      // L2 Promotion can be used to widen the effect of a cache-policy to a wider
      // set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}