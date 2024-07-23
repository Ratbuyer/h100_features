#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <assert.h>

PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled()
{
  void *driver_ptr = nullptr;
  cudaDriverEntryPointQueryResult driver_status;
  auto code = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr, cudaEnableDefault, &driver_status);
  assert(code == cudaSuccess && "Could not get driver API");
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
}

// create a 2d tensor map
// for a matrix, row number is tensor_dim1, column number is tensor_dim2
// assuming row major
CUtensorMap create_2d_tensor_map(uint64_t tensor_dim1, uint64_t tensor_dim2, uint32_t tile_dim1, uint32_t tile_dim2, void *tensor_ptr)
{
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
  CUtensorMap local_tensor_map{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {tensor_dim1, tensor_dim2};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {tensor_dim2 * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {tile_dim1, tile_dim2};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
      &local_tensor_map, // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
      rank,        // cuuint32_t tensorRank,
      tensor_ptr,  // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      box_size,    // const cuuint32_t *boxDim,
      elem_stride, // const cuuint32_t *elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res == CUDA_SUCCESS && "tensormap creation failed.");

  return local_tensor_map;
}