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