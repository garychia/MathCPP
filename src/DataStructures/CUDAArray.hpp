#include "CUDAHelpers.hpp"
#include "CUDAUtilities.cuh"
#include "Container.hpp"

#include "cuda_runtime.h"

namespace DataStructures {
template <class T> class CUDAArray {
private:
  // Data stored on the GPU side.
  T *gpuData = nullptr;
  // Number of elements stored.
  size_t size = 0;

public:
  /**
   * @brief Construct a new CUDAArray object
   */
  CUDAArray() : gpuData(nullptr), size(0) {}

  /**
   * Construct a new CUDAArray object of an initial size and populated
   * with a value.
   *
   * @param size the initial size of the CUDAArray to be generated.
   * @param value the value the CUDAArray will be filled with.
   **/
  CUDAArray(size_t size, const T &value) {
    this->size = size;
    CheckCUDAStatus(cudaMalloc(&gpuData, size * sizeof(T)));
    CudaHelpers::GPUArrayPopulate(gpuData, size, value);
  }

  /**
   * @brief Construct a new CUDAArray object populated with the elements of a
   * given array.
   *
   * @param size the size of the CUDAArray.
   * @param arr the
   */
  CUDAArray(size_t size, const T *arr) {
    this->size = size;
    CheckCUDAStatus(cudaMalloc(&gpuData, size * sizeof(T)));
    CheckCUDAStatus(
        cudaMemcpy(gpuData, arr, sizeof(T) * size, cudaMemcpyHostToDevice));
  }

  /**
   * @brief Construct a new CUDAArray object copied from another CUDAArray.
   *
   * @param other a CUDAArray to be copied.
   */
  CUDAArray(const CUDAArray<T> &other) {
    size = other.size;
    CheckCUDAStatus(cudaMalloc(&gpuData, size * sizeof(T)));
    CheckCUDAStatus(cudaMemcpy(gpuData, other.gpuData, sizeof(T) * size,
                               cudaMemcpyDeviceToDevice));
  }

  /**
   * @brief Construct a new CUDAArray object using move semantics.
   *
   * @param other a r-value reference to a CUDAArray.
   */
  CUDAArray(CUDAArray<T> &&other) noexcept
      : size(other.size), gpuData(other.gpuData) {
    other.size = 0;
    other.gpuData = nullptr;
  }

  /**
   * @brief Destroy the CUDAArray object
   *
   */
  virtual ~CUDAArray() {
    if (gpuData)
      CheckCUDAStatus(cudaFree(gpuData));
    size = 0;
  }

  /**
   * @brief Copy a CUDArray by assignment.
   *
   * @param other a CUDAArray to be copied.
   * @return CUDAArray<T>& a reference to this CUDAArray.
   */
  virtual CUDAArray<T> &operator=(const CUDAArray<T> &other) {
    if (gpuData)
      CheckCUDAStatus(cudaFree(gpuData));
    size = other.size;
    CheckCUDAStatus(cudaMalloc(&gpuData, sizeof(T) * size));
    CheckCUDAStatus(cudaMemcpy(gpuData, other.gpuData, sizeof(T) * size,
                               cudaMemcpyDeviceToDevice));
    return *this;
  }

  /**
   * @brief Move a CUDAArray.
   *
   * @param other a CUDAArray to be moved.
   * @return CUDAArray<T>& the reference to this CUDArray.
   */
  virtual CUDAArray<T> &operator=(CUDAArray<T> &&other) {
    if (gpuData)
      CheckCUDAStatus(cudaFree(gpuData));
    gpuData = nullptr;
    size = other.size;
    gpuData = other.gpuData;
    other.size = 0;
    other.gpuData = nullptr;
    return *this;
  }

  /**
   * @brief Return the size of a CUDAArray.
   *
   * @return size_t the number of elements this CUDAArray stores.
   */
  size_t Size() const { return size; }

  /**
   * @brief Check if this CUDAArray is empty or not.
   *
   * @return true if it is empty.
   * @return false if it has any element.
   */
  virtual bool IsEmpty() const { return 0 == size; }

  /**
   * @brief Copy data from the GPU to the CPU.
   *
   * @param dest the destination where data will be stored on the CPU side.
   */
  void CopyToCPU(T *dest) const {
    CheckCUDAStatus(
        cudaMemcpy(dest, gpuData, sizeof(T) * size, cudaMemcpyDeviceToHost));
  }

  /**
   * @brief Retrieve the GPU pointer that points to data on the GPU side.
   *
   * @return T* a pointer to the data on the GPU side.
   */
  T *GetGPUPtr() const { return gpuData; }
};
} // namespace DataStructures
