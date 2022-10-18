#include "Container.hpp"
#include "CUDAUtilities.cuh"
#include "CUDAHelpers.hpp"

#include "cuda_runtime.h"

namespace DataStructures
{
    template <class T>
    class CUDAArray
    {
    private:
        T *gpuData = nullptr;
        std::size_t size = 0;

    public:
        /*
        Constructor that Generates an Empty CUDAArray.
        */
        CUDAArray() : gpuData(nullptr), size(0) {}

        /*
        Constructor with Initial Size and an Initial Value.
        @param s the initial size of the CUDAArray to be generated.
        @param value the value the CUDAArray will be filled with.
        */
        CUDAArray(std::size_t size, const T &value)
        {
            this->size = size;
            CheckCUDAStatus(cudaMalloc(&gpuData, size * sizeof(T)));
            CudaHelpers::GPUArrayPopulate(gpuData, size, value);
        }

        CUDAArray(std::size_t size, const T *arr)
        {
            this->size = size;
            CheckCUDAStatus(cudaMalloc(&gpuData, size * sizeof(T)));
            CheckCUDAStatus(cudaMemcpy(gpuData, arr, sizeof(T) * size, cudaMemcpyHostToDevice));
        }

        /*
        Copy Constructor
        @param other a CUDAArray to be copied.
        */
        CUDAArray(const CUDAArray<T> &other)
        {
            size = other.size;
            CheckCUDAStatus(cudaMalloc(&gpuData, size * sizeof(T)));
            CheckCUDAStatus(cudaMemcpy(gpuData, other.gpuData, sizeof(T) * size, cudaMemcpyDeviceToDevice));
        }

        /*
        Move Constructor
        @param other a CUDAArray to be moved.
        */
        CUDAArray(CUDAArray<T> &&other) noexcept : size(other.size), gpuData(other.gpuData)
        {
            other.size = 0;
            other.gpuData = nullptr;
        }

        /*
        Destructor
        */
        virtual ~CUDAArray()
        {
            if (gpuData)
            {
                CheckCUDAStatus(cudaFree(gpuData));
                gpuData = nullptr;
            }
            size = 0;
        }

        /*
        Copy Assignment
        @param other a CUDAArray to be copied.
        @return a reference to this CUDAArray.
        */
        virtual CUDAArray<T> &operator=(const CUDAArray<T> &other)
        {
            if (gpuData)
                CheckCUDAStatus(cudaFree(gpuData));
            size = other.size;
            CheckCUDAStatus(cudaMalloc(&gpuData, sizeof(T) * size));
            CheckCUDAStatus(cudaMemcpy(gpuData, other.gpuData, sizeof(T) * size, cudaMemcpyDeviceToDevice));
            return *this;
        }

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

        /*
        Returns the number of elements this CUDAArray stores.
        @return the number of elements this CUDAArray stores.
        */
        std::size_t Size() const
        {
            return size;
        }

        /*
        Checks if this CUDAArray is empty or not.
        @return a bool that indicates whether this CUDAArray is empty.
        */
        virtual bool IsEmpty() const
        {
            return 0 == size;
        }

        void CopyToCPU(T* dest) const
        {
            CheckCUDAStatus(cudaMemcpy(dest, gpuData, sizeof(T) * size, cudaMemcpyDeviceToHost));
        }

        T *GetGPUPtr() const
        {
            return gpuData;
        }
    };
} // namespace DataStructures
