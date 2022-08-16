#ifndef GPUARRAY_H
#define GPUARRAY_H

namespace GPUArray
{
    template <class ArrayType, class IndexType>
    __global__ void PopulateArray(ArrayType *arr, IndexType size, ArrayType value)
    {
        const IndexType i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < size)
            arr[i] = value;
    }
} // namespace GPUArray

#endif // GPUARRAY_H
