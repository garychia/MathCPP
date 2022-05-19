#ifndef BINARY_SEARCH_HPP
#define BINARY_SEARCH_HPP

#include <cstddef>

#include "../DataStructures/tuple.hpp"

namespace Algorithms
{
    /*
    Performs Binary Search on an array-like data structure.
    @param arr an array-like data structure.
    @param target the target to find.
    @param start the first index within the search range. (optional)
    @param end the last index within the search range. (optional)
    */
    template <class IndexType, class ArrayLike, class TargetType, std::size_t ArraySize>
    IndexType BinarySearch(const ArrayLike &arr, const TargetType &target, IndexType start = 0, IndexType end = -1)
    {
        if (end == -1)
            end = ArraySize - 1;
        while (start <= end)
        {
            const IndexType middle = start + (end - start) / 2;
            if (arr[middle] == target)
                return middle;
            else if (target < arr[middle])
                end = middle - 1;
            else
                start = middle + 1;
        }
        return -1;
    }

    /*
    Finds the starting and ending indices of a target.
    @param arr an array-like data structure.
    @param target the target to find.
    @param start the first index within the search range (inclusive). (optional)
    @param end the last index within the search range (inclusive). (optional)
    */
    template <class IndexType, class ArrayLike, class TargetType, std::size_t ArraySize>
    DataStructure::Tuple<IndexType> SearchRange(const ArrayLike &arr, const TargetType &target, IndexType start = 0, IndexType end = -1)
    {
        IndexType startIndex = -1;
        IndexType endIndex = -1;
        end = (end == -1) ? ArraySize : end + 1;

        while (start < end)
        {
            const IndexType middle = start + (end - start) / 2;
            if (arr[middle] == target)
                end = middle;
            else if (target < arr[middle])
                end = middle;
            else
                start = middle + 1;
        }
        if (start >= ArraySize || arr[start] != target)
            return DataStructure::Tuple<IndexType>(2, -1);
        
        startIndex = start;
        end = ArraySize;
        while (start < end - 1)
        {
            const IndexType middle = start + (end - start) / 2;
            if (arr[middle] == target)
                start = middle;
            else if (target < arr[middle])
                end = middle;
            else
                start = middle + 1;
        }
        endIndex = start;
        if (endIndex >= ArraySize || arr[endIndex] != target)
            return DataStructure::Tuple<IndexType>(2, -1);
        return DataStructure::Tuple<IndexType>({startIndex, endIndex});
    }
} // namespace Algorithms

#endif