#ifndef BINARY_SEARCH_HPP
#define BINARY_SEARCH_HPP

#include "Tuple.hpp"

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
    IndexType BinarySearch(const ArrayLike &arr, const TargetType &target, IndexType start = 0, IndexType end = -1);

    /*
    Finds the starting and ending indices of a target.
    @param arr an array-like data structure.
    @param target the target to find.
    @param start the first index within the search range (inclusive). (optional)
    @param end the last index within the search range (inclusive). (optional)
    */
    template <class IndexType, class ArrayLike, class TargetType, std::size_t ArraySize>
    DataStructure::Tuple<IndexType> SearchRange(const ArrayLike &arr, const TargetType &target, IndexType start = 0, IndexType end = -1);
} // namespace Algorithms

#include "BinarySearch.tpp"

#endif