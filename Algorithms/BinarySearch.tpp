namespace Algorithms
{
    template <class IndexType, class ArrayLike, class TargetType, std::size_t ArraySize>
    IndexType BinarySearch(const ArrayLike &arr, const TargetType &target, IndexType start, IndexType end)
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

    template <class IndexType, class ArrayLike, class TargetType, std::size_t ArraySize>
    DataStructures::Tuple<IndexType> SearchRange(const ArrayLike &arr, const TargetType &target, IndexType start, IndexType end)
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
            return DataStructures::Tuple<IndexType>(2, -1);

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
            return DataStructures::Tuple<IndexType>(2, -1);
        return DataStructures::Tuple<IndexType>({startIndex, endIndex});
    }
}