#include <iostream>

#include "binary_search.hpp"

int main(void)
{
    int arr[] = {1, 2, 3, 4, 4, 5, 6, 19, 23};
    std::cout << Algorithms::SearchRange<int, int[9], int, 9>(arr, 0) << std::endl;
    return 0;
}