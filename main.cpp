#include <iostream>

#include "matrix.hpp"

int main(void)
{
    DataStructure::List<int> myList = {1, 2, 3, 4};
    std::cout << myList.PopFront() << std::endl;
    std::cout << myList.PopFront() << std::endl;
    std::cout << myList.PopFront() << std::endl;
    std::cout << myList.PopFront() << std::endl;
    for (int i = 0; i < 100; i++)
        myList.Append(1);
    myList.Clear();
    std::cout << myList.Size() << std::endl;
    return 0;
}