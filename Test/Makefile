compile_test_main: ../vector3d.hpp ./vector3d_test.hpp ./test_main.cpp
	g++ -std=c++17 -Xclang -fopenmp -lgtest ./test_main.cpp -lomp -o test_main

all:
	compile_test_main

clear:
	rm ./test_main