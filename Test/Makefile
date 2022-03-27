all: compile_test_main_omp compile_test_main

compile_test_main_omp: ../DataStructures/vector3d.hpp vector3d_test.hpp vector_test.hpp test_main.cpp
	g++ -Wall -g -std=c++17 -Xclang -fopenmp -lgtest test_main.cpp -lomp -o test_main_omp

compile_test_main: ../DataStructures/vector3d.hpp vector3d_test.hpp vector_test.hpp test_main.cpp
	g++ -Wall -g -std=c++17 -lgtest test_main.cpp -o test_main

clear:
	rm test_main test_main_omp