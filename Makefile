macCompileMain: main.cpp vector3d.hpp
	g++ -Wall -std=c++17 -Xclang -fopenmp ./main.cpp -lomp -o main

all:
	macCompileMain

clear:
	rm main