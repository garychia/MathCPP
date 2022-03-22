compileMain: main.cpp vector3d.hpp
	g++ -std=c++17 ./main.cpp -o main

all:
	compileMain

clear:
	rm main