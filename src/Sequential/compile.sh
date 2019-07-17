#!/bin/bash

rm -f sim

rm -f LBM.o seconds.o main.o sim

CXXFLAGS="-std=c++17 -pedantic -O3 -Wall"

g++ ${CXXFLAGS} -c lbm2.cpp -o lbm2.o
g++ ${CXXFLAGS} -c seconds.cpp -o seconds.o
g++ ${CXXFLAGS} -c main2.cpp -o main2.o
 
g++ lbm2.o seconds.o main2.o -o exe/sim

rm -f *.o