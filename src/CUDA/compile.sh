#!/bin/bash

# change sm_20 to match the architecture of your device
# allowed values are sm_Nn
# where N is the major version number and n is the minor number
# 'sm_20','sm_21','sm_30','sm_32','sm_35','sm_37','sm_50','sm_52','sm_53'

#-Xptxas -v,abi=no for printing the number of lmem bytes for each kernel

# first argument passed is used to define the prefix of the executable
# it is usually defined equal to the "ID_SIM" of the var.h file

# example of usage is:
# sh compile.sh 001

nvcc -gencode arch=compute_35,code=sm_35 -rdc=true --ptxas-options=-v -O3 ./boundaryConditionsSchemes/D3Q19*.cu *.cu *.cpp -lcudadevrt -o ./bin/$1sim_sm35