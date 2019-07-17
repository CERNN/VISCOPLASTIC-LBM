#!/bin/bash

# change sm_20 to match the architecture of your device
# allowed values are sm_Nn
# where N is the major version number and n is the minor number
# 'sm_20','sm_21','sm_30','sm_32','sm_35','sm_37','sm_50','sm_52','sm_53'

#-Xptxas -v,abi=no for printing the number of lmem bytes for each kernel
nvcc -gencode arch=compute_35,code=sm_35 -rdc=true --ptxas-options=-v -O3 *.cu ./../common/*.cpp -lcudadevrt -o ./exe/$1sim_sm35
