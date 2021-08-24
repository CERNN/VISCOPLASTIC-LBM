# First argument is to define the velocity set to compile, must be the same
# as in "var.h"
# Second argument is used to define the prefix of the executable. It is usually 
# defined equal to the "ID_SIM" of the "var.h" file

# example of usage is:
# sh compile.sh D3Q19 011
# sh compile.sh D3Q27 202

# Compute capbility, change it to the compute capability of your device
# Example: 35 stands for compute capability 3.5, 70 for CC 7.0, etc.
CC=37

if [[ "$1" = "D3Q19" || "$1" = "D3Q27" ]]
then
    nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v -O3 --restrict \
        ./IBM/*.cu ./IBM/*.cpp \
        ./IBM/structs/*.cpp ./IBM/structs/*.cu \
        ./IBM/collision/*.cu \
        *.cu *.cpp \
        ./boundaryConditionsSchemes/*.cu \
        -lcudadevrt -lcurand -o ./../../bin/$2sim_$1_sm${CC}
else
    echo "Input error, example of usage is"
    echo "sh compile.sh D3Q19 011"
    echo "sh compile.sh D3Q27 202"
fi