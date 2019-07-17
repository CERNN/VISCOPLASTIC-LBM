# LBM

C++/CUDA project for LBM using BGK operator. Supports D2Q9 and D3Q19.

Report files about the code are in "doc/Reports" folder.

Compilation is made using shell file "compile.sh" for folders "src/CUDA/D3Q19/", "src/CUDA/D2Q9/" and "src/Sequential/"

"sm" represents de compute capability of the GPU. Check yours and use the right executable sm (or create a new shell with yours GPU sm).

The executables are in the respectives "exe/" folders

The results are stored in the "exe/tests/" folders

For changing between D2Q9 and D3Q19 in CUDA, use '#define D2Q9' or '#define D3Q19' in "common/func_idx.cuh"

Simulation parameters are in "var" files for CUDA and in "main.cpp" and "lbm2.h" for sequential

