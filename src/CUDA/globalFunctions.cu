#include "globalFunctions.h"

__global__
void copyFromArray(dfloat3SoA dst, dfloat3SoA src){
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= (NZ+4))
        return;

    size_t i = idxScalar(x, y, z);

    dst.x[i] = src.x[i];
    dst.y[i] = src.y[i];
    dst.z[i] = src.z[i];
}