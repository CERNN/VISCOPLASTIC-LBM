#include "gInitialization.h"


__global__
void gpuInitializationG(
    gPopulations* gPop,
    Macroscopics* macr)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalar(x, y, z);

    for (int i = 0; i < GQ; i++)
    {
        // calculate equilibrium population and initialize populations to equilibrium
        dfloat geq = gpu_g_eq(macr->G[index], macr->u.x[index],macr->u.y[index],macr->u.z[index],i);
        
        gPop->gPop[idxPop(x, y, z, i)] = geq;
        gPop->gPopAux[idxPop(x, y, z, i)] = geq;
    }
}