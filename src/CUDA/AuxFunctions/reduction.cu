/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For additional information on the license terms, see the CUDA EULA at
https://docs.nvidia.com/cuda/eula/index.html
*/


#include "reduction.cuh"


__global__ void reductionArray(float* g_idata, float* g_odata)
{
#if (BLOCK_LBM_SIZE == 512)
    __shared__ float sdata[BLOCK_LBM_SIZE];
#else
    extern __shared__ float sdata[];
#endif

    //global index in the array
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    //thread index in the array
    unsigned int tid = threadIdx.x;

    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));


    sdata[tid] = g_idata[i];
    //do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[bid] = sdata[0];
}

__global__ void reductionMacro(float* g_idata, float* g_odata)
{
#if (BLOCK_LBM_SIZE == 512)
    __shared__ float sdata[BLOCK_LBM_SIZE];
#else
    extern __shared__ float sdata[];
#endif

    const short unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const short unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const short unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    //global index in the array
    unsigned int i = idxScalar(x, y, z);

    //thread index in the array
    unsigned int tid = threadIdx.x;

    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));


    sdata[tid] = g_idata[i];
    //do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) 
        g_odata[bid] = sdata[0];
}