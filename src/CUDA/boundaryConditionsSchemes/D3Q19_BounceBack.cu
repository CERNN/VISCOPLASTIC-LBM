/*
*   LBM-CERNN
*   Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation; either version 2 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License along
*   with this program; if not, write to the Free Software Foundation, Inc.,
*   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*   Contact: cernn-ct@utfpr.edu.br and waine@alunos.utfpr.edu.br
*/

#include "D3Q19_BounceBack.h"

__device__ 
void gpuBCBounceBackN(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
}


__device__ 
void gpuBCBounceBackS(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
}


__device__ 
void gpuBCBounceBackW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
}


__device__ 
void gpuBCBounceBackE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
}


__device__ 
void gpuBCBounceBackF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
}


__device__ 
void gpuBCBounceBackB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
}


__device__ 
void gpuBCBounceBackNW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [7, 8]
}


__device__ 
void gpuBCBounceBackNE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [13, 14]
}


__device__ 
void gpuBCBounceBackNF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    //Dead Pop are: [17, 18]
}


__device__ 
void gpuBCBounceBackNB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [11, 12]
}


__device__ 
void gpuBCBounceBackSW(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [13, 14]
}



__device__ 
void gpuBCBounceBackSE(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [7, 8]
}


__device__ 
void gpuBCBounceBackSF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [11, 12]
}


__device__ 
void gpuBCBounceBackSB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    //Dead Pop are: [17, 18]
}


__device__ 
void gpuBCBounceBackWF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [9, 10]
}


__device__ 
void gpuBCBounceBackWB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [15, 16]

}


__device__ 
void gpuBCBounceBackEF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [15, 16]
}


__device__ 
void gpuBCBounceBackEB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [9, 10]
}


__device__ 
void gpuBCBounceBackNWF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    //Dead Pop are: [7, 8, 9, 10, 17, 18]
}


__device__ 
void gpuBCBounceBackNWB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [7, 8, 11, 12, 15, 16]
}


__device__ 
void gpuBCBounceBackNEF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)];
    //Dead Pop are: [13, 14, 15, 16, 17, 18]
}


__device__ 
void gpuBCBounceBackNEB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)];
    //Dead Pop are: [9, 10, 11, 12, 13, 14]

}


__device__ 
void gpuBCBounceBackSWF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [9, 10, 11, 12, 13, 14]
}


__device__ 
void gpuBCBounceBackSWB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)];
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    //Dead Pop are: [13, 14, 15, 16, 17, 18]
}


__device__ 
void gpuBCBounceBackSEF(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)];
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)];
    //Dead Pop are: [7, 8, 11, 12, 15, 16]
}


__device__ 
void gpuBCBounceBackSEB(dfloat* f, const short unsigned int x, const short unsigned int y,
    const short unsigned int z)
{
    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 1)];
    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)];
    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)];
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)];
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)];
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)];
    //Dead Pop are: [7, 8, 9, 10, 17, 18]
}