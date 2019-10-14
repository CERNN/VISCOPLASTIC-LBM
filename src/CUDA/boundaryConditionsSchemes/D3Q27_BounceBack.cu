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

#include "D3Q27_BounceBack.h"

__device__ 
void gpuBCBounceBackN(dfloat* fNode)
{
    fNode[(4)] = fNode[(3)];
    fNode[(8)] = fNode[(7)];
    fNode[(12)] = fNode[(11)];
    fNode[(13)] = fNode[(14)];
    fNode[(18)] = fNode[(17)];
    fNode[(20)] = fNode[(19)];
    fNode[(22)] = fNode[(21)];
    fNode[(23)] = fNode[(24)];
    fNode[(26)] = fNode[(25)];
}


__device__ 
void gpuBCBounceBackS(dfloat* fNode)
{
    fNode[(3)] = fNode[(4)];
    fNode[(7)] = fNode[(8)];
    fNode[(11)] = fNode[(12)];
    fNode[(14)] = fNode[(13)];
    fNode[(17)] = fNode[(18)];
    fNode[(19)] = fNode[(20)];
    fNode[(21)] = fNode[(22)];
    fNode[(24)] = fNode[(23)];
    fNode[(25)] = fNode[(26)];
}


__device__ 
void gpuBCBounceBackW(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(7)] = fNode[(8)];
    fNode[(9)] = fNode[(10)];
    fNode[(13)] = fNode[(14)];
    fNode[(15)] = fNode[(16)];
    fNode[(19)] = fNode[(20)];
    fNode[(21)] = fNode[(22)];
    fNode[(23)] = fNode[(24)];
    fNode[(26)] = fNode[(25)];
}


__device__ 
void gpuBCBounceBackE(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(8)] = fNode[(7)];
    fNode[(10)] = fNode[(9)];
    fNode[(14)] = fNode[(13)];
    fNode[(16)] = fNode[(15)];
    fNode[(20)] = fNode[(19)];
    fNode[(22)] = fNode[(21)];
    fNode[(24)] = fNode[(23)];
    fNode[(25)] = fNode[(26)];
}


__device__ 
void gpuBCBounceBackF(dfloat* fNode)
{
    fNode[(6)] = fNode[(5)];
    fNode[(10)] = fNode[(9)];
    fNode[(12)] = fNode[(11)];
    fNode[(15)] = fNode[(16)];
    fNode[(17)] = fNode[(18)];
    fNode[(20)] = fNode[(19)];
    fNode[(21)] = fNode[(22)];
    fNode[(24)] = fNode[(23)];
    fNode[(26)] = fNode[(25)];
}


__device__ 
void gpuBCBounceBackB(dfloat* fNode)
{
    fNode[(5)] = fNode[(6)];
    fNode[(9)] = fNode[(10)];
    fNode[(11)] = fNode[(12)];
    fNode[(16)] = fNode[(15)];
    fNode[(18)] = fNode[(17)];
    fNode[(19)] = fNode[(20)];
    fNode[(22)] = fNode[(21)];
    fNode[(23)] = fNode[(24)];
    fNode[(25)] = fNode[(26)];
}


__device__ 
void gpuBCBounceBackNW(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(4)] = fNode[(3)];
    fNode[(9)] = fNode[(10)];
    fNode[(12)] = fNode[(11)];
    fNode[(13)] = fNode[(14)];
    fNode[(15)] = fNode[(16)];
    fNode[(18)] = fNode[(17)];
    fNode[(23)] = fNode[(24)];
    fNode[(26)] = fNode[(25)];
    //Dead Pop are: [7, 8, 19, 20, 21, 22]
}


__device__ 
void gpuBCBounceBackNE(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(4)] = fNode[(3)];
    fNode[(8)] = fNode[(7)];
    fNode[(10)] = fNode[(9)];
    fNode[(12)] = fNode[(11)];
    fNode[(16)] = fNode[(15)];
    fNode[(18)] = fNode[(17)];
    fNode[(20)] = fNode[(19)];
    fNode[(22)] = fNode[(21)];
    //Dead Pop are: [13, 14, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackNF(dfloat* fNode)
{
    fNode[(4)] = fNode[(3)];
    fNode[(6)] = fNode[(5)];
    fNode[(8)] = fNode[(7)];
    fNode[(10)] = fNode[(9)];
    fNode[(12)] = fNode[(11)];
    fNode[(13)] = fNode[(14)];
    fNode[(15)] = fNode[(16)];
    fNode[(20)] = fNode[(19)];
    fNode[(26)] = fNode[(25)];
    //Dead Pop are: [17, 18, 21, 22, 23, 24]
}


__device__ 
void gpuBCBounceBackNB(dfloat* fNode)
{
    fNode[(4)] = fNode[(3)];
    fNode[(5)] = fNode[(6)];
    fNode[(8)] = fNode[(7)];
    fNode[(9)] = fNode[(10)];
    fNode[(13)] = fNode[(14)];
    fNode[(16)] = fNode[(15)];
    fNode[(18)] = fNode[(17)];
    fNode[(22)] = fNode[(21)];
    fNode[(23)] = fNode[(24)];
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
}


__device__ 
void gpuBCBounceBackSW(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(3)] = fNode[(4)];
    fNode[(7)] = fNode[(8)];
    fNode[(9)] = fNode[(10)];
    fNode[(11)] = fNode[(12)];
    fNode[(15)] = fNode[(16)];
    fNode[(17)] = fNode[(18)];
    fNode[(19)] = fNode[(20)];
    fNode[(21)] = fNode[(22)];
    //Dead Pop are: [13, 14, 23, 24, 25, 26]
}



__device__ 
void gpuBCBounceBackSE(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(3)] = fNode[(4)];
    fNode[(10)] = fNode[(9)];
    fNode[(11)] = fNode[(12)];
    fNode[(14)] = fNode[(13)];
    fNode[(16)] = fNode[(15)];
    fNode[(17)] = fNode[(18)];
    fNode[(24)] = fNode[(23)];
    fNode[(25)] = fNode[(26)];
    //Dead Pop are: [7, 8, 19, 20, 21, 22]
}


__device__ 
void gpuBCBounceBackSF(dfloat* fNode)
{
    fNode[(3)] = fNode[(4)];
    fNode[(6)] = fNode[(5)];
    fNode[(7)] = fNode[(8)];
    fNode[(10)] = fNode[(9)];
    fNode[(14)] = fNode[(13)];
    fNode[(15)] = fNode[(16)];
    fNode[(17)] = fNode[(18)];
    fNode[(21)] = fNode[(22)];
    fNode[(24)] = fNode[(23)];
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
}


__device__ 
void gpuBCBounceBackSB(dfloat* fNode)
{
    fNode[(3)] = fNode[(4)];
    fNode[(5)] = fNode[(6)];
    fNode[(7)] = fNode[(8)];
    fNode[(9)] = fNode[(10)];
    fNode[(11)] = fNode[(12)];
    fNode[(14)] = fNode[(13)];
    fNode[(16)] = fNode[(15)];
    fNode[(19)] = fNode[(20)];
    fNode[(25)] = fNode[(26)];
    //Dead Pop are: [17, 18, 21, 22, 23, 24]
}


__device__ 
void gpuBCBounceBackWF(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(6)] = fNode[(5)];
    fNode[(7)] = fNode[(8)];
    fNode[(12)] = fNode[(11)];
    fNode[(13)] = fNode[(14)];
    fNode[(15)] = fNode[(16)];
    fNode[(17)] = fNode[(18)];
    fNode[(21)] = fNode[(22)];
    fNode[(26)] = fNode[(25)];
    //Dead Pop are: [9, 10, 19, 20, 23, 24]
}


__device__ 
void gpuBCBounceBackWB(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(5)] = fNode[(6)];
    fNode[(7)] = fNode[(8)];
    fNode[(9)] = fNode[(10)];
    fNode[(11)] = fNode[(12)];
    fNode[(13)] = fNode[(14)];
    fNode[(18)] = fNode[(17)];
    fNode[(19)] = fNode[(20)];
    fNode[(23)] = fNode[(24)];
    //Dead Pop are: [15, 16, 21, 22, 25, 26]

}


__device__ 
void gpuBCBounceBackEF(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(6)] = fNode[(5)];
    fNode[(8)] = fNode[(7)];
    fNode[(10)] = fNode[(9)];
    fNode[(12)] = fNode[(11)];
    fNode[(14)] = fNode[(13)];
    fNode[(17)] = fNode[(18)];
    fNode[(20)] = fNode[(19)];
    fNode[(24)] = fNode[(23)];
    //Dead Pop are: [15, 16, 21, 22, 25, 26]
}


__device__ 
void gpuBCBounceBackEB(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(5)] = fNode[(6)];
    fNode[(8)] = fNode[(7)];
    fNode[(11)] = fNode[(12)];
    fNode[(14)] = fNode[(13)];
    fNode[(16)] = fNode[(15)];
    fNode[(18)] = fNode[(17)];
    fNode[(22)] = fNode[(21)];
    fNode[(25)] = fNode[(26)];
    //Dead Pop are: [9, 10, 19, 20, 23, 24]
}


__device__ 
void gpuBCBounceBackNWF(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(4)] = fNode[(3)];
    fNode[(6)] = fNode[(5)];
    fNode[(12)] = fNode[(11)];
    fNode[(13)] = fNode[(14)];
    fNode[(15)] = fNode[(16)];
    fNode[(26)] = fNode[(25)];
    //Dead Pop are: [7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24]
}


__device__ 
void gpuBCBounceBackNWB(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(4)] = fNode[(3)];
    fNode[(5)] = fNode[(6)];
    fNode[(9)] = fNode[(10)];
    fNode[(13)] = fNode[(14)];
    fNode[(18)] = fNode[(17)];
    fNode[(23)] = fNode[(24)];
    //Dead Pop are: [7, 8, 11, 12, 15, 16, 19, 20, 21, 22, 25, 26]
}


__device__ 
void gpuBCBounceBackNEF(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(4)] = fNode[(3)];
    fNode[(6)] = fNode[(5)];
    fNode[(8)] = fNode[(7)];
    fNode[(10)] = fNode[(9)];
    fNode[(12)] = fNode[(11)];
    fNode[(20)] = fNode[(19)];
    //Dead Pop are: [13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackNEB(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(4)] = fNode[(3)];
    fNode[(5)] = fNode[(6)];
    fNode[(8)] = fNode[(7)];
    fNode[(16)] = fNode[(15)];
    fNode[(18)] = fNode[(17)];
    fNode[(22)] = fNode[(21)];
    //Dead Pop are: [9, 10, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26]

}


__device__ 
void gpuBCBounceBackSWF(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(3)] = fNode[(4)];
    fNode[(6)] = fNode[(5)];
    fNode[(7)] = fNode[(8)];
    fNode[(15)] = fNode[(16)];
    fNode[(17)] = fNode[(18)];
    fNode[(21)] = fNode[(22)];
    //Dead Pop are: [9, 10, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackSWB(dfloat* fNode)
{
    fNode[(1)] = fNode[(2)];
    fNode[(3)] = fNode[(4)];
    fNode[(5)] = fNode[(6)];
    fNode[(7)] = fNode[(8)];
    fNode[(9)] = fNode[(10)];
    fNode[(11)] = fNode[(12)];
    fNode[(19)] = fNode[(20)];
    //Dead Pop are: [13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
}


__device__ 
void gpuBCBounceBackSEF(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(3)] = fNode[(4)];
    fNode[(6)] = fNode[(5)];
    fNode[(10)] = fNode[(9)];
    fNode[(14)] = fNode[(13)];
    fNode[(17)] = fNode[(18)];
    fNode[(24)] = fNode[(23)];
    //Dead Pop are: [7, 8, 11, 12, 15, 16, 19, 20, 21, 22, 25, 26]
}


__device__ 
void gpuBCBounceBackSEB(dfloat* fNode)
{
    fNode[(2)] = fNode[(1)];
    fNode[(3)] = fNode[(4)];
    fNode[(5)] = fNode[(6)];
    fNode[(11)] = fNode[(12)];
    fNode[(14)] = fNode[(13)];
    fNode[(16)] = fNode[(15)];
    fNode[(25)] = fNode[(26)];
    //Dead Pop are: [7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24]
}