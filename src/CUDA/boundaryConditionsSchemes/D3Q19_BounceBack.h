/*
*   @file bcBounceBackD3Q19.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Bounce back boundary condition for D3Q19
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __BC_BOUNCE_BACK_D3Q19_H
#define __BC_BOUNCE_BACK_D3Q19_H

#include "./../globalFunctions.h"
#include <cuda_runtime.h>


/*
*   @brief Applies bounce back boundary condition on north wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackN(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on south wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackS(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on west wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackW(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on east wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackE(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on northwest wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNW(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on northeast wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNE(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on north-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on north-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on southwest wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSW(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on southeast wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSE(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on south-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on south-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on west-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackWF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on west-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackWB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on east-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackEF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on east-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackEB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on northwest-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNWF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on northwest-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNWB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on northeast-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNEF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on northeast-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackNEB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on southwest-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSWF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on southwest-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSWB(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on southeast-front wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSEF(dfloat* fNode);


/*
*   @brief Applies bounce back boundary condition on southeast-back wall node
*   @param fNode[Q]: populations to apply boundary conditions
*/
__device__ 
void gpuBCBounceBackSEB(dfloat* fNode);


#endif // !__BC_BOUNCE_BACK_D3Q19_H
