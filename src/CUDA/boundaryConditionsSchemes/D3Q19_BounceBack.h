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
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackN(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on south wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackS(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on west wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackW(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on east wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackE(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northwest wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNW(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northeast wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNE(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on north-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on north-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southwest wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSW(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southeast wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSE(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on south-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on south-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on west-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackWF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on west-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackWB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on east-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackEF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on east-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackEB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northwest-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNWF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northwest-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNWB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northeast-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNEF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northeast-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNEB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southwest-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSWF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southwest-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSWB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southeast-front wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSEF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southeast-back wall node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSEB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


#endif // !__BC_BOUNCE_BACK_D3Q19_H
