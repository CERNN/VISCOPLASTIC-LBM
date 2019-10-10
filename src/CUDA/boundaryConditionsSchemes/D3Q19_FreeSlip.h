/*
*   @file bcFreeSlipD3Q19.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Free slip boundary condition for D3Q19
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __BC_FREE_SLIP_D3Q19_H
#define __BC_FREE_SLIP_D3Q19_H

#include "./../globalFunctions.h"
#include <cuda_runtime.h>

/*
*   @brief Applies free slip boundary condition on north symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipN(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on south symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipS(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on west symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipW(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on east symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipE(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on front symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on back symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on northwest symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipNW(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on northeast symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipNE(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on north-front symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipNF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on north-back symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipNB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on southwest symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipSW(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on southeast symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipSE(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on south-front symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipSF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on south-back symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipSB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on west-front symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipWF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on west-back symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipWB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on east-front symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipEF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on east-back symmetry node
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipEB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);

#endif // !__BC_FREE_SLIP_D3Q19_H