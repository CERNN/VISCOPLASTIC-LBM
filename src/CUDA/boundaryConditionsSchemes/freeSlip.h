/*
*   @file freeSlip.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Free slip boundary condition
           For usage example, see templates
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __BC_FREE_SLIP_H
#define __BC_FREE_SLIP_H

#include "./../globalFunctions.h"
#include <cuda_runtime.h>

/*
*   @brief Applies free slip boundary condition on north symmetry node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipN(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on south symmetry node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipS(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on west symmetry node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on east symmetry node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on front symmetry node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies free slip boundary condition on back symmetry node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCFreeSlipB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);

#endif // !__BC_FREE_SLIP_H