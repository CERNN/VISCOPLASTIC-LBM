/*
*   @file bounceBack.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Bounce back boundary condition
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __BC_BOUNCE_BACK_H
#define __BC_BOUNCE_BACK_H

#include "./../globalFunctions.h"
#include "./../structs/nodeTypeMap.h"
#include <cuda_runtime.h>


/*
*   @brief Applies bounce back boundary condition on north wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackN(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on south wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackS(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on west wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on east wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northwest wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northeast wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on north-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on north-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southwest wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southeast wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on south-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on south-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on west-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackWF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on west-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackWB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on east-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackEF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on east-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackEB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northwest-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNWF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northwest-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNWB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northeast-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNEF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on northeast-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackNEB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southwest-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSWF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southwest-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSWB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southeast-front wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSEF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition on southeast-back wall node
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__ 
void gpuBCBounceBackSEB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z);


#endif // !__BC_BOUNCE_BACK_H
