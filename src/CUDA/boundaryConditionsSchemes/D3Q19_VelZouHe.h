/*
*   @file D3Q19_VelZouHe.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Zou-He velocity boundary condition for D3Q19
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __BC_VEL_ZOUHE_D3Q19_H
#define __BC_VEL_ZOUHE_D3Q19_H

#include "./../globalFunctions.h"
#include <cuda_runtime.h>


/*
*   @brief Applies velocity Zou-He boundary condition on north wall node, given velocities
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeN(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on south wall node, given velocities
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeS(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on west wall node, given velocities
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on east wall node, given velocities
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on front wall node, given velocities
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on back wall node, given velocities
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


#endif // !__BC_VEL_ZOUHE_D3Q19_H