/*
*   @file bcVelZouHeD3Q19.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Zou-He
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __BC_VEL_ZOUHE_D3Q19_H
#define __BC_VEL_ZOUHE_D3Q19_H

#include "./../globalFunctions.h"
#include <cuda_runtime.h>


/*
*   @brief Applies velocity Zou-He boundary condition on north wall node, given velocities
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeN(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on south wall node, given velocities
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeS(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on west wall node, given velocities
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeW(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on east wall node, given velocities
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeE(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on front wall node, given velocities
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeF(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity Zou-He boundary condition on back wall node, given velocities
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: node's x velocity
*   @param uy_w: node's y velocity
*   @param uz_w: node's z velocity
*/
__device__
void gpuBCVelZouHeB(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


#endif // !__BC_VEL_ZOUHE_D3Q19_H