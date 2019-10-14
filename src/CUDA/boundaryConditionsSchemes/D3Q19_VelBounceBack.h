/*
*   @file bcVelBounceBackD3Q19.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Bounce back velocity boundary condition for D3Q19
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __BC_VEL_BOUNCE_BACK_D3Q19_H
#define __BC_VEL_BOUNCE_BACK_D3Q19_H

#include "./../globalFunctions.h"
#include <cuda_runtime.h>

/*
*   @brief Applies velocity bounce-back boundary condition on north wall node, 
*          given pressure
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: wall's x velocity
*   @param uy_w: wall's y velocity
*   @param uz_w: wall's z velocity
*/
__device__
void gpuBCVelBounceBackN(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity bounce-back boundary condition on south wall node, given pressure
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: wall's x velocity
*   @param uy_w: wall's y velocity
*   @param uz_w: wall's z velocity
*/
__device__
void gpuBCVelBounceBackS(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity bounce-back boundary condition on west wall node, given pressure
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: wall's x velocity
*   @param uy_w: wall's y velocity
*   @param uz_w: wall's z velocity
*/
__device__
void gpuBCVelBounceBackW(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity bounce-back boundary condition on east wall node, given pressure
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: wall's x velocity
*   @param uy_w: wall's y velocity
*   @param uz_w: wall's z velocity
*/
__device__
void gpuBCVelBounceBackE(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity bounce-back boundary condition on front wall node, given pressure
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: wall's x velocity
*   @param uy_w: wall's y velocity
*   @param uz_w: wall's z velocity
*/
__device__
void gpuBCVelBounceBackF(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);


/*
*   @brief Applies velocity bounce-back boundary condition on back wall node, given pressure
*   @param fNode[Q]: populations to apply boundary conditions
*   @param ux_w: wall's x velocity
*   @param uy_w: wall's y velocity
*   @param uz_w: wall's z velocity
*/
__device__
void gpuBCVelBounceBackB(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w);

#endif // !__BC_VEL_BOUNCE_BACK_D3Q19_H
