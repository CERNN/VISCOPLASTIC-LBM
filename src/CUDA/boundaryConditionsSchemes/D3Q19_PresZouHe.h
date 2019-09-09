/*
*   @file bcPresZouHeD3Q19.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Zou-He pressure boundary condition for D3Q19
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __BC_PRES_ZOUHE_D3Q19_H
#define __BC_PRES_ZOUHE_D3Q19_H

#include "./../globalFunctions.h"
#include <cuda_runtime.h>


/*
*   @brief Applies pressure Zou-He boundary condition on north wall node, given pressure
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param rho_w: node's densisty
*/
__device__
void gpuBCPresZouHeN(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat rho_w);


/*
*   @brief Applies pressure Zou-He boundary condition on south wall node, given pressure
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param rho_w: node's densisty
*/
__device__
void gpuBCPresZouHeS(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat rho_w);


/*
*   @brief Applies pressure Zou-He boundary condition on west wall node, given pressure
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param rho_w: node's densisty
*/
__device__
void gpuBCPresZouHeW(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat rho_w);


/*
*   @brief Applies pressure Zou-He boundary condition on east wall node, given pressure
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param rho_w: node's densisty
*/
__device__
void gpuBCPresZouHeE(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat rho_w);


/*
*   @brief Applies pressure Zou-He boundary condition on front wall node, given pressure
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param rho_w: node's densisty
*/
__device__
void gpuBCPresZouHeF(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat rho_w);


/*
*   @brief Applies pressure Zou-He boundary condition on back wall node, given pressure
*   @param f[(NX, NY, NZ, Q)]: grid of populations
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param rho_w: node's densisty
*/
__device__
void gpuBCPresZouHeB(dfloat* f, const short unsigned int x, const short unsigned int y,
   const short unsigned int z, const dfloat rho_w);


#endif // !__BC_PRES_ZOUHE_D3Q19_H