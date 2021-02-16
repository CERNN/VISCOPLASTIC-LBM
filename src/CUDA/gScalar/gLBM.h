/*
*   @file gLbm.h
*   @author Marco Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @brief Scalar global functions and collision-streaming process
*   @version 0.1.0
*   @date 16/08/2019
*/


#ifndef __G_LBM_H
#define __G_LBM_H

#include "..\structs\macroscopics.h"
#include "..\NNF\nnf.cuh"
//#include "..\structs\macrProc.h"
#include "gVar.h"

/*
*   @brief Calculates the source term
*/
__device__
dfloat __forceinline__ g_gpu_source_term()
{ 
    return  0.0;
}




/*
*   @brief Calculates the equilibrium distribution function for the additional scalar variable
*   @param lambda: scalar value
*   @param ux: fluid velocity in x direction
*   @param uy: fluid velocity in y direction
*   @param uz: fluid velocity in z direction
*   @param i: lattice velocity index
*/
__device__
dfloat __forceinline__ gpu_g_eq(const dfloat G, const dfloat ux, const dfloat uy, const dfloat uz, const char i)
{
    dfloat g_eq = 1.0;
    dfloat a =  1_gc_s_2* (gcx[i]*ux + gcy[i]*uy + gcz[i]*uz);
    dfloat b = - 0.5 * 1_gc_s_2 * (ux*ux + uy*uy + uz*uz);
      
    //g_eq += a;
    g_eq += a + a*a/2.0 + b;
    
    g_eq = gw[i] * G * g_eq;

    return g_eq;
}
