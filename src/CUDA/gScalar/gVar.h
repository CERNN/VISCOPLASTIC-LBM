
#ifndef __G_VAR_H
#define __G_VAR_H

#include <builtin_types.h> // for device variables


#define gD3Q19

constexpr dfloat G_TAU = 0.8;              // relaxation time
                         
constexpr dfloat G_OMEGA = 1.0/G_TAU;        // (tau)^-1
constexpr dfloat G_T_OMEGA = 1-G_OMEGA;      // 1-omega, for collision
constexpr dfloat G_TT_OMEGA = 1-0.5*G_OMEGA; // 1-0.5*omega, for force term


#ifdef gD3Q7
#include "velocitySets/gD3Q7.h"
#endif // !D3Q19
#ifdef gD3Q19
#include "velocitySets/gD3Q19.h"
#endif // !D3Q27


#endif // !__G_VAR_H