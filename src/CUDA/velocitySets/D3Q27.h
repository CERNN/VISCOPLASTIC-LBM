#ifndef __D3Q27_H
#define __D3Q27_H

#include <builtin_types.h> // for device variables

/*
------ POPULATIONS -------
    [ i]: (cx,cy,cz)
    [ 0]: ( 0, 0, 0)
    [ 1]: ( 1, 0, 0)
    [ 2]: (-1, 0, 0)
    [ 3]: ( 0, 1, 0)
    [ 4]: ( 0,-1, 0)
    [ 5]: ( 0, 0, 1)
    [ 6]: ( 0, 0,-1)
    [ 7]: ( 1, 1, 0)
    [ 8]: (-1,-1, 0)
    [ 9]: ( 1, 0, 1)
    [10]: (-1, 0,-1)
    [11]: ( 0, 1, 1)
    [12]: ( 0,-1,-1)
    [13]: ( 1,-1, 0)
    [14]: (-1, 1, 0)
    [15]: ( 1, 0,-1)
    [16]: (-1, 0, 1)
    [17]: ( 0, 1,-1)
    [18]: ( 0,-1, 1)
    [19]: ( 1, 1, 1)
    [20]: (-1,-1,-1)
    [21]: ( 1, 1,-1)
    [22]: (-1,-1, 1)
    [23]: ( 1,-1, 1)
    [24]: (-1, 1,-1)
    [25]: (-1, 1, 1)
    [26]: ( 1,-1,-1)
--------------------------
*/

constexpr unsigned char Q = 27;         // number of velocities
constexpr dfloat W0 = 8.0 / 27;        // weight dist 0 population (0, 0, 0)
constexpr dfloat W1 = 2.0 / 27;        // weight dist 1 populations (1, 0, 0)
constexpr dfloat W2 = 1.0 / 54;        // weight dist 2 populations (1, 1, 0)
constexpr dfloat W3 = 1.0 / 216;       // weight dist 3 populations (1, 1, 1)
constexpr dfloat a_s_2 = 3.0; //inverse of c_s^2

// velocities weight vector
__device__ const dfloat w[Q] = { W0,
    W1, W1, W1, W1, W1, W1,
    W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2,
    W3, W3, W3, W3, W3, W3, W3, W3
};

// populations velocities vector
__device__ const char cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1};
__device__ const char cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1};
__device__ const char cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1};


#endif // !__D3Q27_H