/*
*   @file globalFunctions.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Global inline functions, utilized by many files
*   @version 0.3.0
*   @date 16/12/2019
*/

/*
*   LBM-CERNN
*   Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation; either version 2 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License along
*   with this program; if not, write to the Free Software Foundation, Inc.,
*   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*   Contact: cernn-ct@utfpr.edu.br and waine@alunos.utfpr.edu.br
*/

#ifndef __GLOBAL_FUNCTIONS_H
#define __GLOBAL_FUNCTIONS_H

#include <builtin_types.h>
#include <curand_kernel.h>
#include <curand.h>

#include "var.h"
#include "IBM/ibmVar.h"
#include "structs/globalStructs.h"

/*
*   @brief Evaluate the population of equilibrium
*   @param rhow: product between density and population's weight
*   @param uc3: three times the scalar product of velocity and the discretized 
*               velocity i (3 * u * ci)
*   @param p1_mmu: 1 minus the scalar product of velocity and itself times 1.5 
*                  (1 - 1.5 * u * u)
*   @return equilibrium population
*/
__device__
dfloat __forceinline__ gpu_f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) -> 
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}


/*
*   @brief Evaluate the force term
*   @param wi: population's weight
*   @param fx_mult: term to multiply force in x
*   @param fy_mult: term to multiply force in y
*   @param fz_mult: term to multiply force in z
*   @return force term for population
*/
__device__
dfloat __forceinline__ gpu_force_term(const dfloat wi, const dfloat fx_mult, 
    const dfloat fy_mult, const dfloat fz_mult)
{
    // i: population number
    // a, b: alfa and beto for somatory
    // F_{i} = w_{i}[3*c_{ia}+9*u_{b}*(c_{ia}-d_{ab}/3)]*F_{a}
    return (wi * (FX*fx_mult + FY*fy_mult + FZ*fz_mult));
}


/*
*   @brief Generate normalized random number in gaussian distribution
*   @param seed: seed of rand
*   @param seq: value that must be different for each call (unless the same 
*               distribution is wanted)
*   @param sigma: sigma value interval for gaussian
*   @return normalized random number
*/
__device__
dfloat __forceinline__ randGauss(const unsigned long long int seed, const unsigned long long int seq, const dfloat sigma)
{
    curandState_t state;

    /* initialize the state */
    curand_init(seed, /* the seed controls the sequence of random values that are produced */
        seq, /* the sequence number is important for multiple cores */
        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
        &state);
    dfloat rand = curand_normal_double(&state);
    while(rand > sigma || rand < -sigma)
        rand = curand_normal_double(&state);
    return rand/sigma;
}


/*
*   @brief Evaluate the position of the element of a 3D matrix ([NX][NY][NZ]) 
*         in a 1D array
*   @param x: x axis value
*   @param y: y axis value
*   @param z: z axis value
*   @return element index
*/
__host__ __device__
size_t __forceinline__ idxScalar(unsigned int x, unsigned int y, unsigned int z)
{
    return NX * ((size_t)NY*z + y) + x;
}


/*
*   @brief Evaluate the position of the element of a 3D matrix ([NX][NY][NZ+2*MACR_BORDER_NODES]) 
*         in a 1D array
*   @param x: x axis value
*   @param y: y axis value
*   @param z: z axis value (-MACR_BORDER_NODES <= z < NZ+MACR_BORDER_NODES)
*   @return element index
*/
__host__ __device__
size_t __forceinline__ idxScalarWBorder(unsigned int x, unsigned int y, unsigned int z)
{
    return NX * ((size_t)NY*(z+MACR_BORDER_NODES) + y) + x;
}



/*
*   @brief Evaluate the element of the population of a 4D matrix 
*          ([NX][NY][NZ+1][Q]) in a 1D array
*   @param x: x axis value
*   @param y: y axis value
*   @param z: z axis value
*   @param d: population number
*   @return element index
*/
__host__ __device__
size_t __forceinline__ idxPop(const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int d)
{
    return NX*(NY*((size_t)(NZ+1)*d + z) + y) + x;
}


/*
*   @brief Evaluate the distance between a point in 2D and another
*   @param x1: point 1 x value
*   @param y1: point 1 y value
*   @param x2: point 2 x value
*   @param y2: point 2 y value
*   @return euclidian distance between the two points 
*/
__host__ __device__
dfloat __forceinline__ distPoints2D(const dfloat x1, const dfloat y1, const dfloat x2, const dfloat y2)
{   
    return sqrt((float)(x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

/**
*   @brief Copy values from src to dst (shape [NZ, NY, NX])
*
*   @param dst: destiny arrays
*   @param src: source arrays
*/
__global__
void copyFromArray(dfloat3SoA dst, dfloat3SoA src);

/**
*   @brief Clamp a value to a given range.
*   @param value: The value to be clamped.
*   @param minVal: The minimum allowable value.
*   @param maxVal: The maximum allowable value.
*   @return The clamped value.
*/
__host__ __device__
dfloat clamp01(dfloat value);
/**
*   @brief Linearly interpolate between two vectors.
*   @param v1: The start vector.
*   @param v2: The end vector.
*   @param t: Interpolation factor between 0 and 1.
*   @return The interpolated vector between v1 and v2.
*/
__host__ __device__
dfloat3 vector_lerp(dfloat3 v1, dfloat3 v2, dfloat t);

/**
*   @brief Compute the cross product of two vectors.
*   @param v1: First vector.
*   @param v2: Second vector.
*   @return The cross product vector of v1 and v2.
*/
__host__ __device__
dfloat3 cross_product(dfloat3 v1, dfloat3 v2);
/**
*   @brief Compute the dot product of two vectors.
*   @param v1: First vector.
*   @param v2: Second vector.
*   @return The dot product of v1 and v2.
*/
__host__ __device__
dfloat dot_product(dfloat3 v1, dfloat3 v2);

/**
*   @brief Determine the length of a vector
*   @param v: Vector to be computed.
*   @return The vector length.
*/
__host__ __device__
dfloat vector_length(dfloat3 v);

/**
*   @brief Normalize a vector.
*   @param v: Vector to be normalized.
*   @return The normalized vector.
*/
__host__ __device__
dfloat3 vector_normalize(dfloat3 v);

/**
*   @brief Compute the transpose of a 3x3 matrix.
*   @param matrix: The input 3x3 matrix to be transposed.
*   @param result: The output 3x3 matrix that will contain the transposed matrix.
*/
__host__ __device__
void transpose_matrix_3x3(dfloat matrix[3][3], dfloat result[3][3]);

/**
*   @brief Multiply two 3x3 matrices.
*   @param A: The first 3x3 matrix to be multiplied.
*   @param B: The second 3x3 matrix to be multiplied.
*   @param result: The output 3x3 matrix that will contain the product of matrices A and B..
*/
__host__ __device__
void multiply_matrices_3x3(dfloat A[3][3], dfloat B[3][3], dfloat result[3][3]);

/**
*   @brief Perform quaternion multiplication.
*   @param q1: First quaternion.
*   @param q2: Second quaternion.
*   @return The product of q1 and q2.
*/
__host__ __device__
dfloat4 quart_multiplication(dfloat4 q1, dfloat4 q2);

/**
*   @brief Compute the conjugate of a quaternion.
*   @param q: Quaternion to be conjugated.
*   @return The conjugate of q.
*/
__host__ __device__
dfloat4 quart_conjugate(dfloat4 q);

/**
*   @brief Convert a quaternion to a rotation matrix.
*   @param q: Quaternion to be converted.
*   @param R: Output rotation matrix.
*/
__host__ __device__
void quart_to_rotation_matrix(dfloat4 q, dfloat R[3][3]);

/**
*   @brief Rotate a vector by a rotation matrix.
*   @param v: Vector to be rotated.
*   @param R: Rotation matrix.
*   @return The rotated vector.
*/
__host__ __device__
dfloat3 rotate_vector_by_matrix(dfloat R[3][3],dfloat3 v);

/**
*   @brief Rotate a vector by a quaternion (using rotation matrix).
*   @param v: Vector to be rotated.
*   @param q: Quaternion representing rotation.
*   @return The rotated vector.
*/
__host__ __device__
dfloat3 rotate_vector_by_quart_R(dfloat3 v, dfloat4 q);


/**
*   @brief Compute the rotation quaternion that aligns two vectors.
*   @param v1: First vector.
*   @param v2: Second vector.
*   @return Quaternion representing the rotation from v1 to v2.
*/
__host__ __device__
dfloat4 compute_rotation_quart(dfloat3 v1, dfloat3 v2);

/**
*   @brief Convert an axis-angle representation to a quaternion.
*   @param axis: Rotation axis.
*   @param angle: Rotation angle.
*   @return The quaternion representation of the axis-angle rotation.
*/
__host__ __device__
dfloat4 axis_angle_to_quart(dfloat3 axis, dfloat angle);

/**
*   @brief Convert Euler angles to a quaternion.
*   @param roll: Rotation angle around x-axis.
*   @param pitch: Rotation angle around y-axis.
*   @param yaw: Rotation angle around z-axis.
*   @return The quaternion representation of the Euler angles.
*   @source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
*/
__host__ __device__
dfloat4 euler_to_quart(dfloat roll, dfloat pitch, dfloat yaw);

/**
*   @brief Convert a quaternion to Euler angles.
*   @param q: Quaternion to be converted.
*   @return Euler angles representing the rotation of q.
*/
__host__ __device__
dfloat3 quart_to_euler(dfloat4 q);

/**
*   @brief Rotate a 3x3 matrix using a quaternion.
*   @param q: The quaternion representing the rotation.
*   @param I: The 3x3 matrix to be rotated.
*/
__host__ __device__
void rotate_matrix_by_R_w_quart(dfloat4 q, dfloat I[3][3]);

/**
*   @brief Rotate an inertia tensor represented as a 6-component structure using a quaternion.
*   @param q: The quaternion representing the rotation.
*   @param I6: The inertia tensor in the form of a 6-component structure.
*   @return The rotated inertia tensor as a 6-component structure.
*/
__host__ __device__
dfloat6 rotate_inertia_by_quart(dfloat4 q, dfloat6 I6);


/**
*   @brief Convert a dfloat6 structure to a 3x3 matrix.
*   @param I: dfloat6 structure containing inertia tensor components.
*   @param invA: Output 3x3 matrix.
*/
__host__ __device__
void dfloat6_to_matrix(dfloat6 I, dfloat M[3][3]);

/**
*   @brief Convert a 3x3 matrix to a dfloat6 structure.
*   @param M: Input 3x3 matrix.
*   @return I: dfloat6 structure to store the inertia tensor components.
*/
__host__ __device__
dfloat6 matrix_to_dfloat6(dfloat M[3][3]);



#endif // !__GLOBAL_FUNCTIONS_H
