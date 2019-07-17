#ifndef ERROR_DEF_H
#define ERROR_DEF_H

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

#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <device_launch_parameters.h>


#define checkCudaErrors(err)  __checkCudaErrors(err,#err,__FILE__,__LINE__)
#define getLastCudaError(msg)  __getLastCudaError(msg,__FILE__,__LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d)\"%s\": [%d] %s.\n",
            file, line, func, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}


inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n",
            file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

#endif // !ERROR_DEF_H