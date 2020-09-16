/**
*   @file globalStructs.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Global general structs
*   @version 0.3.0
*   @date 26/08/2020
*/

#ifndef __GLOBAL_STRUCTS_H
#define __GLOBAL_STRUCTS_H

#include "var.h"

typedef struct dfloat3 {
    dfloat x;
    dfloat y;
    dfloat z;

    dfloat3()
    {
        x = -1;
        y = -1;
        z = -1;
    }
} dfloat3;

typedef struct dfloat3SoA {
    dfloat* x; // x array
    dfloat* y; // y array
    dfloat* z; // z array

    dfloat3SoA()
    {
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

    ~dfloat3SoA()
    {
        x = nullptr;
        y = nullptr;
        z = nullptr;
    }

} dfloat3SoA;

#endif //__GLOBAL_STRUCTS_H
