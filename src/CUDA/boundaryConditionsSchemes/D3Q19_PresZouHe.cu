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

#include "D3Q19_PresZouHe.h"

#ifdef BC_SCHEME_PRES_ZOUHE
#ifdef D3Q19

__device__
void gpuBCPresZouHeN(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z, const dfloat rho_w)
{
    dfloat* f = fPostStream;
    const dfloat uy_w = -1 + (1/rho_w) * (f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 2)]
        + f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 9)]
        + f[idxPop(x, y, z, 15)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 10)]
        + f[idxPop(x, y, z, 0)] + 2 * (f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 7)]
            + f[idxPop(x, y, z, 14)] + f[idxPop(x, y, z, 11)] + f[idxPop(x, y, z, 17)]));

    const dfloat nyx = 0.5 * (f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 9)] + f[idxPop(x, y, z, 15)]
        - (f[idxPop(x, y, z, 2)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 10)]));
    const dfloat nyz = 0.5 * (f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 9)] + f[idxPop(x, y, z, 16)]
        - (f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 15)] + f[idxPop(x, y, z, 10)]));

    f[idxPop(x, y, z, 4)] = f[idxPop(x, y, z, 3)] + rho_w * (-uy_w) / 3;
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 7)] + rho_w * (-uy_w) / 6 + nyx;
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)] + rho_w * (-uy_w) / 6 - nyx;
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)] + rho_w * (-uy_w) / 6 + nyz;
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)] + rho_w * (-uy_w) / 6 - nyz;
}


__device__
void gpuBCPresZouHeS(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z, const dfloat rho_w)
{
    dfloat* f = fPostStream;
    const dfloat uy_w = 1 - (1/rho_w) * (f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 2)]
        + f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 9)]
        + f[idxPop(x, y, z, 15)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 10)]
        + f[idxPop(x, y, z, 0)] + 2 * (f[idxPop(x, y, z, 4)] + f[idxPop(x, y, z, 13)]
            + f[idxPop(x, y, z, 8)] + f[idxPop(x, y, z, 18)] + f[idxPop(x, y, z, 12)]));

    const dfloat nyx = 0.5 * (f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 9)] + f[idxPop(x, y, z, 15)]
        - (f[idxPop(x, y, z, 2)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 10)]));
    const dfloat nyz = 0.5 * (f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 9)] + f[idxPop(x, y, z, 16)]
        - (f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 15)] + f[idxPop(x, y, z, 10)]));

    f[idxPop(x, y, z, 3)] = f[idxPop(x, y, z, 4)] + rho_w * uy_w / 3;
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)] + rho_w * (uy_w) / 6 - nyx;
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 13)] + rho_w * (uy_w) / 6 + nyx;
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)] + rho_w * (uy_w) / 6 - nyz;
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)] + rho_w * (uy_w) / 6 + nyz;
}


__device__
void gpuBCPresZouHeW(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z, const dfloat rho_w)
{
    dfloat* f = fPostStream;
    const dfloat ux_w = 1 - (1/rho_w) * (f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 4)]
        + f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 11)]
        + f[idxPop(x, y, z, 17)] + f[idxPop(x, y, z, 18)] + f[idxPop(x, y, z, 12)]
        + f[idxPop(x, y, z, 0)] + 2 * (f[idxPop(x, y, z, 2)] + f[idxPop(x, y, z, 14)]
            + f[idxPop(x, y, z, 8)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 10)]));

    const dfloat nxy = 0.5 * (f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 11)] + f[idxPop(x, y, z, 17)]
        - (f[idxPop(x, y, z, 4)] + f[idxPop(x, y, z, 18)] + f[idxPop(x, y, z, 12)]));
    const dfloat nxz = 0.5 * (f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 18)] + f[idxPop(x, y, z, 11)]
        - (f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 17)] + f[idxPop(x, y, z, 12)]));

    f[idxPop(x, y, z, 1)] = f[idxPop(x, y, z, 2)] + rho_w * ux_w / 3;
    f[idxPop(x, y, z, 13)] = f[idxPop(x, y, z, 14)] + rho_w * (ux_w) / 6 + nxy;
    f[idxPop(x, y, z, 7)] = f[idxPop(x, y, z, 8)] + rho_w * (ux_w) / 6 - nxy;
    f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)] + rho_w * (ux_w) / 6 - nxz;
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)] + rho_w * (ux_w) / 6 + nxz;
}


__device__
void gpuBCPresZouHeE(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z, const dfloat rho_w)
{
    dfloat* f = fPostStream;
    const dfloat ux_w = -1 + (1/rho_w) * (f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 4)]
        + f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 11)]
        + f[idxPop(x, y, z, 17)] + f[idxPop(x, y, z, 18)] + f[idxPop(x, y, z, 12)]
        + f[idxPop(x, y, z, 0)] + 2 * (f[idxPop(x, y, z, 2)] + f[idxPop(x, y, z, 14)]
            + f[idxPop(x, y, z, 8)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 10)]));

    const dfloat nxy = 0.5 * (f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 11)] + f[idxPop(x, y, z, 17)]
        - (f[idxPop(x, y, z, 4)] + f[idxPop(x, y, z, 18)] + f[idxPop(x, y, z, 12)]));
    const dfloat nxz = 0.5 * (f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 18)] + f[idxPop(x, y, z, 11)]
        - (f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 17)] + f[idxPop(x, y, z, 12)]));

    f[idxPop(x, y, z, 2)] = f[idxPop(x, y, z, 2)] + rho_w * ux_w / 3;
    f[idxPop(x, y, z, 14)] = f[idxPop(x, y, z, 14)] + rho_w * (ux_w) / 6 + nxy;
    f[idxPop(x, y, z, 8)] = f[idxPop(x, y, z, 8)] + rho_w * (ux_w) / 6 - nxy;
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 10)] + rho_w * (ux_w) / 6 - nxz;
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 16)] + rho_w * (ux_w) / 6 + nxz;
}


__device__
void gpuBCPresZouHeF(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z, const dfloat rho_w)
{
    dfloat* f = fPostStream;
    const dfloat uz_w = -1 + (1 / rho_w) * (f[idxPop(x, y, z, 1)]  + f[idxPop(x, y, z, 2)]
        + f[idxPop(x, y, z, 3)]  + f[idxPop(x, y, z, 4)]  + f[idxPop(x, y, z, 7)]
        + f[idxPop(x, y, z, 14)] + f[idxPop(x, y, z, 8)]  + f[idxPop(x, y, z, 13)]
        + f[idxPop(x, y, z, 0)]  +2*(f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 9)]
        + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 11)] + f[idxPop(x, y, z, 18)]));
    
    const dfloat nzx = 0.5 * (f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 7)] + f[idxPop(x, y, z, 13)]
        - (f[idxPop(x, y, z, 2)] + f[idxPop(x, y, z, 14)] + f[idxPop(x, y, z, 8)]));
        // - rho_w*ux_w/3;
    
    const dfloat nzy = 0.5 * (f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 7)] + f[idxPop(x, y, z, 14)]
        - (f[idxPop(x, y, z, 4)] + f[idxPop(x, y, z, 13)] + f[idxPop(x, y, z, 8)]));
        // - rho_w*uy_w/3

    f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)] + rho_w*(-uz_w)/3;
    f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)] + rho_w*(-uz_w) / 6 - nzx;
    f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z,  9)] + rho_w*(-uz_w) / 6 + nzx;
    f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)] + rho_w*(-uz_w) / 6 - nzy;
    f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)] + rho_w*(-uz_w) / 6 + nzy;
}


__device__
void gpuBCPresZouHeB(dfloat* fPostStream, dfloat* fPostCol, const short unsigned int x, const short unsigned int y,
    const short unsigned int z, const dfloat rho_w)
{
    dfloat* f = fPostStream;
    const dfloat uz_w = 1 - (1 / rho_w) * (f[idxPop(x, y, z, 1)]  + f[idxPop(x, y, z, 2)]
        + f[idxPop(x, y, z, 3)]  + f[idxPop(x, y, z, 4)]  + f[idxPop(x, y, z, 7)]
        + f[idxPop(x, y, z, 14)] + f[idxPop(x, y, z, 8)]  + f[idxPop(x, y, z, 13)]
        + f[idxPop(x, y, z, 0)]  +2*(f[idxPop(x, y, z, 6)] + f[idxPop(x, y, z, 10)]
        + f[idxPop(x, y, z, 12)] + f[idxPop(x, y, z, 15)] + f[idxPop(x, y, z, 17)]));
    
    const dfloat nzx = 0.5 * (f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 7)] + f[idxPop(x, y, z, 13)]
        - (f[idxPop(x, y, z, 2)] + f[idxPop(x, y, z, 14)] + f[idxPop(x, y, z, 8)]));
        // - rho_w*ux_w/3;
    
    const dfloat nzy = 0.5 * (f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 7)] + f[idxPop(x, y, z, 14)]
        - (f[idxPop(x, y, z, 4)] + f[idxPop(x, y, z, 13)] + f[idxPop(x, y, z, 8)]));
        // - rho_w*uy_w/3

    f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)] + rho_w*uz_w/3;
    f[idxPop(x, y, z,  9)] = f[idxPop(x, y, z, 10)] + rho_w*uz_w / 6 - nzx;
    f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)] + rho_w*uz_w / 6 + nzx;
    f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)] + rho_w*uz_w / 6 - nzy;
    f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)] + rho_w*uz_w / 6 + nzy;
}

#endif //!D3Q19
#endif //COMP_PRES_ZOU_HE || COMP_ALL_BC