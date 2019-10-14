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

#include "D3Q19_VelZouHe.h"


__device__
void gpuBCVelZouHeN(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    const dfloat rho_w = (1 / (1 + uy_w)) * (fNode[(1)] + fNode[(2)]
        + fNode[(5)] + fNode[(6)] + fNode[(9)]
        + fNode[(15)] + fNode[(16)] + fNode[(10)]
        + fNode[(0)] + 2 * (fNode[(3)] + fNode[(7)] 
        + fNode[(14)] + fNode[(11)] + fNode[(17)]));

    const dfloat nyx = 0.5 * (fNode[(1)] + fNode[(9)] + fNode[(15)]
        - (fNode[(2)] + fNode[(16)] + fNode[(10)])) - ux_w * rho_w / 3;
    const dfloat nyz = 0.5 * (fNode[(5)] + fNode[(9)] + fNode[(16)]
        - (fNode[(6)] + fNode[(15)] + fNode[(10)])) - uz_w * rho_w / 3;

    fNode[(4)] = fNode[(3)] + rho_w * (-uy_w) / 3;
    fNode[(8)] = fNode[(7)] + rho_w * (-uy_w - ux_w) / 6 + nyx;
    fNode[(13)] = fNode[(14)] + rho_w * (-uy_w + ux_w) / 6 - nyx;
    fNode[(12)] = fNode[(11)] + rho_w * (-uy_w - uz_w) / 6 + nyz;
    fNode[(18)] = fNode[(17)] + rho_w * (-uy_w + uz_w) / 6 - nyz;
}


__device__
void gpuBCVelZouHeS(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    const dfloat rho_w = (1 / (1 - uy_w)) * (fNode[(1)] + fNode[(2)]
        + fNode[(5)] + fNode[(6)] + fNode[(9)]
        + fNode[(15)] + fNode[(16)] + fNode[(10)]
        + fNode[(0)] + 2 * (fNode[(4)] + fNode[(13)]
        + fNode[(8)] + fNode[(18)] + fNode[(12)]));

    const dfloat nyx = 0.5 * (fNode[(1)] + fNode[(9)] + fNode[(15)]
        - (fNode[(2)] + fNode[(16)] + fNode[(10)])) - ux_w * rho_w / 3;
    const dfloat nyz = 0.5 * (fNode[(5)] + fNode[(9)] + fNode[(16)]
        - (fNode[(6)] + fNode[(15)] + fNode[(10)])) - uz_w * rho_w / 3;
    
    fNode[(3)] = fNode[(4)] + rho_w * uy_w / 3;
    fNode[(7)] = fNode[(8)] + rho_w * (uy_w + ux_w) / 6 - nyx;
    fNode[(14)] = fNode[(13)] + rho_w * (uy_w - ux_w) / 6 + nyx;
    fNode[(11)] = fNode[(12)] + rho_w * (uy_w + uz_w) / 6 - nyz;
    fNode[(17)] = fNode[(18)] + rho_w * (uy_w - uz_w) / 6 + nyz;
}


__device__
void gpuBCVelZouHeW(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    const dfloat rho_w = (1 / (1 - ux_w)) * (fNode[(3)] + fNode[(4)]
        + fNode[(5)] + fNode[(6)] + fNode[(11)]
        + fNode[(17)] + fNode[(18)] + fNode[(12)]
        + fNode[(0)] + 2 * (fNode[(2)] + fNode[(14)]
        + fNode[(8)] + fNode[(16)] + fNode[(10)]));

    const dfloat nxy = 0.5 * (fNode[(3)] + fNode[(11)] + fNode[(17)]
        - (fNode[(4)] + fNode[(18)] + fNode[(12)])) - uy_w * rho_w / 3;
    const dfloat nxz = 0.5 * (fNode[(5)] + fNode[(18)] + fNode[(11)]
        - (fNode[(6)] + fNode[(17)] + fNode[(12)])) - uz_w * rho_w / 3;

    fNode[(1)] = fNode[(2)] + rho_w * ux_w / 3;
    fNode[(13)] = fNode[(14)] + rho_w * (ux_w - uy_w) / 6 + nxy;
    fNode[(7)] = fNode[(8)] + rho_w * (ux_w + uy_w) / 6 - nxy;
    fNode[(9)] = fNode[(10)] + rho_w * (ux_w + uz_w) / 6 - nxz;
    fNode[(15)] = fNode[(16)] + rho_w * (ux_w - uy_w) / 6 + nxz;
}


__device__
void gpuBCVelZouHeE(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    const dfloat rho_w = (1 / (1 + ux_w)) * (fNode[(3)] + fNode[(4)]
        + fNode[(5)] + fNode[(6)] + fNode[(11)]
        + fNode[(17)] + fNode[(18)] + fNode[(12)]
        + fNode[(0)] + 2 * (fNode[(2)] + fNode[(14)]
        + fNode[(8)] + fNode[(16)] + fNode[(10)]));

    const dfloat nxy = 0.5 * (fNode[(3)] + fNode[(11)] + fNode[(17)]
        - (fNode[(4)] + fNode[(18)] + fNode[(12)])) - uy_w * rho_w / 3;
    const dfloat nxz = 0.5 * (fNode[(5)] + fNode[(18)] + fNode[(11)]
        - (fNode[(6)] + fNode[(17)] + fNode[(12)])) - uz_w * rho_w / 3;

    fNode[(2)] = fNode[(2)] + rho_w * ux_w / 3;
    fNode[(14)] = fNode[(14)] + rho_w * (ux_w - uy_w) / 6 + nxy;
    fNode[(8)] = fNode[(8)] + rho_w * (ux_w + uy_w) / 6 - nxy;
    fNode[(10)] = fNode[(10)] + rho_w * (ux_w + uz_w) / 6 - nxz;
    fNode[(16)] = fNode[(16)] + rho_w * (ux_w - uy_w) / 6 + nxz;
}


__device__
void gpuBCVelZouHeF(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    const dfloat rho_w = (1 / (1 + uz_w)) * (fNode[(1)] + fNode[(2)]
        + fNode[(3)] + fNode[(4)] + fNode[(7)]
        + fNode[(14)] + fNode[(8)] + fNode[(13)]
        + fNode[(0)] + 2 * (fNode[(5)] + fNode[(9)]
        + fNode[(16)] + fNode[(11)] + fNode[(18)]));

    const dfloat nzx = 0.5 * (fNode[(1)] + fNode[(7)] + fNode[(13)]
        - (fNode[(2)] + fNode[(14)] + fNode[(8)])) - rho_w * ux_w / 3;

    const dfloat nzy = 0.5 * (fNode[(3)] + fNode[(7)] + fNode[(14)]
        - (fNode[(4)] + fNode[(13)] + fNode[(8)])) - rho_w * uy_w / 3;

    fNode[(6)] = fNode[(5)] + rho_w * (-uz_w) / 3;
    fNode[(15)] = fNode[(16)] + rho_w * (-uz_w + ux_w) / 6 - nzx;
    fNode[(10)] = fNode[(9)] + rho_w * (-uz_w - ux_w) / 6 + nzx;
    fNode[(17)] = fNode[(18)] + rho_w * (-uz_w + uy_w) / 6 - nzy;
    fNode[(12)] = fNode[(11)] + rho_w * (-uz_w - uy_w) / 6 + nzy;

}


__device__
void gpuBCVelZouHeB(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    const dfloat rho_w = (1 / (1 + uz_w)) * (fNode[(1)] + fNode[(2)]
        + fNode[(3)] + fNode[(4)] + fNode[(7)]
        + fNode[(14)] + fNode[(8)] + fNode[(13)]
        + fNode[(0)] + 2 * (fNode[(6)] + fNode[(15)]
        + fNode[(10)] + fNode[(17)] + fNode[(12)]));

    const dfloat nzx = 0.5 * (fNode[(1)] + fNode[(7)] + fNode[(13)]
        - (fNode[(2)] + fNode[(14)] + fNode[(8)])) - rho_w * ux_w / 3;

    const dfloat nzy = 0.5 * (fNode[(3)] + fNode[(7)] + fNode[(14)]
        - (fNode[(4)] + fNode[(13)] + fNode[(8)])) - rho_w * uy_w / 3;

    fNode[(5)] = fNode[(6)] + rho_w * (uz_w) / 3;
    fNode[(9)] = fNode[(10)] + rho_w * (uz_w + ux_w) / 6 - nzx;
    fNode[(16)] = fNode[(15)] + rho_w * (uz_w - ux_w) / 6 + nzx;
    fNode[(11)] = fNode[(12)] + rho_w * (uz_w + ux_w) / 6 - nzy;
    fNode[(18)] = fNode[(17)] + rho_w * (uz_w - ux_w) / 6 + nzy;

}
