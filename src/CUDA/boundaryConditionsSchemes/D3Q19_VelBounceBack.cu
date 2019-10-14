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

#include "D3Q19_VelBounceBack.h"


__device__
void gpuBCVelBounceBackN(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    // uses node's rho as the wall's rho
    const dfloat rho_w = fNode[(0)] + fNode[(1)] + fNode[(2)] +
        fNode[(3)] + fNode[(4)] + fNode[(5)] + fNode[(6)] +
        fNode[(7)] + fNode[(8)] + fNode[(9)] + fNode[(10)] +
        fNode[(11)] + fNode[(12)] + fNode[(13)] + fNode[(14)] +
        fNode[(15)] + fNode[(16)] + fNode[(17)] + fNode[(18)];

    fNode[(4)] = fNode[(3)] - 6 * rho_w*W1*(uy_w);
    fNode[(8)] = fNode[(7)] - 6 * rho_w*W2*(uy_w + ux_w);
    fNode[(12)] = fNode[(11)] - 6 * rho_w*W2*(uy_w + uz_w);
    fNode[(13)] = fNode[(14)] - 6 * rho_w*W2*(uy_w - ux_w);
    fNode[(18)] = fNode[(17)] - 6 * rho_w*W2*(uy_w - uz_w);


}


__device__
void gpuBCVelBounceBackS(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    // uses node's rho as the wall's rho
    const dfloat rho_w = fNode[(0)] + fNode[(1)] + fNode[(2)] +
        fNode[(3)] + fNode[(4)] + fNode[(5)] + fNode[(6)] +
        fNode[(7)] + fNode[(8)] + fNode[(9)] + fNode[(10)] +
        fNode[(11)] + fNode[(12)] + fNode[(13)] + fNode[(14)] +
        fNode[(15)] + fNode[(16)] + fNode[(17)] + fNode[(18)];

    fNode[(3)] = fNode[(4)] - 6 * rho_w*W1*(-uy_w);
    fNode[(7)] = fNode[(8)] - 6 * rho_w*W2*(-uy_w - ux_w);
    fNode[(11)] = fNode[(12)] - 6 * rho_w*W2*(-uy_w - uz_w);
    fNode[(14)] = fNode[(13)] - 6 * rho_w*W2*(-uy_w + ux_w);
    fNode[(17)] = fNode[(18)] - 6 * rho_w*W2*(-uy_w + uz_w);
}


__device__
void gpuBCVelBounceBackW(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    // uses node's rho as the wall's rho
    const dfloat rho_w = fNode[(0)] + fNode[(1)] + fNode[(2)] +
        fNode[(3)] + fNode[(4)] + fNode[(5)] + fNode[(6)] +
        fNode[(7)] + fNode[(8)] + fNode[(9)] + fNode[(10)] +
        fNode[(11)] + fNode[(12)] + fNode[(13)] + fNode[(14)] +
        fNode[(15)] + fNode[(16)] + fNode[(17)] + fNode[(18)];
    
    fNode[(1)] = fNode[(2)] - 6 * rho_w*W1*(-ux_w);
    fNode[(7)] = fNode[(8)] - 6 * rho_w*W2*(-ux_w - uy_w);
    fNode[(9)] = fNode[(10)] - 6 * rho_w*W2*(-ux_w - uz_w);
    fNode[(13)] = fNode[(14)] - 6 * rho_w*W2*(-ux_w + uy_w);
    fNode[(15)] = fNode[(16)] - 6 * rho_w*W2*(-ux_w + uz_w);
}


__device__
void gpuBCVelBounceBackE(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    // uses node's rho as the wall's rho
    const dfloat rho_w = fNode[(0)] + fNode[(1)] + fNode[(2)] +
        fNode[(3)] + fNode[(4)] + fNode[(5)] + fNode[(6)] +
        fNode[(7)] + fNode[(8)] + fNode[(9)] + fNode[(10)] +
        fNode[(11)] + fNode[(12)] + fNode[(13)] + fNode[(14)] +
        fNode[(15)] + fNode[(16)] + fNode[(17)] + fNode[(18)];

    fNode[(2)] = fNode[(1)] - 6 * rho_w*W1*(ux_w);
    fNode[(8)] = fNode[(7)] - 6 * rho_w*W2*(ux_w + uy_w);
    fNode[(10)] = fNode[(9)] - 6 * rho_w*W2*(ux_w + uz_w);
    fNode[(14)] = fNode[(13)] - 6 * rho_w*W2*(ux_w - uy_w);
    fNode[(16)] = fNode[(15)] - 6 * rho_w*W2*(ux_w - uz_w);
}


__device__
void gpuBCVelBounceBackF(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    // uses node's rho as the wall's rho
    const dfloat rho_w = fNode[(0)] + fNode[(1)] + fNode[(2)] +
        fNode[(3)] + fNode[(4)] + fNode[(5)] + fNode[(6)] +
        fNode[(7)] + fNode[(8)] + fNode[(9)] + fNode[(10)] +
        fNode[(11)] + fNode[(12)] + fNode[(13)] + fNode[(14)] +
        fNode[(15)] + fNode[(16)] + fNode[(17)] + fNode[(18)];

    fNode[(6)] = fNode[(5)] - 6 * rho_w*W1*(uz_w);
    fNode[(10)] = fNode[(9)] - 6 * rho_w*W2*(uz_w + ux_w);
    fNode[(12)] = fNode[(11)] - 6 * rho_w*W2*(uz_w + uy_w);
    fNode[(15)] = fNode[(16)] - 6 * rho_w*W2*(uz_w - ux_w);
    fNode[(17)] = fNode[(18)] - 6 * rho_w*W2*(uz_w - uy_w);

}


__device__
void gpuBCVelBounceBackB(dfloat* fNode, const dfloat ux_w, const dfloat uy_w, const dfloat uz_w)
{
    // uses node's rho as the wall's rho
    const dfloat rho_w = fNode[(0)] + fNode[(1)] + fNode[(2)] +
        fNode[(3)] + fNode[(4)] + fNode[(5)] + fNode[(6)] +
        fNode[(7)] + fNode[(8)] + fNode[(9)] + fNode[(10)] +
        fNode[(11)] + fNode[(12)] + fNode[(13)] + fNode[(14)] +
        fNode[(15)] + fNode[(16)] + fNode[(17)] + fNode[(18)];

    fNode[(5)] = fNode[(6)] - 6 * rho_w*W1*(-uz_w);
    fNode[(9)] = fNode[(10)] - 6 * rho_w*W2*(-uz_w - ux_w);
    fNode[(11)] = fNode[(12)] - 6 * rho_w*W2*(-uz_w - uy_w);
    fNode[(16)] = fNode[(15)] - 6 * rho_w*W2*(-uz_w + ux_w);
    fNode[(18)] = fNode[(17)] - 6 * rho_w*W2*(-uz_w + uy_w);
}
