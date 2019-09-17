#include "lbm.h"

__global__ 
void gpuBCMacrCollisionStream(
    dfloat* const pop,
    dfloat* const popAux,
    NodeTypeMap* const mapBC,
    Macroscopics* const macr,
    bool const save,
    int const step)
{
    const short unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const short unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const short unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    // Apply boundary conditions
    if(mapBC[idxScalar(x, y, z)].getSchemeBC() != BC_NULL)
        gpuBoundaryConditions(&(mapBC[idxScalar(x, y, z)]), pop, x, y, z);

    // Adjacent coordinates
    const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;

    // Load populations
    dfloat fAux[Q];

#pragma unroll
    for (char i = 0; i < Q; i++)
        fAux[i] = pop[idxPop(x, y, z, i)];

    // Calculate macroscopics
    // rho = sum(f[i])
    // ux = (sum(f[i]*cx[i])+0.5*FX) / rho
    // uy = (sum(f[i]*cy[i])+0.5*FY) / rho
    // uz = (sum(f[i]*cz[i])+0.5*FZ) / rho
    
    
#ifdef D3Q19
    const dfloat rhoVar = fAux[0] + fAux[1] + fAux[2] + fAux[3] + fAux[4] 
        + fAux[5] + fAux[6] + fAux[7] + fAux[8] + fAux[9] + fAux[10] 
        + fAux[11] + fAux[12] + fAux[13] + fAux[14] + fAux[15] + fAux[16] 
        + fAux[17] + fAux[18];
    const dfloat uxVar = ((fAux[1] + fAux[7] + fAux[9] + fAux[13] + fAux[15])
        - (fAux[2] + fAux[8] + fAux[10] + fAux[14] + fAux[16]) + 0.5*FX) / rhoVar;
    const dfloat uyVar = ((fAux[3] + fAux[7] + fAux[11] + fAux[14] + fAux[17])
        - (fAux[4] + fAux[8] + fAux[12] + fAux[13] + fAux[18]) + 0.5*FY) / rhoVar;
    const dfloat uzVar = ((fAux[5] + fAux[9] + fAux[11] + fAux[16] + fAux[18])
        - (fAux[6] + fAux[10] + fAux[12] + fAux[15] + fAux[17]) + 0.5*FZ) / rhoVar;
#endif // !D3Q19
#ifdef D3Q27
    const dfloat rhoVar = fAux[0] + fAux[1] + fAux[2] + fAux[3] + fAux[4] 
        + fAux[5] + fAux[6] + fAux[7] + fAux[8] + fAux[9] + fAux[10] 
        + fAux[11] + fAux[12] + fAux[13] + fAux[14] + fAux[15] + fAux[16] 
        + fAux[17] + fAux[18] + fAux[19] + fAux[20] + fAux[21] + fAux[22]
        + fAux[23] + fAux[24] + fAux[25] + fAux[26];
    const dfloat uxVar = ((fAux[1] + fAux[7] + fAux[9] + fAux[13] + fAux[15]
        + fAux[19] + fAux[21] + fAux[23] + fAux[26]) 
        - (fAux[2] + fAux[8] + fAux[10] + fAux[14] + fAux[16] + fAux[20]
        + fAux[22] + fAux[24] + fAux[25]) + 0.5*FX) / rhoVar;
    const dfloat uyVar = ((fAux[3] + fAux[7] + fAux[11] + fAux[14] + fAux[17]
        + fAux[19] + fAux[21] + fAux[24] + fAux[25])
        - (fAux[4] + fAux[8] + fAux[12] + fAux[13] + fAux[18] + fAux[20]
        + fAux[22] + fAux[23] + fAux[26]) + 0.5*FY) / rhoVar;
    const dfloat uzVar = ((fAux[5] + fAux[9] + fAux[11] + fAux[16] + fAux[18]
        + fAux[19] + fAux[22] + fAux[23] + fAux[25])
        - (fAux[6] + fAux[10] + fAux[12] + fAux[15] + fAux[17] + fAux[20]
        + fAux[21] + fAux[24] + fAux[26]) + 0.5*FZ) / rhoVar;
#endif // !D3Q27

    if (save)
    {
        macr->rho[idxScalar(x, y, z)] = rhoVar;
        macr->ux[idxScalar(x, y, z)] = uxVar;
        macr->uy[idxScalar(x, y, z)] = uyVar;
        macr->uz[idxScalar(x, y, z)] = uzVar;
    }

    // Calculate temporary variables
    const dfloat p1_muu15 = 1 - 1.5 * (uxVar * uxVar + 
        uyVar * uyVar + uzVar * uzVar);
    const dfloat rhoW0 = rhoVar * W0;
    const dfloat rhoW1 = rhoVar * W1;
    const dfloat rhoW2 = rhoVar * W2;
    const dfloat W1t9d2 = W1 * 9 / 2;
    const dfloat W2t9d2 = W2 * 9 / 2;
#ifdef D3Q27
    const dfloat rhoW3 = rhoVar * W3;
    const dfloat W3t9d2 = W3 * 9 / 2;
#endif
    const dfloat ux3 = 3 * uxVar;
    const dfloat uy3 = 3 * uyVar;
    const dfloat uz3 = 3 * uzVar;
    
    // Calculate fneq
    // feq[i] = rho*w[i] * (1 - 1.5*u*u + 3*u*c[i] + 4.5*(u*c[i])^2) ->
    // fneq[i] = f[i]-feq[i]
    fAux[0] = fAux[0] - gpu_f_eq(rhoW0, 0, p1_muu15);
    fAux[1] = fAux[1] - gpu_f_eq(rhoW1,  ux3, p1_muu15);
    fAux[2] = fAux[2] - gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fAux[3] = fAux[3] - gpu_f_eq(rhoW1,  uy3, p1_muu15);
    fAux[4] = fAux[4] - gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fAux[5] = fAux[5] - gpu_f_eq(rhoW1,  uz3, p1_muu15);
    fAux[6] = fAux[6] - gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fAux[7] = fAux[7] - gpu_f_eq(rhoW2,  ux3 + uy3, p1_muu15);
    fAux[8] = fAux[8] - gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fAux[9] = fAux[9] - gpu_f_eq(rhoW2,  ux3 + uz3, p1_muu15);
    fAux[10] = fAux[10] - gpu_f_eq(rhoW2, -ux3 - uz3, p1_muu15);
    fAux[11] = fAux[11] - gpu_f_eq(rhoW2,  uy3 + uz3, p1_muu15);
    fAux[12] = fAux[12] - gpu_f_eq(rhoW2, -uy3 - uz3, p1_muu15);
    fAux[13] = fAux[13] - gpu_f_eq(rhoW2,  ux3 - uy3, p1_muu15);
    fAux[14] = fAux[14] - gpu_f_eq(rhoW2, -ux3 + uy3, p1_muu15);
    fAux[15] = fAux[15] - gpu_f_eq(rhoW2,  ux3 - uz3, p1_muu15);
    fAux[16] = fAux[16] - gpu_f_eq(rhoW2, -ux3 + uz3, p1_muu15);
    fAux[17] = fAux[17] - gpu_f_eq(rhoW2,  uy3 - uz3, p1_muu15);
    fAux[18] = fAux[18] - gpu_f_eq(rhoW2, -uy3 + uz3, p1_muu15);
#ifdef D3Q27
    fAux[19] = fAux[19] - gpu_f_eq(rhoW3,  ux3 + uy3 + uz3, p1_muu15);
    fAux[20] = fAux[20] - gpu_f_eq(rhoW3, -ux3 - uy3 - uz3, p1_muu15);
    fAux[21] = fAux[21] - gpu_f_eq(rhoW3,  ux3 + uy3 - uz3, p1_muu15);
    fAux[22] = fAux[22] - gpu_f_eq(rhoW3, -ux3 - uy3 + uz3, p1_muu15);
    fAux[23] = fAux[23] - gpu_f_eq(rhoW3,  ux3 - uy3 + uz3, p1_muu15);
    fAux[24] = fAux[24] - gpu_f_eq(rhoW3, -ux3 + uy3 - uz3, p1_muu15);
    fAux[25] = fAux[25] - gpu_f_eq(rhoW3, -ux3 + uy3 + uz3, p1_muu15);
    fAux[26] = fAux[26] - gpu_f_eq(rhoW3,  ux3 - uy3 - uz3, p1_muu15);
#endif

    // Calculate pineq(alfa, beta)/3
#ifdef D3Q19
    const dfloat pineqXX = (fAux[1] + fAux[2] + fAux[7] + fAux[8] + fAux[9] 
            + fAux[10] + fAux[13] + fAux[14] + fAux[15] + fAux[16]);
    const dfloat pineqYY = (fAux[3] + fAux[4] + fAux[7] + fAux[8] + fAux[11]
            + fAux[12] + fAux[13] + fAux[14] + fAux[17] + fAux[18]);
    const dfloat pineqZZ = (fAux[5] + fAux[6] + fAux[9] + fAux[10] + fAux[11]
            + fAux[12] + fAux[15] + fAux[16] + fAux[17] + fAux[18]);
    const dfloat pineqXYt2 = (fAux[7] + fAux[8] - fAux[13] - fAux[14]) * 2;
    const dfloat pineqXZt2 = (fAux[9] + fAux[10] - fAux[15] - fAux[16]) * 2;
    const dfloat pineqYZt2 = (fAux[11] + fAux[12] - fAux[17] - fAux[18]) * 2;
#endif // !D3Q19 
#ifdef D3Q27
    const dfloat aux = fneq[19] + fneq[20] + fneq[21] + fneq[22] + fneq[23]
            + fneq[24] + fneq[25] + fneq[26];
    const dfloat pineqXXd3 = (fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] 
            + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16] + aux) / 3;
    const dfloat pineqYYd3 = (fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11]
            + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18] + aux) / 3;
    const dfloat pineqZZd3 = (fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11]
            + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18] + aux) / 3;
    const dfloat pineqXYt2 = (fneq[7] + fneq[8] - fneq[13] - fneq[14] + fneq[19]
            + fneq[20] + fneq[21] + fneq[22] - fneq[23] - fneq[24] - fneq[25]
            - fneq[26]) * 2;
    const dfloat pineqXZt2 = (fneq[9] + fneq[10] - fneq[15] - fneq[16] + fneq[19]
            + fneq[20] - fneq[21] - fneq[22] + fneq[23] + fneq[24] - fneq[25]
            - fneq[26]) * 2;
    const dfloat pineqYZt2 = (fneq[11] + fneq[12] - fneq[17] - fneq[18] + fneq[19]
            + fneq[20] - fneq[21] - fneq[22] - fneq[23] - fneq[24] + fneq[25]
            + fneq[26]) * 2;
#endif // !D3Q27

    // Calculate regularized population
    // fReg[i] = 4.5*w[i](Q[i, alfa, beta]*pi[i, alfa, beta] 
    //          - c[i, alfa]*F[alfa]/3)
    // Obs.: fAux is used as fReg
    dfloat regTerms; // Q[i, alfa, beta]*pi[i, alfa, beta] - c[i, alfa]*F[alfa]/3)
    
    regTerms = -pineqXX/3 - pineqYY/3 - pineqZZ/3;
    fAux[0] = 4.5*W0*regTerms;

    regTerms +=  FX_D3 + pineqXX;
    fAux[1] = W1t9d2*regTerms;

    regTerms += -2*FX_D3;
    fAux[2] = W1t9d2*regTerms;

    regTerms +=  FX_D3 + FY_D3 - pineqXX + pineqYY;
    fAux[3] = W1t9d2*regTerms;

    regTerms += -2*FY_D3;
    fAux[4] = W1t9d2*regTerms;

    regTerms +=  FY_D3 + FZ_D3 - pineqYY + pineqZZ;
    fAux[5] = W1t9d2*regTerms;

    regTerms += -2*FZ_D3;
    fAux[6] = W1t9d2*regTerms;
    
    regTerms +=  FX_D3 + FY_D3 + FZ_D3 + pineqXX + pineqXYt2 + pineqYY - pineqZZ;
    fAux[7] = W2t9d2*regTerms;

    regTerms += -2*FX_D3 - 2*FY_D3;
    fAux[8] = W2t9d2*regTerms;

    regTerms +=  2*FX_D3 + FY_D3 + FZ_D3 - pineqXYt2 + pineqXZt2 - pineqYY + pineqZZ;
    fAux[9] = W2t9d2*regTerms;

    regTerms += -2*FX_D3 - 2*FZ_D3;
    fAux[10] = W2t9d2*regTerms;

    regTerms +=  FX_D3 + FY_D3 + 2*FZ_D3 - pineqXX - pineqXZt2 + pineqYY + pineqYZt2;
    fAux[11] = W2t9d2*regTerms;

    regTerms += -2*FY_D3 - 2*FZ_D3;
    fAux[12] = W2t9d2*regTerms;

    regTerms +=  FX_D3 + FZ_D3 + pineqXX - pineqXYt2 - pineqYZt2 - pineqZZ;
    fAux[13] = W2t9d2*regTerms;

    regTerms += -2*FX_D3 + 2*FY_D3;
    fAux[14] = W2t9d2*regTerms;

    regTerms +=  2*FX_D3 - FY_D3 - FZ_D3 + pineqXYt2 - pineqXZt2 - pineqYY + pineqZZ;
    fAux[15] = W2t9d2*regTerms;

    regTerms += -2*FX_D3 + 2*FZ_D3;
    fAux[16] = W2t9d2*regTerms;

    regTerms +=  FX_D3 + FY_D3 - 2*FZ_D3 - pineqXX + pineqXZt2 + pineqYY - pineqYZt2;
    fAux[17] = W2t9d2*regTerms;

    regTerms += -2*FY_D3 + 2*FZ_D3;
    fAux[18] = W2t9d2*regTerms;

#ifdef D3Q27
    fAux[19] = W3t9d2*( FX_D3 + FY_D3 + FZ_D3 + 2*pineqXXd3 + pineqXYt2 + pineqXZt2 + 2*pineqYYd3 + pineqYZt2 + 2*pineqZZd3);
    fAux[20] = W3t9d2*(-FX_D3 - FY_D3 - FZ_D3 + 2*pineqXXd3 + pineqXYt2 + pineqXZt2 + 2*pineqYYd3 + pineqYZt2 + 2*pineqZZd3);
    fAux[21] = W3t9d2*( FX_D3 + FY_D3 - FZ_D3 + 2*pineqXXd3 + pineqXYt2 - pineqXZt2 + 2*pineqYYd3 - pineqYZt2 + 2*pineqZZd3);
    fAux[22] = W3t9d2*(-FX_D3 - FY_D3 + FZ_D3 + 2*pineqXXd3 + pineqXYt2 - pineqXZt2 + 2*pineqYYd3 - pineqYZt2 + 2*pineqZZd3);
    fAux[23] = W3t9d2*( FX_D3 - FY_D3 + FZ_D3 + 2*pineqXXd3 - pineqXYt2 + pineqXZt2 + 2*pineqYYd3 - pineqYZt2 + 2*pineqZZd3);
    fAux[24] = W3t9d2*(-FX_D3 + FY_D3 - FZ_D3 + 2*pineqXXd3 - pineqXYt2 + pineqXZt2 + 2*pineqYYd3 - pineqYZt2 + 2*pineqZZd3);
    fAux[25] = W3t9d2*(-FX_D3 + FY_D3 + FZ_D3 + 2*pineqXXd3 - pineqXYt2 - pineqXZt2 + 2*pineqYYd3 + pineqYZt2 + 2*pineqZZd3);
    fAux[26] = W3t9d2*( FX_D3 - FY_D3 - FZ_D3 + 2*pineqXXd3 - pineqXYt2 - pineqXZt2 + 2*pineqYYd3 + pineqYZt2 + 2*pineqZZd3);
#endif

    // Collision to fAux
    // fAux = (1 - 1/TAU)*f1 + f_eq + (1 - 0.5/TAU)*force ->
    // fAux = (1 - OMEGA)*f1 + f_eq + (1 - 0.5*0MEGA)*force->
    // fAux = T_OMEGA * f1 + f_eq + TT_OMEGA*force
    // Force term is:
    // Q[i, alfa, beta] = c[i, alfa]*c[i, beta] - d_kronecker[alfa, beta]/3
    // force[i] = w[i]*(3*c[i, alfa]+9*Q[i, alfa, beta]*u[beta])*F[alfa]

    fAux[ 0] = T_OMEGA * fAux[ 0] + gpu_f_eq(rhoW0, 0, p1_muu15)
               + TT_OMEGA * gpu_force_term(W0,-ux3,-uy3,-uz3);

    fAux[ 1] = T_OMEGA * fAux[ 1] + gpu_f_eq(rhoW1,  ux3, p1_muu15) 
               + TT_OMEGA * gpu_force_term(W1, ux3*2+3,-uy3,-uz3);

    fAux[ 2] = T_OMEGA * fAux[ 2] + gpu_f_eq(rhoW1, -ux3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W1, ux3*2-3,-uy3,-uz3);

    fAux[ 3] = T_OMEGA * fAux[ 3] + gpu_f_eq(rhoW1,  uy3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W1,-ux3, uy3*2+3,-uz3);

    fAux[ 4] = T_OMEGA * fAux[ 4] + gpu_f_eq(rhoW1, -uy3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W1,-ux3, uy3*2-3,-uz3);

    fAux[ 5] = T_OMEGA * fAux[ 5] + gpu_f_eq(rhoW1,  uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W1,-ux3,-uy3, uz3*2+3);

    fAux[ 6] = T_OMEGA * fAux[ 6] + gpu_f_eq(rhoW1, -uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W1,-ux3,-uy3, uz3*2-3);

    fAux[ 7] = T_OMEGA * fAux[ 7] + gpu_f_eq(rhoW2,  ux3 + uy3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2+uy3*3+3, ux3*3+uy3*2+3,-uz3);

    fAux[ 8] = T_OMEGA * fAux[ 8] + gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2+uy3*3-3, ux3*3+uy3*2-3,-uz3);

    fAux[ 9] = T_OMEGA * fAux[ 9] + gpu_f_eq(rhoW2,  ux3 + uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2+uz3*3+3,-uy3, ux3*3+uz3*2+3);

    fAux[10] = T_OMEGA * fAux[10] + gpu_f_eq(rhoW2, -ux3 - uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2+uz3*3-3,-uy3, ux3*3+uz3*2-3);

    fAux[11] = T_OMEGA * fAux[11] + gpu_f_eq(rhoW2,  uy3 + uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2,-ux3, uy3*2+uz3*3+3, uy3*3+uz3*2+3);

    fAux[12] = T_OMEGA * fAux[12] + gpu_f_eq(rhoW2, -uy3 - uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2,-ux3, uy3*2+uz3*3-3, uy3*3+uz3*2-3);

    fAux[13] = T_OMEGA * fAux[13] + gpu_f_eq(rhoW2,  ux3 - uy3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2-uy3*3+3,-ux3*3+uy3*2-3,-uz3);

    fAux[14] = T_OMEGA * fAux[14] + gpu_f_eq(rhoW2, -ux3 + uy3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2-uy3*3-3,-ux3*3+uy3*2+3,-uz3);

    fAux[15] = T_OMEGA * fAux[15] + gpu_f_eq(rhoW2,  ux3 - uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2-uz3*3+3,-uy3,-ux3*3+uz3*2-3);

    fAux[16] = T_OMEGA * fAux[16] + gpu_f_eq(rhoW2, -ux3 + uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2, ux3*2-uz3*3-3,-uy3,-ux3*3+uz3*2+3);

    fAux[17] = T_OMEGA * fAux[17] + gpu_f_eq(rhoW2,  uy3 - uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2,-ux3, uy3*2-uz3*3+3,-uy3*3+uz3*2-3);

    fAux[18] = T_OMEGA * fAux[18] + gpu_f_eq(rhoW2, -uy3 + uz3, p1_muu15)
               + TT_OMEGA * gpu_force_term(W2,-ux3, uy3*2-uz3*3-3,-uy3*3+uz3*2+3);
#ifdef D3Q27
    fAux[19] = T_OMEGA * fAux[19] + feq[19] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2+uy3*3+uz3*3+3, ux3*3+uy3*2+uz3*3+3, ux3*3+uy3*3+uz3*2+3);
    
    fAux[20] = T_OMEGA * fAux[20] + feq[20] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2+uy3*3+uz3*3-3, ux3*3+uy3*2+uz3*3-3, ux3*3+uy3*3+uz3*2-3); 
    
    fAux[21] = T_OMEGA * fAux[21] + feq[21] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2+uy3*3-uz3*3+3, ux3*3+uy3*2-uz3*3+3,-ux3*3-uy3*3+uz3*2-3); 
    
    fAux[22] = T_OMEGA * fAux[22] + feq[22] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2+uy3*3-uz3*3-3, ux3*3+uy3*2-uz3*3-3,-ux3*3-uy3*3+uz3*2+3);
    
    fAux[23] = T_OMEGA * fAux[23] + feq[23] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2-uy3*3+uz3*3+3,-ux3*3+uy3*2-uz3*3-3, ux3*3-uy3*3+uz3*2+3);
    
    fAux[24] = T_OMEGA * fAux[24] + feq[24] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2-uy3*3+uz3*3-3,-ux3*3+uy3*2-uz3*3+3, ux3*3-uy3*3+uz3*2-3);
    
    fAux[25] = T_OMEGA * fAux[25] + feq[25] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2-uy3*3-uz3*3-3,-ux3*3+uy3*2+uz3*3+3,-ux3*3+uy3*3+uz3*2+3);
    
    fAux[26] = T_OMEGA * fAux[26] + feq[26] + 
               TT_OMEGA * gpu_force_term(W3, ux3*2-uy3*3-uz3*3+3,-ux3*3+uy3*2+uz3*3-3,-ux3*3+uy3*3+uz3*2-3);
#endif

    // Streaming to popAux
    // popAux(x+cx, y+cy, z+cz, i) = pop(x, y, z, i) 
    // The populations that shoudn't be streamed will be changed by the boundary conditions

    popAux[idxPop(x, y, z, 0)] = fAux[0];
    popAux[idxPop(xp1, y, z, 1)] = fAux[1];
    popAux[idxPop(xm1, y, z, 2)] = fAux[2];
    popAux[idxPop(x, yp1, z, 3)] = fAux[3];
    popAux[idxPop(x, ym1, z, 4)] = fAux[4];
    popAux[idxPop(x, y, zp1, 5)] = fAux[5];
    popAux[idxPop(x, y, zm1, 6)] = fAux[6];
    popAux[idxPop(xp1, yp1, z, 7)] = fAux[7];
    popAux[idxPop(xm1, ym1, z, 8)] = fAux[8];
    popAux[idxPop(xp1, y, zp1, 9)] = fAux[9];
    popAux[idxPop(xm1, y, zm1, 10)] = fAux[10];
    popAux[idxPop(x, yp1, zp1, 11)] = fAux[11];
    popAux[idxPop(x, ym1, zm1, 12)] = fAux[12];
    popAux[idxPop(xp1, ym1, z, 13)] = fAux[13];
    popAux[idxPop(xm1, yp1, z, 14)] = fAux[14];
    popAux[idxPop(xp1, y, zm1, 15)] = fAux[15];
    popAux[idxPop(xm1, y, zp1, 16)] = fAux[16];
    popAux[idxPop(x, yp1, zm1, 17)] = fAux[17];
    popAux[idxPop(x, ym1, zp1, 18)] = fAux[18];
#ifdef D3Q27
    popAux[idxPop(xp1, yp1, zp1, 19)] = fAux[19];
    popAux[idxPop(xm1, ym1, zm1, 20)] = fAux[20];
    popAux[idxPop(xp1, yp1, zm1, 21)] = fAux[21];
    popAux[idxPop(xm1, ym1, zp1, 22)] = fAux[22];
    popAux[idxPop(xp1, ym1, zp1, 23)] = fAux[23];
    popAux[idxPop(xm1, yp1, zm1, 24)] = fAux[24];
    popAux[idxPop(xm1, yp1, zp1, 25)] = fAux[25];
    popAux[idxPop(xp1, ym1, zm1, 26)] = fAux[26];
#endif
}


__global__
void gpuUpdateMacr(
    Populations* pop,
    Macroscopics* macr)
{
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    // load populations
    dfloat fAux[Q];
    for (unsigned char i = 0; i < Q; i++)
        fAux[i] = pop->pop[idxPop(x, y, z, i)];

    // calc for macroscopics
    // rho = sum(f[i])
    // ux = sum(f[i]*cx[i]) / rho
    // uy = sum(f[i]*cy[i]) / rho
    // uz = sum(f[i]*cz[i]) / rho
#ifdef D3Q19
    const dfloat rhoVar = fAux[0] + fAux[1] + fAux[2] + fAux[3] + fAux[4] + fAux[5] + fAux[6]
        + fAux[7] + fAux[8] + fAux[9] + fAux[10] + fAux[11] + fAux[12] + fAux[13] + fAux[14]
        + fAux[15] + fAux[16] + fAux[17] + fAux[18];
    const dfloat uxVar = ((fAux[1] + fAux[7] + fAux[9] + fAux[13] + fAux[15])
        - (fAux[2] + fAux[8] + fAux[10] + fAux[14] + fAux[16])) / rhoVar;
    const dfloat uyVar = ((fAux[3] + fAux[7] + fAux[11] + fAux[14] + fAux[17])
        - (fAux[4] + fAux[8] + fAux[12] + fAux[13] + fAux[18])) / rhoVar;
    const dfloat uzVar = ((fAux[5] + fAux[9] + fAux[11] + fAux[16] + fAux[18])
        - (fAux[6] + fAux[10] + fAux[12] + fAux[15] + fAux[17])) / rhoVar;
#endif // !D3Q19
#ifdef D3Q27
    const dfloat rhoVar = fAux[0] + fAux[1] + fAux[2] + fAux[3] + fAux[4] 
        + fAux[5] + fAux[6] + fAux[7] + fAux[8] + fAux[9] + fAux[10] 
        + fAux[11] + fAux[12] + fAux[13] + fAux[14] + fAux[15] + fAux[16] 
        + fAux[17] + fAux[18] + fAux[19] + fAux[20] + fAux[21] + fAux[22]
        + fAux[23] + fAux[24] + fAux[25] + fAux[26];
    const dfloat uxVar = ((fAux[1] + fAux[7] + fAux[9] + fAux[13] + fAux[15]
        + fAux[19] + fAux[21] + fAux[23] + fAux[26]) 
        - (fAux[2] + fAux[8] + fAux[10] + fAux[14] + fAux[16] + fAux[20]
        + fAux[22] + fAux[24] + fAux[25]) + 0.5*FX) / rhoVar;
    const dfloat uyVar = ((fAux[3] + fAux[7] + fAux[11] + fAux[14] + fAux[17]
        + fAux[19] + fAux[21] + fAux[24] + fAux[25])
        - (fAux[4] + fAux[8] + fAux[12] + fAux[13] + fAux[18] + fAux[20]
        + fAux[22] + fAux[23] + fAux[26]) + 0.5*FY) / rhoVar;
    const dfloat uzVar = ((fAux[5] + fAux[9] + fAux[11] + fAux[16] + fAux[18]
        + fAux[19] + fAux[22] + fAux[23] + fAux[25])
        - (fAux[6] + fAux[10] + fAux[12] + fAux[15] + fAux[17] + fAux[20]
        + fAux[21] + fAux[24] + fAux[26]) + 0.5*FZ) / rhoVar;
#endif // !D3Q27
    macr->rho[idxScalar(x, y, z)] = rhoVar;
    macr->ux[idxScalar(x, y, z)] = uxVar;
    macr->uy[idxScalar(x, y, z)] = uyVar;
    macr->uz[idxScalar(x, y, z)] = uzVar;
}
