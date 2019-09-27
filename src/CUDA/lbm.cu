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
    // Terms to use to recursive calculations
#ifdef D3Q19
    dfloat terms[6];
#endif 
#ifdef D3Q27
    dfloat terms[Q]; // TODO
#endif
    dfloat multiplyTerm = 1;
    dfloat auxTerm = 0;
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
    const dfloat ux3ux3d2 = ux3*ux3/2;
    const dfloat ux3uy3 = ux3*uy3;
    const dfloat ux3uz3 = ux3*uz3;
    const dfloat uy3uy3d2 = uy3*uy3/2;
    const dfloat uy3uz3 = uy3*uz3;
    const dfloat uz3uz3d2 = uz3*uz3/2;
    
    // Calculate equilibrium terms (0.5*uc3^2 + uc3)
    // terms[0] -> population 0
    // terms[1] -> population 1
    // terms[2] -> population 2
    // terms[3] -> population 3
    // terms[4] -> population 4
    terms[0] = p1_muu15;
    terms[1] = terms[0] + ( ux3 + ux3ux3d2);
    terms[2] = terms[0] + (-ux3 + ux3ux3d2);
    terms[3] = terms[0] + ( uy3 + uy3uy3d2);
    terms[4] = terms[0] + (-uy3 + uy3uy3d2);
#ifdef D3Q27
    terms[19] = terms[7] + ( uz3 + ux3uz3 + uy3uz3 + uz3uz3d2);
    terms[20] = terms[8] + (-uz3 + ux3uz3 + uy3uz3 + uz3uz3d2);
    terms[21] = terms[7] + (-uz3 - ux3uz3 - uy3uz3 + uz3uz3d2);
    terms[22] = terms[8] + ( uz3 - ux3uz3 - uy3uz3 + uz3uz3d2);
    terms[23] = terms[9] + (-uy3 - ux3uy3 + uy3uy3d2 - uy3uz3);
    terms[24] = terms[10] + ( uy3 - ux3uy3 + uy3uy3d2 - uy3uz3);
    terms[25] = terms[11] + (-ux3 + ux3ux3d2 - ux3uy3 - ux3uz3);
    terms[26] = terms[12] + ( ux3 + ux3ux3d2 - ux3uy3 - ux3uz3);
#endif

    // Calculate fneq
    // feq[i] = rho*w[i] * (1 - 1.5*u*u + 3*u*c[i] + 4.5*(u*c[i])^2) ->
    // fneq[i] = f[i]-feq[i] -> fi[i] -= feq[i]
    multiplyTerm = rhoW0;
    fAux[0] -= multiplyTerm*terms[0];
    multiplyTerm = rhoW1;
    fAux[1] -= multiplyTerm*terms[1];
    fAux[2] -= multiplyTerm*terms[2];
    fAux[3] -= multiplyTerm*terms[3];
    fAux[4] -= multiplyTerm*terms[4];
    fAux[5] -= multiplyTerm*(terms[0] + ( uz3 + uz3uz3d2));
    fAux[6] -= multiplyTerm*(terms[0] + (-uz3 + uz3uz3d2));
    multiplyTerm = rhoW2;
    fAux[7] -=  multiplyTerm*(terms[1] + ( uy3 + ux3uy3 + uy3uy3d2));
    fAux[8] -=  multiplyTerm*(terms[2] + (-uy3 + ux3uy3 + uy3uy3d2));
    fAux[9] -=  multiplyTerm*(terms[1] + ( uz3 + ux3uz3 + uz3uz3d2));
    fAux[10] -= multiplyTerm*(terms[2] + (-uz3 + ux3uz3 + uz3uz3d2));
    fAux[11] -= multiplyTerm*(terms[3] + ( uz3 + uy3uz3 + uz3uz3d2));
    fAux[12] -= multiplyTerm*(terms[4] + (-uz3 + uy3uz3 + uz3uz3d2));
    fAux[13] -= multiplyTerm*(terms[1] + (-uy3 - ux3uy3 + uy3uy3d2));
    fAux[14] -= multiplyTerm*(terms[2] + ( uy3 - ux3uy3 + uy3uy3d2));
    fAux[15] -= multiplyTerm*(terms[1] + (-uz3 - ux3uz3 + uz3uz3d2));
    fAux[16] -= multiplyTerm*(terms[2] + ( uz3 - ux3uz3 + uz3uz3d2));
    fAux[17] -= multiplyTerm*(terms[3] + (-uz3 - uy3uz3 + uz3uz3d2));
    fAux[18] -= multiplyTerm*(terms[4] + ( uz3 - uy3uz3 + uz3uz3d2));
#ifdef D3Q27
    multiplyTerm = rhoW3;
    fAux[19] = fAux[19] - multiplyTerm*terms[19];
    fAux[20] = fAux[20] - multiplyTerm*terms[20];
    fAux[21] = fAux[21] - multiplyTerm*terms[21];
    fAux[22] = fAux[22] - multiplyTerm*terms[22];
    fAux[23] = fAux[23] - multiplyTerm*terms[23];
    fAux[24] = fAux[24] - multiplyTerm*terms[24];
    fAux[25] = fAux[25] - multiplyTerm*terms[25];
    fAux[26] = fAux[26] - multiplyTerm*terms[26];
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
    const dfloat aux = fneq[19] + faux[20] + faux[21] + faux[22] + faux[23]
            + faux[24] + faux[25] + faux[26];
    const dfloat pineqXXd3 = (faux[1] + faux[2] + faux[7] + faux[8] + faux[9] 
            + faux[10] + faux[13] + faux[14] + faux[15] + faux[16] + aux) / 3;
    const dfloat pineqYYd3 = (faux[3] + faux[4] + faux[7] + faux[8] + faux[11]
            + faux[12] + faux[13] + faux[14] + faux[17] + faux[18] + aux) / 3;
    const dfloat pineqZZd3 = (faux[5] + faux[6] + faux[9] + faux[10] + faux[11]
            + faux[12] + faux[15] + faux[16] + faux[17] + faux[18] + aux) / 3;
    const dfloat pineqXYt2 = (faux[7] + faux[8] - faux[13] - faux[14] + faux[19]
            + faux[20] + faux[21] + faux[22] - faux[23] - faux[24] - faux[25]
            - faux[26]) * 2;
    const dfloat pineqXZt2 = (faux[9] + faux[10] - faux[15] - faux[16] + faux[19]
            + faux[20] - faux[21] - faux[22] + faux[23] + faux[24] - faux[25]
            - faux[26]) * 2;
    const dfloat pineqYZt2 = (faux[11] + faux[12] - faux[17] - faux[18] + faux[19]
            + faux[20] - faux[21] - faux[22] - faux[23] - faux[24] + faux[25]
            + faux[26]) * 2;
#endif // !D3Q27

    // Calculate regularization terms 
    // Q[i, alfa, beta]*pi[i, alfa, beta] - c[i, alfa]*F[alfa]/3
    // terms[0] -> population 0
    // terms[1] -> population 1
    // terms[2] -> population 2
    // terms[3] -> population 3
    // terms[4] -> population 4
    terms[0] = -pineqXX/3 - pineqYY/3 - pineqZZ/3;
    terms[1] = terms[0] + (-FX_D3 + pineqXX);
    terms[2] = terms[0] + ( FX_D3 + pineqXX);
    terms[3] = terms[0] + (-FY_D3 + pineqYY);
    terms[4] = terms[0] + ( FY_D3 + pineqYY);
#ifdef D3Q27
    terms[7] = terms[1] + (-FY_D3 + pineqXYt2 + pineqYY);
    terms[8] = terms[2] + ( FY_D3 + pineqXYt2 + pineqYY);
    terms[9] = terms[1] + (-FZ_D3 + pineqXZt2 + pineqZZ);
    terms[10] = terms[2] + ( FZ_D3 + pineqXZt2 + pineqZZ);
    terms[11] = terms[3] + (-FZ_D3 + pineqYZt2 + pineqZZ);
    terms[12] = terms[4] + ( FZ_D3 + pineqYZt2 + pineqZZ);
#endif
    
    // Calculate regularized population
    // fReg[i] = 4.5*w[i](Q[i, alfa, beta]*pi[i, alfa, beta] 
    //          - c[i, alfa]*F[alfa]/3)
    // Obs.: fAux is used as fReg
    multiplyTerm = W0*4.5;
    fAux[0] = multiplyTerm*terms[0];
    multiplyTerm = W1t9d2;
    fAux[1] = multiplyTerm*terms[1];
    fAux[2] = multiplyTerm*terms[2];
    fAux[3] = multiplyTerm*terms[3];
    fAux[4] = multiplyTerm*terms[4];
    fAux[5] = multiplyTerm*(terms[0] + (-FZ_D3 + pineqZZ));
    fAux[6] = multiplyTerm*(terms[0] + ( FZ_D3 + pineqZZ));
    multiplyTerm = W2t9d2;
    fAux[7] = multiplyTerm*(terms[1] + (-FY_D3 + pineqXYt2 + pineqYY));
    fAux[8] = multiplyTerm*(terms[2] + ( FY_D3 + pineqXYt2 + pineqYY));
    fAux[9] = multiplyTerm*(terms[1] + (-FZ_D3 + pineqXZt2 + pineqZZ));
    fAux[10] = multiplyTerm*(terms[2] + ( FZ_D3 + pineqXZt2 + pineqZZ));
    fAux[11] = multiplyTerm*(terms[3] + (-FZ_D3 + pineqYZt2 + pineqZZ));
    fAux[12] = multiplyTerm*(terms[4] + ( FZ_D3 + pineqYZt2 + pineqZZ));
    fAux[13] = multiplyTerm*(terms[1] + ( FY_D3 - pineqXYt2 + pineqYY));
    fAux[14] = multiplyTerm*(terms[2] + (-FY_D3 - pineqXYt2 + pineqYY));
    fAux[15] = multiplyTerm*(terms[1] + ( FZ_D3 - pineqXZt2 + pineqZZ));
    fAux[16] = multiplyTerm*(terms[2] + (-FZ_D3 - pineqXZt2 + pineqZZ));
    fAux[17] = multiplyTerm*(terms[3] + ( FZ_D3 - pineqYZt2 + pineqZZ));
    fAux[18] = multiplyTerm*(terms[4] + (-FZ_D3 - pineqYZt2 + pineqZZ));
#ifdef D3Q27
    multiplyTerm = W3t9d2;
    fAux[19] = multiplyTerm*terms[19];
    fAux[20] = multiplyTerm*terms[20];
    fAux[21] = multiplyTerm*terms[21];
    fAux[22] = multiplyTerm*terms[22];
    fAux[23] = multiplyTerm*terms[23];
    fAux[24] = multiplyTerm*terms[24];
    fAux[25] = multiplyTerm*terms[25];
    fAux[26] = multiplyTerm*terms[26];
#endif

    
    // Collision to fAux
    // fAux = (1 - 1/TAU)*f1 + f_eq + (1 - 0.5/TAU)*force ->
    // fAux = (1 - OMEGA)*f1 + f_eq + (1 - 0.5*0MEGA)*force->
    // fAux = T_OMEGA * f1 + f_eq + TT_OMEGA*force
    // Force term is:
    // Q[i, alfa, beta] = c[i, alfa]*c[i, beta] - d_kronecker[alfa, beta]/3
    // force[i] = w[i]*(3*c[i, alfa]+9*Q[i, alfa, beta]*u[beta])*F[alfa]

#pragma unroll
    for(char i = 0; i < Q; i++)
        fAux[i] *= T_OMEGA;

    // Calculate equilibrium terms (0.5*uc3^2 + uc3)
    // terms[0] -> population 0
    // terms[1] -> population 1
    // terms[2] -> population 2
    // terms[3] -> population 3
    // terms[4] -> population 4
    terms[0] = p1_muu15;
    terms[1] = terms[0] + ( ux3 + ux3ux3d2);
    terms[2] = terms[0] + (-ux3 + ux3ux3d2);
    terms[3] = terms[0] + ( uy3 + uy3uy3d2);
    terms[4] = terms[0] + (-uy3 + uy3uy3d2);
#ifdef D3Q27
    terms[19] = terms[7] + ( uz3 + ux3uz3 + uy3uz3 + uz3uz3d2);
    terms[20] = terms[8] + (-uz3 + ux3uz3 + uy3uz3 + uz3uz3d2);
    terms[21] = terms[7] + (-uz3 - ux3uz3 - uy3uz3 + uz3uz3d2);
    terms[22] = terms[8] + ( uz3 - ux3uz3 - uy3uz3 + uz3uz3d2);
    terms[23] = terms[9] + (-uy3 - ux3uy3 + uy3uy3d2 - uy3uz3);
    terms[24] = terms[10] + ( uy3 - ux3uy3 + uy3uy3d2 - uy3uz3);
    terms[25] = terms[11] + (-ux3 + ux3ux3d2 - ux3uy3 - ux3uz3);
    terms[26] = terms[12] + ( ux3 + ux3ux3d2 - ux3uy3 - ux3uz3);
#endif

    // add equilibrium term
    multiplyTerm = rhoW0;
    fAux[0] += multiplyTerm*terms[0];
    multiplyTerm = rhoW1;
    fAux[1] += multiplyTerm*terms[1];
    fAux[2] += multiplyTerm*terms[2];
    fAux[3] += multiplyTerm*terms[3];
    fAux[4] += multiplyTerm*terms[4];
    fAux[5] += multiplyTerm*(terms[0] + ( uz3 + uz3uz3d2));
    fAux[6] += multiplyTerm*(terms[0] + (-uz3 + uz3uz3d2));
    multiplyTerm = rhoW2;
    fAux[7]  += multiplyTerm*(terms[1] + ( uy3 + ux3uy3 + uy3uy3d2));
    fAux[8]  += multiplyTerm*(terms[2] + (-uy3 + ux3uy3 + uy3uy3d2));
    fAux[9]  += multiplyTerm*(terms[1] + ( uz3 + ux3uz3 + uz3uz3d2));
    fAux[10] += multiplyTerm*(terms[2] + (-uz3 + ux3uz3 + uz3uz3d2));
    fAux[11] += multiplyTerm*(terms[3] + ( uz3 + uy3uz3 + uz3uz3d2));
    fAux[12] += multiplyTerm*(terms[4] + (-uz3 + uy3uz3 + uz3uz3d2));
    fAux[13] += multiplyTerm*(terms[1] + (-uy3 - ux3uy3 + uy3uy3d2));
    fAux[14] += multiplyTerm*(terms[2] + ( uy3 - ux3uy3 + uy3uy3d2));
    fAux[15] += multiplyTerm*(terms[1] + (-uz3 - ux3uz3 + uz3uz3d2));
    fAux[16] += multiplyTerm*(terms[2] + ( uz3 - ux3uz3 + uz3uz3d2));
    fAux[17] += multiplyTerm*(terms[3] + (-uz3 - uy3uz3 + uz3uz3d2));
    fAux[18] += multiplyTerm*(terms[4] + ( uz3 - uy3uz3 + uz3uz3d2));
#ifdef D3Q27
    multiplyTerm = rhoW3;
    fAux[19] += multiplyTerm*terms[19];
    fAux[20] += multiplyTerm*terms[20];
    fAux[21] += multiplyTerm*terms[21];
    fAux[22] += multiplyTerm*terms[22];
    fAux[23] += multiplyTerm*terms[23];
    fAux[24] += multiplyTerm*terms[24];
    fAux[25] += multiplyTerm*terms[25];
    fAux[26] += multiplyTerm*terms[26];
#endif

    // calculate force term
    // term[0] -> population 0
    // term[1] -> population 1
    // term[2] -> population 3
    // term[3] -> population 7
    // term[4] -> population 9
    // term[5] -> population 11
    terms[0] = - FX*ux3 - FY*uy3 - FZ*uz3;
    terms[1] = terms[0] + (FX*( 3*ux3 + 3));
    terms[2] = terms[0] + (FY*( 3*uy3 + 3));
    terms[3] = terms[1] + (FX*( 3*uy3) + FY*( 3*ux3 + 3*uy3 + 3));
    terms[4] = terms[1] + (FX*( 3*uz3) + FZ*( 3*ux3 + 3*uz3 + 3));
    terms[5] = terms[2] + (FY*( 3*uz3) + FZ*( 3*uy3 + 3*uz3 + 3));

#ifdef D3Q27
    terms[19] = terms[3] + (FX*( 3*uz3) + FY*( 3*uz3) + FZ*( 3*ux3 + 3*uy3 + 3*uz3 + 3));
    terms[20] = terms[19] + (FX*(-6) + FY*(-6) + FZ*(-6));
    terms[21] = terms[19] + (FX*(-6*uz3) + FY*(-6*uz3) + FZ*(-6*ux3 - 6*uy3 - 6));
    terms[22] = terms[21] + (FX*(-6) + FY*(-6) + FZ*( 6));
    terms[23] = terms[19] + (FX*(-6*uy3) + FY*(-6*ux3 - 6*uz3 - 6) + FZ*(-6*uy3));
    terms[24] = terms[23] + (FX*(-6) + FY*( 6) + FZ*(-6));
    terms[25] = terms[19] + (FX*(-6*uy3 - 6*uz3 - 6) + FY*(-6*ux3) + FZ*(-6*ux3));
    terms[26] = terms[25] + (FX*( 6) + FY*(-6) + FZ*(-6));
#endif

    // add force term
    multiplyTerm = W0*TT_OMEGA;
    fAux[0] += multiplyTerm*terms[0];
    
    multiplyTerm = W1*TT_OMEGA;
    fAux[1] += multiplyTerm*terms[1];
    fAux[2] += multiplyTerm*(terms[1] + (FX*(-6)));
    fAux[3] += multiplyTerm*terms[2];
    fAux[4] += multiplyTerm*(terms[2] + (FY*(-6)));
    auxTerm = terms[0] + (FZ*( 3*uz3 + 3));
    fAux[5] += multiplyTerm*auxTerm;
    fAux[6] += multiplyTerm*(auxTerm + (FZ*(-6)));

    multiplyTerm = W2*TT_OMEGA;
    fAux[7] += multiplyTerm*terms[3];
    fAux[8] += multiplyTerm*(terms[3] + (FX*(-6) + FY*(-6)));
    fAux[9] += multiplyTerm*terms[4];
    fAux[10] += multiplyTerm*(terms[4] + (FX*(-6) + FZ*(-6)));
    fAux[11] += multiplyTerm*(terms[5]);
    fAux[12] += multiplyTerm*(terms[5] + (FY*(-6) + FZ*(-6)));
    auxTerm = terms[3] + (FX*(-6*uy3) + FY*(-6*ux3 - 6));
    fAux[13] += multiplyTerm*(auxTerm);
    fAux[14] += multiplyTerm*(auxTerm + (FX*(-6) + FY*( 6)));
    auxTerm = terms[4] + (FX*(-6*uz3) + FZ*(-6*ux3 - 6));
    fAux[15] += multiplyTerm*auxTerm;
    fAux[16] += multiplyTerm*(auxTerm + (FX*(-6) + FZ*( 6)));
    auxTerm = terms[5] + (FY*(-6*uz3) + FZ*(-6*uy3 - 6));
    fAux[17] += multiplyTerm*auxTerm;
    fAux[18] += multiplyTerm*(auxTerm + (FY*(-6) + FZ*( 6)));

#ifdef D3Q27
    multiplyTerm = W3*TT_OMEGA;
    fAux[19] += multiplyTerm*terms[19];
    fAux[20] += multiplyTerm*terms[20];
    fAux[21] += multiplyTerm*terms[21];
    fAux[22] += multiplyTerm*terms[22];
    fAux[23] += multiplyTerm*terms[23];
    fAux[24] += multiplyTerm*terms[24];
    fAux[25] += multiplyTerm*terms[25];
    fAux[26] += multiplyTerm*terms[26];
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
    macr->rho[idxScalar(x, y, z)] = rhoVar;
    macr->ux[idxScalar(x, y, z)] = uxVar;
    macr->uy[idxScalar(x, y, z)] = uyVar;
    macr->uz[idxScalar(x, y, z)] = uzVar;
}
