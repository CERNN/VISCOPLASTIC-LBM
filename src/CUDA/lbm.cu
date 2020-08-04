#include "lbm.h"

__global__ 
void gpuMacrCollisionStream(
    dfloat* const pop,
    dfloat* const popAux,
    NodeTypeMap* const mapBC,
    Macroscopics const macr,
    bool const save,
    int const step)
{
    const short unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const short unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const short unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    if(!mapBC[idxScalar(x, y, z)].getIsUsed())
        return;

    // Adjacent coordinates
    const unsigned short int xp1 = (x + 1) % NX;
    const unsigned short int yp1 = (y + 1) % NY;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int xm1 = (NX + x - 1) % NX;
    const unsigned short int ym1 = (NY + y - 1) % NY;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;

    // Node populations
    dfloat fNode[Q];

    // Load populations
    #pragma unroll
    for (char i = 0; i < Q; i++)
        fNode[i] = pop[idxPop(x, y, z, i)];

    // Calculate macroscopics
    // rho = sum(f[i])
    // ux = (sum(f[i]*cx[i])+0.5*FX) / rho
    // uy = (sum(f[i]*cy[i])+0.5*FY) / rho
    // uz = (sum(f[i]*cz[i])+0.5*FZ) / rho
    #ifdef D3Q19
    const dfloat rhoVar = fNode[0] + fNode[1] + fNode[2] + fNode[3] + fNode[4] 
        + fNode[5] + fNode[6] + fNode[7] + fNode[8] + fNode[9] + fNode[10] 
        + fNode[11] + fNode[12] + fNode[13] + fNode[14] + fNode[15] + fNode[16] 
        + fNode[17] + fNode[18];
    const dfloat invRho = 1/rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15])
        - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16]) + 0.5*FX) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18]) + 0.5*FY) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17]) + 0.5*FZ) * invRho;
    #endif // !D3Q19
    #ifdef D3Q27
    const dfloat rhoVar = fNode[0] + fNode[1] + fNode[2] + fNode[3] + fNode[4] 
        + fNode[5] + fNode[6] + fNode[7] + fNode[8] + fNode[9] + fNode[10] 
        + fNode[11] + fNode[12] + fNode[13] + fNode[14] + fNode[15] + fNode[16] 
        + fNode[17] + fNode[18] + fNode[19] + fNode[20] + fNode[21] + fNode[22]
        + fNode[23] + fNode[24] + fNode[25] + fNode[26];
        const dfloat invRho = 1/rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15]
        + fNode[19] + fNode[21] + fNode[23] + fNode[26]) 
        - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16] + fNode[20]
        + fNode[22] + fNode[24] + fNode[25]) + 0.5*FX) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17]
        + fNode[19] + fNode[21] + fNode[24] + fNode[25])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18] + fNode[20]
        + fNode[22] + fNode[23] + fNode[26]) + 0.5*FY) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18]
        + fNode[19] + fNode[22] + fNode[23] + fNode[25])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17] + fNode[20]
        + fNode[21] + fNode[24] + fNode[26]) + 0.5*FZ) * invRho;
    #endif // !D3Q27

    if (save)
    {
        macr.rho[idxScalar(x, y, z)] = rhoVar;
        macr.ux[idxScalar(x, y, z)] = uxVar;
        macr.uy[idxScalar(x, y, z)] = uyVar;
        macr.uz[idxScalar(x, y, z)] = uzVar;
    }

    // Calculate temporary variables
    const dfloat p1_muu15 = 1 - 1.5 * (uxVar * uxVar + 
        uyVar * uyVar + uzVar * uzVar);
    const dfloat rhoW0 = rhoVar * W0;
    const dfloat rhoW1 = rhoVar * W1;
    const dfloat rhoW2 = rhoVar * W2;
    const dfloat W1t9d2 = W1 * 4.5;
    const dfloat W2t9d2 = W2 * 4.5;
    #ifdef D3Q27
    const dfloat rhoW3 = rhoVar * W3;
    const dfloat W3t9d2 = W3 * 4.5;
    #endif
    const dfloat ux3 = 3 * uxVar;
    const dfloat uy3 = 3 * uyVar;
    const dfloat uz3 = 3 * uzVar;
    const dfloat ux3ux3d2 = ux3*ux3*0.5;
    const dfloat ux3uy3 = ux3*uy3;
    const dfloat ux3uz3 = ux3*uz3;
    const dfloat uy3uy3d2 = uy3*uy3*0.5;
    const dfloat uy3uz3 = uy3*uz3;
    const dfloat uz3uz3d2 = uz3*uz3*0.5;

    // Terms to use to recursive calculations
    #ifdef D3Q19
    dfloat terms[6];
    #endif 
    #ifdef D3Q27
    dfloat terms[11];
    #endif
    dfloat multiplyTerm = 1;
    dfloat auxTerm;


    // Calculate pineq(alfa, beta)
    // pineqAB = pops - popsEquilibrium
    #ifdef D3Q19
    const dfloat pineqXX = (fNode[1] + fNode[2] + fNode[7] + fNode[8] + fNode[9] 
            + fNode[10] + fNode[13] + fNode[14] + fNode[15] + fNode[16]) -
            (2*rhoW1*(p1_muu15 + ux3ux3d2) + 
             4*rhoW2*(2*p1_muu15 + 2*ux3ux3d2 + uy3uy3d2 + uz3uz3d2));
    const dfloat pineqYY = (fNode[3] + fNode[4] + fNode[7] + fNode[8] + fNode[11]
            + fNode[12] + fNode[13] + fNode[14] + fNode[17] + fNode[18]) -
            (2*rhoW1*(p1_muu15 + uy3uy3d2) + 
             4*rhoW2*(2*p1_muu15 + ux3ux3d2 + 2*uy3uy3d2 + uz3uz3d2)); 
    const dfloat pineqZZ = (fNode[5] + fNode[6] + fNode[9] + fNode[10] + fNode[11]
            + fNode[12] + fNode[15] + fNode[16] + fNode[17] + fNode[18]) -
            (2*rhoW1*(p1_muu15 + uz3uz3d2) + 
             4*rhoW2*(2*p1_muu15 + ux3ux3d2 + uy3uy3d2 + 2*uz3uz3d2));
    const dfloat pineqXYt2 = ((fNode[7] + fNode[8] - fNode[13] - fNode[14]) - 
            (4*rhoW2*(ux3uy3))) * 2;
    const dfloat pineqXZt2 = ((fNode[9] + fNode[10] - fNode[15] - fNode[16]) -
            (4*rhoW2*(ux3uz3))) * 2;
    const dfloat pineqYZt2 = ((fNode[11] + fNode[12] - fNode[17] - fNode[18]) -
            (4*rhoW2*(uy3uz3))) * 2;
    #endif // !D3Q19 
    #ifdef D3Q27
    const dfloat aux = (fNode[19] + fNode[20] + fNode[21] + fNode[22] + fNode[23]
            + fNode[24] + fNode[25] + fNode[26]) - 
            (8*rhoW3*(ux3ux3d2 + uy3uy3d2 + uz3uz3d2));
    const dfloat pineqXX = (fNode[1] + fNode[2] + fNode[7] + fNode[8] + fNode[9] 
            + fNode[10] + fNode[13] + fNode[14] + fNode[15] + fNode[16] + aux) -
            (2*rhoW1*(p1_muu15 + ux3ux3d2) + 
             4*rhoW2*(2*p1_muu15 + 2*ux3ux3d2 + uy3uy3d2 + uz3uz3d2));
    const dfloat pineqYY = (fNode[3] + fNode[4] + fNode[7] + fNode[8] + fNode[11]
            + fNode[12] + fNode[13] + fNode[14] + fNode[17] + fNode[18] + aux) -
            (2*rhoW1*(p1_muu15 + uy3uy3d2) + 
             4*rhoW2*(ux3ux3d2 + 2*uy3uy3d2 + uz3uz3d2)); 
    const dfloat pineqZZ = (fNode[5] + fNode[6] + fNode[9] + fNode[10] + fNode[11]
            + fNode[12] + fNode[15] + fNode[16] + fNode[17] + fNode[18] + aux) -
            (2*rhoW1*(p1_muu15 + uz3uz3d2) + 
             4*rhoW2*(ux3ux3d2 + uy3uy3d2 + 2*uz3uz3d2));
    const dfloat pineqXYt2 = ((fNode[7] + fNode[8] - fNode[13] - fNode[14] + fNode[19]
            + fNode[20] + fNode[21] + fNode[22] - fNode[23] - fNode[24] - fNode[25]
            - fNode[26]) - 
            (4*rhoW2*(ux3uy3) + 8*rhoW3*(ux3uy3))) * 2;
    const dfloat pineqXZt2 = ((fNode[9] + fNode[10] - fNode[15] - fNode[16] + fNode[19]
            + fNode[20] - fNode[21] - fNode[22] + fNode[23] + fNode[24] - fNode[25]
            - fNode[26]) - 
            (4*rhoW2*(ux3uz3) + 8*rhoW3*(ux3uz3))) * 2;
    const dfloat pineqYZt2 = ((fNode[11] + fNode[12] - fNode[17] - fNode[18] + fNode[19]
            + fNode[20] - fNode[21] - fNode[22] - fNode[23] - fNode[24] + fNode[25]
            + fNode[26]) - 
            (4*rhoW2*(uy3uz3) + 8*rhoW3*(uy3uz3))) * 2;
    #endif // !D3Q27

    // Calculate regularization terms 
    // terms[i] = Q[i, alfa, beta]*pi[i, alfa, beta] - c[i, alfa]*F[alfa]/3
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
    // terms[5] -> population 7
    // terms[6] -> population 8
    // terms[7] -> population 9
    // terms[8] -> population 10
    // terms[9] -> population 11
    // terms[10] -> population 12
    terms[5] = terms[1] + (-FY_D3 + pineqXYt2 + pineqYY);
    terms[6] = terms[2] + ( FY_D3 + pineqXYt2 + pineqYY);
    terms[7] = terms[1] + (-FZ_D3 + pineqXZt2 + pineqZZ);
    terms[8] = terms[2] + ( FZ_D3 + pineqXZt2 + pineqZZ);
    terms[9] = terms[3] + (-FZ_D3 + pineqYZt2 + pineqZZ);
    terms[10] = terms[4] + ( FZ_D3 + pineqYZt2 + pineqZZ);
    #endif
    
    // Calculate regularized population to fNode
    // fNode[i] = 4.5*w[i](Q[i, alfa, beta]*pi[i, alfa, beta] 
    //          - c[i, alfa]*F[alfa]/3)
    multiplyTerm = W0*4.5;
    fNode[0] = multiplyTerm*terms[0];
    multiplyTerm = W1t9d2;
    fNode[1] = multiplyTerm*terms[1];
    fNode[2] = multiplyTerm*terms[2];
    fNode[3] = multiplyTerm*terms[3];
    fNode[4] = multiplyTerm*terms[4];
    fNode[5] = multiplyTerm*(terms[0] + (-FZ_D3 + pineqZZ));
    fNode[6] = multiplyTerm*(terms[0] + ( FZ_D3 + pineqZZ));
    multiplyTerm = W2t9d2;
    fNode[7] = multiplyTerm*(terms[1] + (-FY_D3 + pineqXYt2 + pineqYY));
    fNode[8] = multiplyTerm*(terms[2] + ( FY_D3 + pineqXYt2 + pineqYY));
    fNode[9] = multiplyTerm*(terms[1] + (-FZ_D3 + pineqXZt2 + pineqZZ));
    fNode[10] = multiplyTerm*(terms[2] + ( FZ_D3 + pineqXZt2 + pineqZZ));
    fNode[11] = multiplyTerm*(terms[3] + (-FZ_D3 + pineqYZt2 + pineqZZ));
    fNode[12] = multiplyTerm*(terms[4] + ( FZ_D3 + pineqYZt2 + pineqZZ));
    fNode[13] = multiplyTerm*(terms[1] + ( FY_D3 - pineqXYt2 + pineqYY));
    fNode[14] = multiplyTerm*(terms[2] + (-FY_D3 - pineqXYt2 + pineqYY));
    fNode[15] = multiplyTerm*(terms[1] + ( FZ_D3 - pineqXZt2 + pineqZZ));
    fNode[16] = multiplyTerm*(terms[2] + (-FZ_D3 - pineqXZt2 + pineqZZ));
    fNode[17] = multiplyTerm*(terms[3] + ( FZ_D3 - pineqYZt2 + pineqZZ));
    fNode[18] = multiplyTerm*(terms[4] + (-FZ_D3 - pineqYZt2 + pineqZZ));
    #ifdef D3Q27
    multiplyTerm = W3t9d2;
    fNode[19] = multiplyTerm*(terms[5] + (-FZ_D3 + pineqXZt2 + pineqYZt2 + pineqZZ));
    fNode[20] = multiplyTerm*(terms[6] + ( FZ_D3 + pineqXZt2 + pineqYZt2 + pineqZZ));
    fNode[21] = multiplyTerm*(terms[5] + ( FZ_D3 - pineqXZt2 - pineqYZt2 + pineqZZ));
    fNode[22] = multiplyTerm*(terms[6] + (-FZ_D3 - pineqXZt2 - pineqYZt2 + pineqZZ));
    fNode[23] = multiplyTerm*(terms[7] + ( FY_D3 - pineqXYt2 + pineqYY - pineqYZt2));
    fNode[24] = multiplyTerm*(terms[8] + (-FY_D3 - pineqXYt2 + pineqYY - pineqYZt2));
    fNode[25] = multiplyTerm*(terms[9] + ( FX_D3 + pineqXX - pineqXYt2 - pineqXZt2));
    fNode[26] = multiplyTerm*(terms[10] + (-FX_D3 + pineqXX - pineqXYt2 - pineqXZt2));
    #endif

    // Collision to fNode:
    // fNode = (1 - 1/TAU)*f1 + fEq + (1 - 0.5/TAU)*force ->
    // fNode = (1 - OMEGA)*f1 + fEq + (1 - 0.5*0MEGA)*force->
    // fNode = T_OMEGA * f1 + fEq + TT_OMEGA*force

    // Sequence is:
    // fNode *= T_OMEGA
    // fNode += fEq
    // fNode += TT_OMEGA*force

    #pragma unroll
    for(char i = 0; i < Q; i++)
        fNode[i] *= T_OMEGA;

    // Calculate equilibrium terms 
    // terms = 0.5*uc3^2 + uc3
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
    // terms[5] -> population 7
    // terms[6] -> population 8
    // terms[7] -> population 9
    // terms[8] -> population 10
    // terms[9] -> population 11
    // terms[10] -> population 12
    terms[5] = terms[1] + ( uy3 + ux3uy3 + uy3uy3d2);
    terms[6] = terms[2] + (-uy3 + ux3uy3 + uy3uy3d2);
    terms[7] = terms[1] + ( uz3 + ux3uz3 + uz3uz3d2);
    terms[8] = terms[2] + (-uz3 + ux3uz3 + uz3uz3d2);
    terms[9] = terms[3] + ( uz3 + uy3uz3 + uz3uz3d2);
    terms[10] = terms[4] + (-uz3 + uy3uz3 + uz3uz3d2);
    #endif

    // fNode += fEq
    multiplyTerm = rhoW0;
    fNode[0] += multiplyTerm*terms[0];
    multiplyTerm = rhoW1;
    fNode[1] += multiplyTerm*terms[1];
    fNode[2] += multiplyTerm*terms[2];
    fNode[3] += multiplyTerm*terms[3];
    fNode[4] += multiplyTerm*terms[4];
    fNode[5] += multiplyTerm*(terms[0] + ( uz3 + uz3uz3d2));
    fNode[6] += multiplyTerm*(terms[0] + (-uz3 + uz3uz3d2));
    multiplyTerm = rhoW2;
    fNode[7]  += multiplyTerm*(terms[1] + ( uy3 + ux3uy3 + uy3uy3d2));
    fNode[8]  += multiplyTerm*(terms[2] + (-uy3 + ux3uy3 + uy3uy3d2));
    fNode[9]  += multiplyTerm*(terms[1] + ( uz3 + ux3uz3 + uz3uz3d2));
    fNode[10] += multiplyTerm*(terms[2] + (-uz3 + ux3uz3 + uz3uz3d2));
    fNode[11] += multiplyTerm*(terms[3] + ( uz3 + uy3uz3 + uz3uz3d2));
    fNode[12] += multiplyTerm*(terms[4] + (-uz3 + uy3uz3 + uz3uz3d2));
    fNode[13] += multiplyTerm*(terms[1] + (-uy3 - ux3uy3 + uy3uy3d2));
    fNode[14] += multiplyTerm*(terms[2] + ( uy3 - ux3uy3 + uy3uy3d2));
    fNode[15] += multiplyTerm*(terms[1] + (-uz3 - ux3uz3 + uz3uz3d2));
    fNode[16] += multiplyTerm*(terms[2] + ( uz3 - ux3uz3 + uz3uz3d2));
    fNode[17] += multiplyTerm*(terms[3] + (-uz3 - uy3uz3 + uz3uz3d2));
    fNode[18] += multiplyTerm*(terms[4] + ( uz3 - uy3uz3 + uz3uz3d2));
    #ifdef D3Q27
    multiplyTerm = rhoW3;
    fNode[19] += multiplyTerm*(terms[5] + ( uz3 + ux3uz3 + uy3uz3 + uz3uz3d2));
    fNode[20] += multiplyTerm*(terms[6] + (-uz3 + ux3uz3 + uy3uz3 + uz3uz3d2));
    fNode[21] += multiplyTerm*(terms[5] + (-uz3 - ux3uz3 - uy3uz3 + uz3uz3d2));
    fNode[22] += multiplyTerm*(terms[6] + ( uz3 - ux3uz3 - uy3uz3 + uz3uz3d2));
    fNode[23] += multiplyTerm*(terms[7] + (-uy3 - ux3uy3 + uy3uy3d2 - uy3uz3));
    fNode[24] += multiplyTerm*(terms[8] + ( uy3 - ux3uy3 + uy3uy3d2 - uy3uz3));
    fNode[25] += multiplyTerm*(terms[9] + (-ux3 + ux3ux3d2 - ux3uy3 - ux3uz3));
    fNode[26] += multiplyTerm*(terms[10] + ( ux3 + ux3ux3d2 - ux3uy3 - ux3uz3));
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
    // term[6] -> population 19
    terms[6] = terms[3] + (FX*( 3*uz3) + FY*( 3*uz3) + FZ*( 3*ux3 + 3*uy3 + 3*uz3 + 3));
    #endif

    // fNode += TT_OMEGA * force
    multiplyTerm = W0*TT_OMEGA;
    fNode[0] += multiplyTerm*terms[0];
    multiplyTerm = W1*TT_OMEGA;
    fNode[1] += multiplyTerm*terms[1];
    fNode[2] += multiplyTerm*(terms[1] + (FX*(-6)));
    fNode[3] += multiplyTerm*terms[2];
    fNode[4] += multiplyTerm*(terms[2] + (FY*(-6)));
    auxTerm = terms[0] + (FZ*( 3*uz3 + 3));
    fNode[5] += multiplyTerm*auxTerm;
    fNode[6] += multiplyTerm*(auxTerm + (FZ*(-6)));
    multiplyTerm = W2*TT_OMEGA;
    fNode[7] += multiplyTerm*terms[3];
    fNode[8] += multiplyTerm*(terms[3] + (FX*(-6) + FY*(-6)));
    fNode[9] += multiplyTerm*terms[4];
    fNode[10] += multiplyTerm*(terms[4] + (FX*(-6) + FZ*(-6)));
    fNode[11] += multiplyTerm*(terms[5]);
    fNode[12] += multiplyTerm*(terms[5] + (FY*(-6) + FZ*(-6)));
    auxTerm = terms[3] + (FX*(-6*uy3) + FY*(-6*ux3 - 6));
    fNode[13] += multiplyTerm*(auxTerm);
    fNode[14] += multiplyTerm*(auxTerm + (FX*(-6) + FY*( 6)));
    auxTerm = terms[4] + (FX*(-6*uz3) + FZ*(-6*ux3 - 6));
    fNode[15] += multiplyTerm*auxTerm;
    fNode[16] += multiplyTerm*(auxTerm + (FX*(-6) + FZ*( 6)));
    auxTerm = terms[5] + (FY*(-6*uz3) + FZ*(-6*uy3 - 6));
    fNode[17] += multiplyTerm*auxTerm;
    fNode[18] += multiplyTerm*(auxTerm + (FY*(-6) + FZ*( 6)));
    #ifdef D3Q27
    multiplyTerm = W3*TT_OMEGA;
    fNode[19] += multiplyTerm*terms[6];
    fNode[20] += multiplyTerm*(terms[6] + (FX*(-6) + FY*(-6) + FZ*(-6)));
    auxTerm = terms[6] + (FX*(-6*uz3) + FY*(-6*uz3) + FZ*(-6*ux3 - 6*uy3 - 6));
    fNode[21] += multiplyTerm*auxTerm;
    fNode[22] += multiplyTerm*(auxTerm + (FX*(-6) + FY*(-6) + FZ*( 6)));
    auxTerm = terms[6] + (FX*(-6*uy3) + FY*(-6*ux3 - 6*uz3 - 6) + FZ*(-6*uy3));
    fNode[23] += multiplyTerm*auxTerm;
    fNode[24] += multiplyTerm*(auxTerm + (FX*(-6) + FY*( 6) + FZ*(-6)));
    auxTerm = terms[6] + (FX*(-6*uy3 - 6*uz3 - 6) + FY*(-6*ux3) + FZ*(-6*ux3));
    fNode[25] += multiplyTerm*auxTerm;
    fNode[26] += multiplyTerm*(auxTerm + (FX*( 6) + FY*(-6) + FZ*(-6)));
    #endif

    // Save post collision populations of boundary conditions nodes
    if(mapBC[idxScalar(x, y, z)].getSavePostCol())  
    {
        #pragma unroll
        for (char i = 0; i < Q; i++)
            pop[idxPop(x, y, z, i)] = fNode[i];
    }

    // Streaming to popAux
    // popAux(x+cx, y+cy, z+cz, i) = pop(x, y, z, i) 
    // The populations that shoudn't be streamed will be changed by the boundary conditions
    popAux[idxPop(x, y, z, 0)] = fNode[0];
    popAux[idxPop(xp1, y, z, 1)] = fNode[1];
    popAux[idxPop(xm1, y, z, 2)] = fNode[2];
    popAux[idxPop(x, yp1, z, 3)] = fNode[3];
    popAux[idxPop(x, ym1, z, 4)] = fNode[4];
    popAux[idxPop(x, y, zp1, 5)] = fNode[5];
    popAux[idxPop(x, y, zm1, 6)] = fNode[6];
    popAux[idxPop(xp1, yp1, z, 7)] = fNode[7];
    popAux[idxPop(xm1, ym1, z, 8)] = fNode[8];
    popAux[idxPop(xp1, y, zp1, 9)] = fNode[9];
    popAux[idxPop(xm1, y, zm1, 10)] = fNode[10];
    popAux[idxPop(x, yp1, zp1, 11)] = fNode[11];
    popAux[idxPop(x, ym1, zm1, 12)] = fNode[12];
    popAux[idxPop(xp1, ym1, z, 13)] = fNode[13];
    popAux[idxPop(xm1, yp1, z, 14)] = fNode[14];
    popAux[idxPop(xp1, y, zm1, 15)] = fNode[15];
    popAux[idxPop(xm1, y, zp1, 16)] = fNode[16];
    popAux[idxPop(x, yp1, zm1, 17)] = fNode[17];
    popAux[idxPop(x, ym1, zp1, 18)] = fNode[18];
    #ifdef D3Q27
    popAux[idxPop(xp1, yp1, zp1, 19)] = fNode[19];
    popAux[idxPop(xm1, ym1, zm1, 20)] = fNode[20];
    popAux[idxPop(xp1, yp1, zm1, 21)] = fNode[21];
    popAux[idxPop(xm1, ym1, zp1, 22)] = fNode[22];
    popAux[idxPop(xp1, ym1, zp1, 23)] = fNode[23];
    popAux[idxPop(xm1, yp1, zm1, 24)] = fNode[24];
    popAux[idxPop(xm1, yp1, zp1, 25)] = fNode[25];
    popAux[idxPop(xp1, ym1, zm1, 26)] = fNode[26];
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
    dfloat fNode[Q];
    for (unsigned char i = 0; i < Q; i++)
        fNode[i] = pop->pop[idxPop(x, y, z, i)];

    // calc for macroscopics
    // rho = sum(f[i])
    // ux = sum(f[i]*cx[i]) / rho
    // uy = sum(f[i]*cy[i]) / rho
    // uz = sum(f[i]*cz[i]) / rho
    #ifdef D3Q19
    const dfloat rhoVar = fNode[0] + fNode[1] + fNode[2] + fNode[3] + fNode[4] 
        + fNode[5] + fNode[6] + fNode[7] + fNode[8] + fNode[9] + fNode[10] 
        + fNode[11] + fNode[12] + fNode[13] + fNode[14] + fNode[15] + fNode[16] 
        + fNode[17] + fNode[18];
    const dfloat invRho = 1/rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15])
        - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16]) + 0.5*FX) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18]) + 0.5*FY) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17]) + 0.5*FZ) * invRho;
    #endif // !D3Q19
    #ifdef D3Q27
    const dfloat rhoVar = fNode[0] + fNode[1] + fNode[2] + fNode[3] + fNode[4] 
        + fNode[5] + fNode[6] + fNode[7] + fNode[8] + fNode[9] + fNode[10] 
        + fNode[11] + fNode[12] + fNode[13] + fNode[14] + fNode[15] + fNode[16] 
        + fNode[17] + fNode[18] + fNode[19] + fNode[20] + fNode[21] + fNode[22]
        + fNode[23] + fNode[24] + fNode[25] + fNode[26];
        const dfloat invRho = 1/rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15]
        + fNode[19] + fNode[21] + fNode[23] + fNode[26]) 
        - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16] + fNode[20]
        + fNode[22] + fNode[24] + fNode[25]) + 0.5*FX) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17]
        + fNode[19] + fNode[21] + fNode[24] + fNode[25])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18] + fNode[20]
        + fNode[22] + fNode[23] + fNode[26]) + 0.5*FY) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18]
        + fNode[19] + fNode[22] + fNode[23] + fNode[25])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17] + fNode[20]
        + fNode[21] + fNode[24] + fNode[26]) + 0.5*FZ) * invRho;
    #endif // !D3Q27
    macr->rho[idxScalar(x, y, z)] = rhoVar;
    macr->ux[idxScalar(x, y, z)] = uxVar;
    macr->uy[idxScalar(x, y, z)] = uyVar;
    macr->uz[idxScalar(x, y, z)] = uzVar;
}


__global__
void gpuApplyBC(NodeTypeMap* mapBC,  
    dfloat* popPostStream,
    dfloat* popPostCol,
    size_t* idxsBCNodes,
    size_t totalBCNodes)
{
    const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i >= totalBCNodes)
        return;
    // converts 1D index to 3D location
    const size_t idx = idxsBCNodes[i];
    const unsigned int x = idx % NX;
    const unsigned int y = (idx/NX) % NY;
    const unsigned int z = idx/(NX*NY);

    gpuBoundaryConditions(&(mapBC[idx]), popPostStream, popPostCol, x, y, z);
}

__global__
void gpuPopulationsTransfer(
    dfloat* popPostStreamBase,
    dfloat* popPostCollBase,
    dfloat* popPostStreamNxt,
    dfloat* popPostCollNxt)
{
    const unsigned short int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned short int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned short int zMax = NZ-1;

    if (x >= NX || y >= NY)
        return;

    // This takes into account that the populations are "teleported"
    // from one side of domain to another. So the population with cz=-1
    // in z = 0 is streamed to z = NZ-1.
    // In this way, to retrieve a population that should have been sent 
    // to the adjacent node, but was "teleported", the part of the domain 
    // to which it was streamed must be read.
    // Also important to notice is that the popBase is above the popNext,
    // so the lower level of popBase must be streamed to the higher level of
    // popNext and vice versa

    // pop[6] -> cz = 1; pop[5] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 5)] = popPostStreamNxt[idxPop(x, y, 0, 5)];
    popPostStreamNxt[idxPop(x, y, zMax, 6)] = popPostStreamBase[idxPop(x, y, zMax, 6)];

    // pop[9] -> cz = 1; pop[10] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 9)] = popPostStreamNxt[idxPop(x, y, 0, 9)];
    popPostStreamNxt[idxPop(x, y, zMax, 10)] = popPostStreamBase[idxPop(x, y, zMax, 10)];
    
    // pop[11] -> cz = 1; pop[12] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 11)] = popPostStreamNxt[idxPop(x, y, 0, 11)];
    popPostStreamNxt[idxPop(x, y, zMax, 12)] = popPostStreamBase[idxPop(x, y, zMax, 12)];
    
    // pop[15] -> cz = 1; pop[16] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 16)] = popPostStreamNxt[idxPop(x, y, 0, 16)];
    popPostStreamNxt[idxPop(x, y, zMax, 15)] = popPostStreamBase[idxPop(x, y, zMax, 15)];

    // pop[18] -> cz = 1; pop[17] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 18)] =   popPostStreamNxt[idxPop(x, y, 0, 18)];
    popPostStreamNxt[idxPop(x, y, zMax, 17)] = popPostStreamBase[idxPop(x, y, zMax, 17)];

    #ifdef D3Q27
    // pop[19] -> cz = 1; pop[20] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 19)] = popPostStreamNxt[idxPop(x, y, 0, 19)];
    popPostStreamNxt[idxPop(x, y, zMax, 20)] = popPostStreamBase[idxPop(x, y, zMax, 20)];

    // pop[22] -> cz = 1; pop[21] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 22)] = popPostStreamNxt[idxPop(x, y, 0, 22)];
    popPostStreamNxt[idxPop(x, y, zMax, 21)] = popPostStreamBase[idxPop(x, y, zMax, 21)];

    // pop[23] -> cz = 1; pop[24] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 23)] = popPostStreamNxt[idxPop(x, y, 0, 23)];
    popPostStreamNxt[idxPop(x, y, zMax, 24)] = popPostStreamBase[idxPop(x, y, zMax, 24)];

    // pop[25] -> cz = 1; pop[26] -> cz = -1;
    popPostStreamBase[idxPop(x, y, 0, 25)] = popPostStreamNxt[idxPop(x, y, 0, 25)];
    popPostStreamNxt[idxPop(x, y, zMax, 26)] = popPostStreamBase[idxPop(x, y, zMax, 26)];
    #endif
}