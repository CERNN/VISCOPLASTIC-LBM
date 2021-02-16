#include "lbm.h"

__global__ 
void gpuMacrCollisionStream(
    dfloat* const pop,
    dfloat* const popAux,
    NodeTypeMap* const mapBC,
    Macroscopics const macr,
    bool const save,
#ifdef SCALAR_TRANSPORT
    dfloat* const gPop,
    dfloat* const gPopAux,
#endif
    int const step)
{
    const short unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const short unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const short unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t idx = idxScalar(x, y, z);
    if(!mapBC[idx].getIsUsed())
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

    #ifdef SCALAR_TRANSPORT
        dfloat gNode[GQ];
        dfloat geq[GQ];    
    #endif 
    // Aux idx

    // Load populations
    #pragma unroll
    for (char i = 0; i < Q; i++)
        fNode[i] = pop[idxPop(x, y, z, i)];

    #ifdef SCALAR_TRANSPORT
        #pragma unroll
        for (char i = 0; i < GQ; i++)
            gNode[i] = gPop[idxPop(x, y, z, i)];
    #endif

    #ifdef IBM
    const dfloat fxVar = macr.f.x[idx];
    const dfloat fyVar = macr.f.y[idx];
    const dfloat fzVar = macr.f.z[idx];
    // Reset IBM forces
    macr.f.x[idx] = FX;
    macr.f.y[idx] = FY;
    macr.f.z[idx] = FZ;
    const dfloat fxVar_D3 = fxVar / 3;
    const dfloat fyVar_D3 = fyVar / 3;
    const dfloat fzVar_D3 = fzVar / 3;
    #else
    const dfloat fxVar = FX;
    const dfloat fyVar = FY;
    const dfloat fzVar = FZ;
    const dfloat fxVar_D3 = FX / 3;
    const dfloat fyVar_D3 = FY / 3;
    const dfloat fzVar_D3 = FZ / 3;
    #endif

    // Calculate macroscopics
    // rho = sum(f[i])
    // ux = (sum(f[i]*cx[i])+0.5*fxVar) / rho
    // uy = (sum(f[i]*cy[i])+0.5*fyVar) / rho
    // uz = (sum(f[i]*cz[i])+0.5*fzVar) / rho
    #ifdef D3Q19
    const dfloat rhoVar = fNode[0] + fNode[1] + fNode[2] + fNode[3] + fNode[4] 
        + fNode[5] + fNode[6] + fNode[7] + fNode[8] + fNode[9] + fNode[10] 
        + fNode[11] + fNode[12] + fNode[13] + fNode[14] + fNode[15] + fNode[16] 
        + fNode[17] + fNode[18];
    const dfloat invRho = 1/rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15])
        - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16]) + 0.5*fxVar) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18]) + 0.5*fyVar) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17]) + 0.5*fzVar) * invRho;
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
        + fNode[22] + fNode[24] + fNode[25]) + 0.5*fxVar) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17]
        + fNode[19] + fNode[21] + fNode[24] + fNode[25])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18] + fNode[20]
        + fNode[22] + fNode[23] + fNode[26]) + 0.5*fyVar) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18]
        + fNode[19] + fNode[22] + fNode[23] + fNode[25])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17] + fNode[20]
        + fNode[21] + fNode[24] + fNode[26]) + 0.5*fzVar) * invRho;
    #endif // !D3Q27

    #ifdef SCALAR_TRANSPORT
        dfloat q = g_gpu_source_term();
        #ifdef gD3Q7
            const dfloat gVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4]
                + gNode[5] + gNode[6] + G_TT_OMEGA * 0.5 * q;
        #endif
        #ifdef gD3Q19
            const dfloat gVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] 
                + gNode[5] + gNode[6]+ gNode[7] + gNode[8] + gNode[9] + gNode[10] 
                + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16]
                + gNode[17] + gNode[18] + G_TT_OMEGA * 0.5 * q;
        #endif
    #endif

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
    const dfloat cs2 = 1.0/3.0;

    // Calculate momNeq(alfa, beta)
    // momNeqAB = pops - popsEquilibrium
    #ifdef D3Q19
    const dfloat sumPopXX = fNode[1] + fNode[2] + fNode[7] + fNode[8] + fNode[9] 
            + fNode[10] + fNode[13] + fNode[14] + fNode[15] + fNode[16];
    const dfloat sumPopYY = fNode[3] + fNode[4] + fNode[7] + fNode[8] + fNode[11]
            + fNode[12] + fNode[13] + fNode[14] + fNode[17] + fNode[18];
    const dfloat sumPopZZ = fNode[5] + fNode[6] + fNode[9] + fNode[10] + fNode[11]
            + fNode[12] + fNode[15] + fNode[16] + fNode[17] + fNode[18];
    const dfloat sumPopXY = fNode[7] + fNode[8] - fNode[13] - fNode[14];
    const dfloat sumPopXZ = fNode[9] + fNode[10] - fNode[15] - fNode[16];
    const dfloat sumPopYZ = fNode[11] + fNode[12] - fNode[17] - fNode[18];
    
    const dfloat momNeqXX = sumPopXX - rhoVar*(uxVar*uxVar + cs2);
    const dfloat momNeqYY = sumPopYY - rhoVar*(uyVar*uyVar + cs2); 
    const dfloat momNeqZZ = sumPopZZ - rhoVar*(uzVar*uzVar + cs2);
    const dfloat momNeqXYt2 = (sumPopXY - rhoVar*uxVar*uyVar) * 2;
    const dfloat momNeqXZt2 = (sumPopXZ - rhoVar*uxVar*uzVar) * 2;
    const dfloat momNeqYZt2 = (sumPopYZ - rhoVar*uyVar*uzVar) * 2;
    #endif // !D3Q19 
    #ifdef D3Q27
    // ERROR: DEPRECATED FOR NON NEWTONIAN!!!!!!!!
    const dfloat aux = (fNode[19] + fNode[20] + fNode[21] + fNode[22] + fNode[23]
            + fNode[24] + fNode[25] + fNode[26]) - 
            (8*rhoW3*(ux3ux3d2 + uy3uy3d2 + uz3uz3d2));
    const dfloat momNeqXX = (fNode[1] + fNode[2] + fNode[7] + fNode[8] + fNode[9] 
            + fNode[10] + fNode[13] + fNode[14] + fNode[15] + fNode[16] + aux) -
            (2*rhoW1*(p1_muu15 + ux3ux3d2) + 
             4*rhoW2*(2*p1_muu15 + 2*ux3ux3d2 + uy3uy3d2 + uz3uz3d2));
    const dfloat momNeqYY = (fNode[3] + fNode[4] + fNode[7] + fNode[8] + fNode[11]
            + fNode[12] + fNode[13] + fNode[14] + fNode[17] + fNode[18] + aux) -
            (2*rhoW1*(p1_muu15 + uy3uy3d2) + 
             4*rhoW2*(ux3ux3d2 + 2*uy3uy3d2 + uz3uz3d2)); 
    const dfloat momNeqZZ = (fNode[5] + fNode[6] + fNode[9] + fNode[10] + fNode[11]
            + fNode[12] + fNode[15] + fNode[16] + fNode[17] + fNode[18] + aux) -
            (2*rhoW1*(p1_muu15 + uz3uz3d2) + 
             4*rhoW2*(ux3ux3d2 + uy3uy3d2 + 2*uz3uz3d2));
    const dfloat momNeqXYt2 = ((fNode[7] + fNode[8] - fNode[13] - fNode[14] + fNode[19]
            + fNode[20] + fNode[21] + fNode[22] - fNode[23] - fNode[24] - fNode[25]
            - fNode[26]) - 
            (4*rhoW2*(ux3uy3) + 8*rhoW3*(ux3uy3))) * 2;
    const dfloat momNeqXZt2 = ((fNode[9] + fNode[10] - fNode[15] - fNode[16] + fNode[19]
            + fNode[20] - fNode[21] - fNode[22] + fNode[23] + fNode[24] - fNode[25]
            - fNode[26]) - 
            (4*rhoW2*(ux3uz3) + 8*rhoW3*(ux3uz3))) * 2;
    const dfloat momNeqYZt2 = ((fNode[11] + fNode[12] - fNode[17] - fNode[18] + fNode[19]
            + fNode[20] - fNode[21] - fNode[22] - fNode[23] - fNode[24] + fNode[25]
            + fNode[26]) - 
            (4*rhoW2*(uy3uz3) + 8*rhoW3*(uy3uz3))) * 2;
    #endif // !D3Q27

    #ifdef NON_NEWTONIAN_FLUID
    const dfloat uFxxd2 = uxVar*fxVar; // d2 = uFxx Divided by two
    const dfloat uFyyd2 = uyVar*fyVar;
    const dfloat uFzzd2 = uzVar*fzVar;
    const dfloat uFxyd2 = (uxVar*fyVar + uyVar*fxVar) / 2;
    const dfloat uFxzd2 = (uxVar*fzVar + uzVar*fxVar) / 2;
    const dfloat uFyzd2 = (uyVar*fzVar + uzVar*fyVar) / 2;
    // Related to stress tensor magnitude.
    // StressMag = (1-omega/2) * auxStressMag
    const dfloat auxStressMag = sqrt(0.5 * (
        (momNeqXX + uFxxd2) * (momNeqXX + uFxxd2) +
        (momNeqYY + uFyyd2) * (momNeqYY + uFyyd2) +
        (momNeqZZ + uFzzd2) * (momNeqZZ + uFzzd2) +
        2 * ((momNeqXYt2/2 + uFxyd2) * (momNeqXYt2/2 + uFxyd2) +
        (momNeqXZt2/2 + uFxzd2) * (momNeqXZt2/2 + uFxzd2) + 
        (momNeqYZt2/2 + uFyzd2) * (momNeqYZt2/2 + uFyzd2))));

    // Update omega (related to fluid viscosity) locally for non newtonian fluid
    #if defined(BINGHAM)
    const dfloat omegaVar = calcOmega(OMEGA_P, auxStressMag);
    #endif

    #ifdef OMEGA_LAST_STEP
    dfloat omegaVar = macr.omega[idxScalar(x, y, z)];
    omegaVar = calcOmega(omegaVar, auxStressMag);
    macr.omega[idxScalar(x, y, z)] = omegaVar;
    #endif

    #else
    const dfloat omegaVar = OMEGA;
    #endif

    const dfloat t_omegaVar = 1 - omegaVar;
    const dfloat tt_omegaVar = 1 - 0.5*omegaVar;

    if (save)
    {
        macr.rho[idx] = rhoVar;
        macr.u.x[idx] = uxVar;
        macr.u.y[idx] = uyVar;
        macr.u.z[idx] = uzVar;
        // Only Bingham does not save local omega
        #if !defined(OMEGA_LAST_STEP) && defined(NON_NEWTONIAN_FLUID)
        macr.omega[idx] = omegaVar;
        #endif

        #ifdef SCALAR_TRANSPORT
        macr.G[idx] = gVar;
        #endif
    }

    // Calculate regularization terms 
    // terms[i] = Q[i, alfa, beta]*pi[i, alfa, beta] - c[i, alfa]*F[alfa]/3
    // terms[0] -> population 0
    // terms[1] -> population 1
    // terms[2] -> population 2
    // terms[3] -> population 3
    // terms[4] -> population 4
    terms[0] = -momNeqXX/3 - momNeqYY/3 - momNeqZZ/3;
    terms[1] = terms[0] + (-fxVar_D3 + momNeqXX);
    terms[2] = terms[0] + ( fxVar_D3 + momNeqXX);
    terms[3] = terms[0] + (-fyVar_D3 + momNeqYY);
    terms[4] = terms[0] + ( fyVar_D3 + momNeqYY);
    #ifdef D3Q27
    // terms[5] -> population 7
    // terms[6] -> population 8
    // terms[7] -> population 9
    // terms[8] -> population 10
    // terms[9] -> population 11
    // terms[10] -> population 12
    terms[5] = terms[1] + (-fyVar_D3 + momNeqXYt2 + momNeqYY);
    terms[6] = terms[2] + ( fyVar_D3 + momNeqXYt2 + momNeqYY);
    terms[7] = terms[1] + (-fzVar_D3 + momNeqXZt2 + momNeqZZ);
    terms[8] = terms[2] + ( fzVar_D3 + momNeqXZt2 + momNeqZZ);
    terms[9] = terms[3] + (-fzVar_D3 + momNeqYZt2 + momNeqZZ);
    terms[10] = terms[4] + ( fzVar_D3 + momNeqYZt2 + momNeqZZ);
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
    fNode[5] = multiplyTerm*(terms[0] + (-fzVar_D3 + momNeqZZ));
    fNode[6] = multiplyTerm*(terms[0] + ( fzVar_D3 + momNeqZZ));
    multiplyTerm = W2t9d2;
    fNode[7] = multiplyTerm*(terms[1] + (-fyVar_D3 + momNeqXYt2 + momNeqYY));
    fNode[8] = multiplyTerm*(terms[2] + ( fyVar_D3 + momNeqXYt2 + momNeqYY));
    fNode[9] = multiplyTerm*(terms[1] + (-fzVar_D3 + momNeqXZt2 + momNeqZZ));
    fNode[10] = multiplyTerm*(terms[2] + ( fzVar_D3 + momNeqXZt2 + momNeqZZ));
    fNode[11] = multiplyTerm*(terms[3] + (-fzVar_D3 + momNeqYZt2 + momNeqZZ));
    fNode[12] = multiplyTerm*(terms[4] + ( fzVar_D3 + momNeqYZt2 + momNeqZZ));
    fNode[13] = multiplyTerm*(terms[1] + ( fyVar_D3 - momNeqXYt2 + momNeqYY));
    fNode[14] = multiplyTerm*(terms[2] + (-fyVar_D3 - momNeqXYt2 + momNeqYY));
    fNode[15] = multiplyTerm*(terms[1] + ( fzVar_D3 - momNeqXZt2 + momNeqZZ));
    fNode[16] = multiplyTerm*(terms[2] + (-fzVar_D3 - momNeqXZt2 + momNeqZZ));
    fNode[17] = multiplyTerm*(terms[3] + ( fzVar_D3 - momNeqYZt2 + momNeqZZ));
    fNode[18] = multiplyTerm*(terms[4] + (-fzVar_D3 - momNeqYZt2 + momNeqZZ));
    #ifdef D3Q27
    multiplyTerm = W3t9d2;
    fNode[19] = multiplyTerm*(terms[5] + (-fzVar_D3 + momNeqXZt2 + momNeqYZt2 + momNeqZZ));
    fNode[20] = multiplyTerm*(terms[6] + ( fzVar_D3 + momNeqXZt2 + momNeqYZt2 + momNeqZZ));
    fNode[21] = multiplyTerm*(terms[5] + ( fzVar_D3 - momNeqXZt2 - momNeqYZt2 + momNeqZZ));
    fNode[22] = multiplyTerm*(terms[6] + (-fzVar_D3 - momNeqXZt2 - momNeqYZt2 + momNeqZZ));
    fNode[23] = multiplyTerm*(terms[7] + ( fyVar_D3 - momNeqXYt2 + momNeqYY - momNeqYZt2));
    fNode[24] = multiplyTerm*(terms[8] + (-fyVar_D3 - momNeqXYt2 + momNeqYY - momNeqYZt2));
    fNode[25] = multiplyTerm*(terms[9] + ( fxVar_D3 + momNeqXX - momNeqXYt2 - momNeqXZt2));
    fNode[26] = multiplyTerm*(terms[10] + (-fxVar_D3 + momNeqXX - momNeqXYt2 - momNeqXZt2));
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
        fNode[i] *= t_omegaVar;

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
    terms[0] = - fxVar*ux3 - fyVar*uy3 - fzVar*uz3;
    terms[1] = terms[0] + (fxVar*( 3*ux3 + 3));
    terms[2] = terms[0] + (fyVar*( 3*uy3 + 3));
    terms[3] = terms[1] + (fxVar*( 3*uy3) + fyVar*( 3*ux3 + 3*uy3 + 3));
    terms[4] = terms[1] + (fxVar*( 3*uz3) + fzVar*( 3*ux3 + 3*uz3 + 3));
    terms[5] = terms[2] + (fyVar*( 3*uz3) + fzVar*( 3*uy3 + 3*uz3 + 3));
    #ifdef D3Q27
    // term[6] -> population 19
    terms[6] = terms[3] + (fxVar*( 3*uz3) + fyVar*( 3*uz3) + fzVar*( 3*ux3 + 3*uy3 + 3*uz3 + 3));
    #endif

    // fNode += TT_OMEGA * force
    multiplyTerm = W0*tt_omegaVar;
    fNode[0] += multiplyTerm*terms[0];
    multiplyTerm = W1*tt_omegaVar;
    fNode[1] += multiplyTerm*terms[1];
    fNode[2] += multiplyTerm*(terms[1] + (fxVar*(-6)));
    fNode[3] += multiplyTerm*terms[2];
    fNode[4] += multiplyTerm*(terms[2] + (fyVar*(-6)));
    auxTerm = terms[0] + (fzVar*( 3*uz3 + 3));
    fNode[5] += multiplyTerm*auxTerm;
    fNode[6] += multiplyTerm*(auxTerm + (fzVar*(-6)));
    multiplyTerm = W2*tt_omegaVar;
    fNode[7] += multiplyTerm*terms[3];
    fNode[8] += multiplyTerm*(terms[3] + (fxVar*(-6) + fyVar*(-6)));
    fNode[9] += multiplyTerm*terms[4];
    fNode[10] += multiplyTerm*(terms[4] + (fxVar*(-6) + fzVar*(-6)));
    fNode[11] += multiplyTerm*(terms[5]);
    fNode[12] += multiplyTerm*(terms[5] + (fyVar*(-6) + fzVar*(-6)));
    auxTerm = terms[3] + (fxVar*(-6*uy3) + fyVar*(-6*ux3 - 6));
    fNode[13] += multiplyTerm*(auxTerm);
    fNode[14] += multiplyTerm*(auxTerm + (fxVar*(-6) + fyVar*( 6)));
    auxTerm = terms[4] + (fxVar*(-6*uz3) + fzVar*(-6*ux3 - 6));
    fNode[15] += multiplyTerm*auxTerm;
    fNode[16] += multiplyTerm*(auxTerm + (fxVar*(-6) + fzVar*( 6)));
    auxTerm = terms[5] + (fyVar*(-6*uz3) + fzVar*(-6*uy3 - 6));
    fNode[17] += multiplyTerm*auxTerm;
    fNode[18] += multiplyTerm*(auxTerm + (fyVar*(-6) + fzVar*( 6)));
    #ifdef D3Q27
    multiplyTerm = W3*tt_omegaVar;
    fNode[19] += multiplyTerm*terms[6];
    fNode[20] += multiplyTerm*(terms[6] + (fxVar*(-6) + fyVar*(-6) + fzVar*(-6)));
    auxTerm = terms[6] + (fxVar*(-6*uz3) + fyVar*(-6*uz3) + fzVar*(-6*ux3 - 6*uy3 - 6));
    fNode[21] += multiplyTerm*auxTerm;
    fNode[22] += multiplyTerm*(auxTerm + (fxVar*(-6) + fyVar*(-6) + fzVar*( 6)));
    auxTerm = terms[6] + (fxVar*(-6*uy3) + fyVar*(-6*ux3 - 6*uz3 - 6) + fzVar*(-6*uy3));
    fNode[23] += multiplyTerm*auxTerm;
    fNode[24] += multiplyTerm*(auxTerm + (fxVar*(-6) + fyVar*( 6) + fzVar*(-6)));
    auxTerm = terms[6] + (fxVar*(-6*uy3 - 6*uz3 - 6) + fyVar*(-6*ux3) + fzVar*(-6*ux3));
    fNode[25] += multiplyTerm*auxTerm;
    fNode[26] += multiplyTerm*(auxTerm + (fxVar*( 6) + fyVar*(-6) + fzVar*(-6)));
    #endif

    // Save post collision populations of boundary conditions nodes
    if(mapBC[idx].getSavePostCol())  
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

    
    #ifdef SCALAR_TRANSPORT
        #ifdef gD3Q7
        geq[0] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,0);
        geq[1] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,1);
        geq[2] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,2);
        geq[3] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,3);
        geq[4] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,4);
        geq[5] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,5);
        geq[6] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,6);
        #endif
        #ifdef gD3Q19
        geq[0] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,0);
        geq[1] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,1);
        geq[2] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,2);
        geq[3] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,3);
        geq[4] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,4);
        geq[5] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,5);
        geq[6] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,6);
        geq[7] =   gpu_g_eq(gVar,uxVar,uyVar,uzVar,7);
        geq[8] =   gpu_g_eq(gVar,uxVar,uyVar,uzVar,8);
        geq[9] =   gpu_g_eq(gVar,uxVar,uyVar,uzVar,9);
        geq[10] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,10);
        geq[11] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,11);
        geq[12] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,12);
        geq[13] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,13);
        geq[14] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,14);
        geq[15] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,15);
        geq[16] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,16);
        geq[17] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,17);
        geq[18] =  gpu_g_eq(gVar,uxVar,uyVar,uzVar,18);
        #endif
        
        #ifdef gD3Q7
        gNode[0] =  G_T_OMEGA * gNode[0] +  G_OMEGA * geq[0] +  G_TT_OMEGA * gW0 * q;
        gNode[1] =  G_T_OMEGA * gNode[1] +  G_OMEGA * geq[1] +  G_TT_OMEGA * gW1 * q;
        gNode[2] =  G_T_OMEGA * gNode[2] +  G_OMEGA * geq[2] +  G_TT_OMEGA * gW1 * q;
        gNode[3] =  G_T_OMEGA * gNode[3] +  G_OMEGA * geq[3] +  G_TT_OMEGA * gW1 * q;
        gNode[4] =  G_T_OMEGA * gNode[4] +  G_OMEGA * geq[4] +  G_TT_OMEGA * gW1 * q;
        gNode[5] =  G_T_OMEGA * gNode[5] +  G_OMEGA * geq[5] +  G_TT_OMEGA * gW1 * q;
        gNode[6] =  G_T_OMEGA * gNode[6] +  G_OMEGA * geq[6] +  G_TT_OMEGA * gW1 * q;
        #endif
        #ifdef gD3Q19
        gNode[0] =  G_T_OMEGA * gNode[0] +  G_OMEGA * geq[0] +  G_TT_OMEGA * gW0 * q;
        gNode[1] =  G_T_OMEGA * gNode[1] +  G_OMEGA * geq[1] +  G_TT_OMEGA * gW1 * q;
        gNode[2] =  G_T_OMEGA * gNode[2] +  G_OMEGA * geq[2] +  G_TT_OMEGA * gW1 * q;
        gNode[3] =  G_T_OMEGA * gNode[3] +  G_OMEGA * geq[3] +  G_TT_OMEGA * gW1 * q;
        gNode[4] =  G_T_OMEGA * gNode[4] +  G_OMEGA * geq[4] +  G_TT_OMEGA * gW1 * q;
        gNode[5] =  G_T_OMEGA * gNode[5] +  G_OMEGA * geq[5] +  G_TT_OMEGA * gW1 * q;
        gNode[6] =  G_T_OMEGA * gNode[6] +  G_OMEGA * geq[6] +  G_TT_OMEGA * gW1 * q;
        gNode[7] =  G_T_OMEGA * gNode[7] +  G_OMEGA * geq[7] +  G_TT_OMEGA * gW2 * q;
        gNode[8] =  G_T_OMEGA * gNode[8] +  G_OMEGA * geq[8] +  G_TT_OMEGA * gW2 * q;
        gNode[9] =  G_T_OMEGA * gNode[9] +  G_OMEGA * geq[9] +  G_TT_OMEGA * gW2 * q;
        gNode[10] = G_T_OMEGA * gNode[10] + G_OMEGA * geq[10] + G_TT_OMEGA * gW2 * q;
        gNode[11] = G_T_OMEGA * gNode[11] + G_OMEGA * geq[11] + G_TT_OMEGA * gW2 * q;
        gNode[12] = G_T_OMEGA * gNode[12] + G_OMEGA * geq[12] + G_TT_OMEGA * gW2 * q;
        gNode[13] = G_T_OMEGA * gNode[13] + G_OMEGA * geq[13] + G_TT_OMEGA * gW2 * q;
        gNode[14] = G_T_OMEGA * gNode[14] + G_OMEGA * geq[14] + G_TT_OMEGA * gW2 * q;
        gNode[15] = G_T_OMEGA * gNode[15] + G_OMEGA * geq[15] + G_TT_OMEGA * gW2 * q;
        gNode[16] = G_T_OMEGA * gNode[16] + G_OMEGA * geq[16] + G_TT_OMEGA * gW2 * q;
        gNode[17] = G_T_OMEGA * gNode[17] + G_OMEGA * geq[17] + G_TT_OMEGA * gW2 * q;
        gNode[18] = G_T_OMEGA * gNode[18] + G_OMEGA * geq[18] + G_TT_OMEGA * gW2 * q;
        #endif
        
        //streaming
        #ifdef gD3Q7
        gPopAux[idxPop(x, y, z, 0)]   = gNode[0];
        gPopAux[idxPop(xp1, y, z, 1)] = gNode[1];
        gPopAux[idxPop(xm1, y, z, 2)] = gNode[2];
        gPopAux[idxPop(x, yp1, z, 3)] = gNode[3];
        gPopAux[idxPop(x, ym1, z, 4)] = gNode[4];
        gPopAux[idxPop(x, y, zp1, 5)] = gNode[5];
        gPopAux[idxPop(x, y, zm1, 6)] = gNode[6];
        #endif
        #ifdef gD3Q19
        gPopAux[idxPop(x, y, z, 0)]   =    gNode[0];
        gPopAux[idxPop(xp1, y, z, 1)] =    gNode[1];
        gPopAux[idxPop(xm1, y, z, 2)] =    gNode[2];
        gPopAux[idxPop(x, yp1, z, 3)] =    gNode[3];
        gPopAux[idxPop(x, ym1, z, 4)] =    gNode[4];
        gPopAux[idxPop(x, y, zp1, 5)] =    gNode[5];
        gPopAux[idxPop(x, y, zm1, 6)] =    gNode[6];
        gPopAux[idxPop(xp1, yp1, z, 7)]  = gNode[7];
        gPopAux[idxPop(xm1, ym1, z, 8)]  = gNode[8];
        gPopAux[idxPop(xp1, y, zp1, 9)]  = gNode[9];
        gPopAux[idxPop(xm1, y, zm1, 10)] = gNode[10];
        gPopAux[idxPop(x, yp1, zp1, 11)] = gNode[11];
        gPopAux[idxPop(x, ym1, zm1, 12)] = gNode[12];
        gPopAux[idxPop(xp1, ym1, z, 13)] = gNode[13];
        gPopAux[idxPop(xm1, yp1, z, 14)] = gNode[14];
        gPopAux[idxPop(xp1, y, zm1, 15)] = gNode[15];
        gPopAux[idxPop(xm1, y, zp1, 16)] = gNode[16];
        gPopAux[idxPop(x, yp1, zm1, 17)] = gNode[17];
        gPopAux[idxPop(x, ym1, zp1, 18)] = gNode[18];
        #endif
    
    #endif 
    //SCALAR_TRANSPORT
}


__global__
void gpuUpdateMacr(
    Populations pop,
    #ifdef SCALAR_TRANSPORT
    GPopulations const gPop,
    #endif
    Macroscopics macr)
{
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t idx_s = idxScalar(x, y, z);
    // load populations
    dfloat fNode[Q];
    for (unsigned char i = 0; i < Q; i++)
        fNode[i] = pop.pop[idxPop(x, y, z, i)];

    #ifdef SCALAR_TRANSPORT
        dfloat gNode[GQ];
        for (char i = 0; i < GQ; i++)
            gNode[i] = gPop.gPop[idxPop(x, y, z, i)];
    #endif

    #ifdef IBM
    const dfloat fxVar = macr.f.x[idx_s];
    const dfloat fyVar = macr.f.y[idx_s];
    const dfloat fzVar = macr.f.z[idx_s];
    #else
    const dfloat fxVar = FX;
    const dfloat fyVar = FY;
    const dfloat fzVar = FZ;
    #endif

    // calc for macroscopics
    // rho = sum(f[i])
    // ux = sum(f[i]*cx[i] + Fx/2) / rho
    // uy = sum(f[i]*cy[i] + Fy/2) / rho
    // uz = sum(f[i]*cz[i] + Fz/2) / rho
    #ifdef D3Q19
    const dfloat rhoVar = fNode[0] + fNode[1] + fNode[2] + fNode[3] + fNode[4] 
        + fNode[5] + fNode[6] + fNode[7] + fNode[8] + fNode[9] + fNode[10] 
        + fNode[11] + fNode[12] + fNode[13] + fNode[14] + fNode[15] + fNode[16] 
        + fNode[17] + fNode[18];
    const dfloat invRho = 1/rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15])
        - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16]) + 0.5*fxVar) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18]) + 0.5*fyVar) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17]) + 0.5*fzVar) * invRho;
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
        + fNode[22] + fNode[24] + fNode[25]) + 0.5*fxVar) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17]
        + fNode[19] + fNode[21] + fNode[24] + fNode[25])
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18] + fNode[20]
        + fNode[22] + fNode[23] + fNode[26]) + 0.5*fyVar) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18]
        + fNode[19] + fNode[22] + fNode[23] + fNode[25])
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17] + fNode[20]
        + fNode[21] + fNode[24] + fNode[26]) + 0.5*fzVar) * invRho;
    #endif // !D3Q27
    macr.rho[idx_s] = rhoVar;
    macr.u.x[idx_s] = uxVar;
    macr.u.y[idx_s] = uyVar;
    macr.u.z[idx_s] = uzVar;

            
    #ifdef SCALAR_TRANSPORT
        #ifdef gD3Q7
            const dfloat gVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] 
                + gNode[5] + gNode[6];
        #endif
        #ifdef gD3Q19
            const dfloat gVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] 
                + gNode[5] + gNode[6]+ gNode[7] + gNode[8] + gNode[9] + gNode[10] 
                + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16]
                + gNode[17] + gNode[18];
        #endif
        macr.G[idx_s] = gVar;
    #endif
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
    dfloat* popPostCollNxt
    #ifdef SCALAR_TRANSPORT
        ,dfloat* gPopPostStreamBase,
        dfloat* gPopPostCollBase,
        dfloat* gPopPostStreamNxt,
        dfloat* gPopPostCollNxt
    #endif
)
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
    
    #ifdef SCALAR_TRANSPORT
        #ifdef gD3Q7    
        // pop[6] -> cz = 1; pop[5] -> cz = -1;
            gPopPostStreamBase[idxPop(x, y, 0, 5)] = gPopPostStreamNxt[idxPop(x, y, 0, 5)];
            gPopPostStreamNxt[idxPop(x, y, zMax, 6)] = gPopPostStreamBase[idxPop(x, y, zMax, 6)];
        #endif
        #ifdef gD3Q19
        // pop[6] -> cz = 1; pop[5] -> cz = -1;
            gPopPostStreamBase[idxPop(x, y, 0, 5)] = gPopPostStreamNxt[idxPop(x, y, 0, 5)];
            gPopPostStreamNxt[idxPop(x, y, zMax, 6)] = gPopPostStreamBase[idxPop(x, y, zMax, 6)];

            // pop[9] -> cz = 1; pop[10] -> cz = -1;
            gPopPostStreamBase[idxPop(x, y, 0, 9)] = gPopPostStreamNxt[idxPop(x, y, 0, 9)];
            gPopPostStreamNxt[idxPop(x, y, zMax, 10)] = gPopPostStreamBase[idxPop(x, y, zMax, 10)];
            
            // pop[11] -> cz = 1; pop[12] -> cz = -1;
            gPopPostStreamBase[idxPop(x, y, 0, 11)] = gPopPostStreamNxt[idxPop(x, y, 0, 11)];
            gPopPostStreamNxt[idxPop(x, y, zMax, 12)] = gPopPostStreamBase[idxPop(x, y, zMax, 12)];
            
            // pop[15] -> cz = 1; pop[16] -> cz = -1;
            gPopPostStreamBase[idxPop(x, y, 0, 16)] = gPopPostStreamNxt[idxPop(x, y, 0, 16)];
            gPopPostStreamNxt[idxPop(x, y, zMax, 15)] = gPopPostStreamBase[idxPop(x, y, zMax, 15)];

            // pop[18] -> cz = 1; pop[17] -> cz = -1;
            gPopPostStreamBase[idxPop(x, y, 0, 18)] =   gPopPostStreamNxt[idxPop(x, y, 0, 18)];
            gPopPostStreamNxt[idxPop(x, y, zMax, 17)] = gPopPostStreamBase[idxPop(x, y, zMax, 17)];
        #endif
    #endif
}