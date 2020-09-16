#include "ibm.h"


__host__
void immersedBoundaryMethod(
    Particle* particles,
    Macroscopics* const macr,
    Populations* const pop,
    dim3 gridLBM,
    dim3 threadsLBM,
    unsigned int gridIBM,
    unsigned int threadsIBM,
    cudaStream_t* stream)
{
    //dim3 grid1(((totalIbmNodes % nThreads) ? (totalIbmNodes / nThreads + 1) : (totalIbmNodes / nThreads)), 1, 1);
    //dim3 threads1(nThreads, 1, 1);

    //dim3 grid2(((NX % nThreads) ? (NX / nThreads + 1) : (NX / nThreads)), NY, NZ);
    //dim3 threads2(nThreads, 1, 1);

    // TODO: Update it to multi GPU
    // Update macroscopics post streaming and reset forces
    gpuUpdateMacrResetForces<<<gridLBM, threadsLBM, 0, stream[0]>>>(&pop[0], &macr[0]); 

    checkCudaErrors(cudaStreamSynchronize(stream[0]));

    for (int i = 0; i < IBM_MAX_INTERATION; i++) {
        // TODO: UPDATE PARTICLE VELOCITY AND ROTATION
        // updateParticleVelocity

        // TODO: check parameters
        gpuForceInterpolationSpread<<<gridIBM, threadsIBM, 0, stream[0]>>>();
        checkCudaErrors(cudaStreamSynchronize(stream[0]));
    }


    particleForce();
    // TODO: make it in a kernel
    particleMovement();
    checkCudaErrors(cudaStreamSynchronize(stream[0]));
}


__global__
void gpuForceInterpolationSpread(
    ParticleNodeSoA* const particlesNodes,
    ParticleCenter* const particleCenters,
    Macroscopics* const macr)
{
    // TODO: update this
    const unsigned short int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i > particleNodes->numNodes)
        return;

    dfloat aux; //aux variable for many things
    size_t idx; // index for many things

    // IBM nodes coordinates
    // const dfloat xIBM = IBMmacr->Xibm[i];
    // const dfloat yIBM = IBMmacr->Yibm[i];
    // const dfloat zIBM = IBMmacr->Zibm[i];

    const dfloat x_ibm = particlesNodes->pos.x[i];
    const dfloat y_ibm = particlesNodes->pos.y[i];
    const dfloat z_ibm = particlesNodes->pos.z[i];

    // Stencil distance
    #if defined STENCIL_2
    const int pdist = 2;
    #endif

    #if defined STENCIL_4
    const int pdist = 4;
    #endif

    // LBM nearest position
    const int x_near = (int)(x_ibm + 0.5);
    const int y_near = (int)(y_ibm + 0.5);
    const int z_near = (int)(z_ibm + 0.5);

    // Minimum number of xyz for LBM interpolation
    const unsigned int x_min = ((x_near-pdist) < 0)? 0 : x_near-pdist; 
    const unsigned int y_min = ((y_near-pdist) < 0)? 0 : y_near-pdist;
    const unsigned int z_min = ((z_near-pdist) < 0)? 0 : z_near-pdist;

    // Maximum number of xyz for LBM interpolation, excluding last
    // (e.g. NX goes just until NX-1)
    const unsigned int x_max = ((x_near+pdist) > NX)? NX : x_near+pdist;
    const unsigned int y_max = ((y_near+pdist) > NY)? NY : y_near+pdist;
    const unsigned int z_max = ((z_near+pdist) > NZ)? NZ : z_near+pdist;

    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

    // Index of particle center of this particle node
    idx = particlesNodes->particleCenterIdx[i];

    // Load position, velocity and rotation velocity
    dfloat x_pc = particleCenters[idx].pos.x;
    dfloat y_pc = particleCenters[idx].pos.y;
    dfloat z_pc = particleCenters[idx].pos.z;
    
    dfloat vx_pc = particleCenters[idx].v.x;
    dfloat vy_pc = particleCenters[idx].v.y;
    dfloat vz_pc = particleCenters[idx].v.z;

    dfloat wx_pc = particleCenters[idx].w.x;
    dfloat wy_pc = particleCenters[idx].w.y;
    dfloat wz_pc = particleCenters[idx].w.z;

    // velocity on node, given the center velocity and rotation 
    // (i.e. no slip boundary condition velocity)
    dfloat ux_cal = vx_pc + (wy_pc * (zIBM - z_pc) - wz_pc * (yIBM - y_pc));
    dfloat uy_cal = vy_pc + (wz_pc * (xIBM - x_pc) - wx_pc * (zIBM - z_pc));
    dfloat uz_cal = vz_pc + (wx_pc * (yIBM - y_pc) - wy_pc * (xIBM - x_pc));

    //  Interpolation
    for (int z = z_min; z < z_max; z++)
    {
        for (int y = y_min; y < y_max; y++)
        {
            for (int x = x_min; x < x_max; x++)
            {
                idx = idxScalar(x, y, z);
                dfloat uxLBM = macr->ux[idx];
                dfloat uyLBM = macr->uy[idx];
                dfloat uzLBM = macr->uz[idx];

                // Dirac delta (kernel)
                aux = stencil(x - xIBM)*stencil(y - yIBM)*stencil(z - zIBM);

                uxVar += uxLBM * aux;
                uyVar += uyLBM * aux;
                uzVar += uzLBM * aux;
            }
        }
    }

    //particlesNodes->pos.x[i];
    particlesNodes->vel.x[i] = uxVar;
    particlesNodes->vel.y[i] = uyVar;
    particlesNodes->vel.z[i] = uzVar;

    // printf("\nUz Calc %lf...\n", uz_cal);
    // printf("\nUz Var %lf...\n", uzVar);

    // load FxIBM FyIBM FzIBM
    dfloat fxIBM = particlesNodes->f.x[i];
    dfloat fyIBM = particlesNodes->f.y[i];
    dfloat fzIBM = particlesNodes->f.z[i];

    aux = 2 * rhoVar * dA * drs;
    dfloat del_Fx = aux * (uxVar - ux_cal);
    dfloat del_Fy = aux * (uyVar - uy_cal);
    dfloat del_Fz = aux * (uzVar - uz_cal);

    fxIBM +=  del_fx;
    fyIBM +=  del_fy;
    fzIBM +=  del_fz;

    //  Spreading
    for (int z = z_min; z < z_max; z++)
    {
        for (int y = y_min; y < y_max; y++)
        {
            for (int x = x_min; x < x_max; x++)
            {
                idx = idxScalar(x, y, z);

                // Dirac delta (kernel)
                aux = stencil(x - xIBM)*stencil(y - yIBM)*stencil(z - zIBM);

                atomicAdd((double*)&(macr->fx[idx]), (double) - del_fx * aux);
                atomicAdd((double*)&(macr->fy[idx]), (double) - del_fy * aux);
                atomicAdd((double*)&(macr->fz[idx]), (double) - del_fz * aux);
            }
        }
    }

    // Update node force
    particlesNodes->f.x[i]; = fxIBM;
    particlesNodes->f.y[i]; = fyIBM;
    particlesNodes->f.z[i]; = fzIBM;

    // Update node delta forne
    particlesNodes->delta_f.x[i]; = del_fx;
    particlesNodes->delta_f.y[i]; = del_fy;
    particlesNodes->delta_f.z[i]; = del_fz;

    // Add node force to particle center
    idx = particlesNodes->particleCenterIdx[i];
    atomicAdd((double*)&(macr->fx[idx]), (double) - del_fx * aux);
    atomicAdd((double*)&(macr->fy[idx]), (double) - del_fy * aux);
    atomicAdd((double*)&(macr->fz[idx]), (double) - del_fz * aux);

}


__global__ 
void gpuUpdateMacrResetForces(Populations* pop, Macroscopics* macr)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    
    size_t idx = idxScalar(x, y, z);

    // Reset forces
    macr->fx[idx] = FX;
    macr->fy[idx] = FY;
    macr->fz[idx] = FZ;

    // load populations
    dfloat fNode[Q];
    for (unsigned char i = 0; i < Q; i++)
        fNode[i] = pop.pop[idxPop(x, y, z, i)];

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

    macr->rho[idx] = rhoVar;
    macr->ux[idx] = uxVar;
    macr->uy[idx] = uyVar;
    macr->uz[idx] = uzVar;
}


__global__
void gpuResetNodesForces(ParticleNodeSoA* const particleNodes){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= particleNodes->numNodes)
        return;

    dfloat3SoA* const force = particleNodes->f;
    dfloat3SoA* const delta_force = particleNodes->delta_f;

    force.x[idx] = 0;
    force.y[idx] = 0;
    force.z[idx] = 0;
    delta_force.x[idx] = 0;
    delta_force.y[idx] = 0;
    delta_force.z[idx] = 0;
}

__host__ 
void updateParticleCenterForce(
    Particle* particle,
    unsigned int numParticles)
{
    // TODO: move this to spread force kernel, except collision
    /* Old arguments
    unsigned int totalIbmNodes,
    unsigned int* qtdNos,
    ParticleCenter* particleCenter,
    dfloat3* nodePosition,
    dfloat3* nodeCumulativeForce,
    dfloat* nodeSurface,
    */

    //particle dynamics
    int n, p;
    dfloat fx = 0, fy = 0, fz = 0;
    dfloat mx = 0, my = 0, mz = 0;
    dfloat r_x = 0, r_y = 0, r_z = 0;
    dfloat3 force;

    for (p = 0; p < NUM_PARTICLES; p++) {
        fx = 0.0;
        fy = 0.0;
        fz = 0.0;
        mx = 0.0;
        my = 0.0;
        mz = 0.0;
        // determine forces
        for (n = qtdNos[p]; n < qtdNos[p + 1]; n++) {
            // TODO: update this
            r_x = nodePosition[n].x - particleCenter[p].pos.x;
            r_y = nodePosition[n].y - particleCenter[p].pos.y;
            r_z = nodePosition[n].z - particleCenter[p].pos.z;

            fx += (nodeCumulativeForce[n].x * nodeSurface[n]);
            fy += (nodeCumulativeForce[n].y * nodeSurface[n]);
            fz += (nodeCumulativeForce[n].z * nodeSurface[n]);

            mx += r_y * (nodeCumulativeForce[n].z * nodeSurface[n]) - r_z * (nodeCumulativeForce[n].y * nodeSurface[n]);
            my += r_z * (nodeCumulativeForce[n].x * nodeSurface[n]) - r_x * (nodeCumulativeForce[n].z * nodeSurface[n]);
            mz += r_x * (nodeCumulativeForce[n].y * nodeSurface[n]) - r_y * (nodeCumulativeForce[n].x * nodeSurface[n]);
        }

        force = particleCollisionSoft(particleCenter, p);

 
        fx += force.x;
        fy += force.y;
        fz += force.z;

        particleCenter[p].f.x = fx;
        particleCenter[p].f.y = fy;
        particleCenter[p].f.z = fz;

        particleCenter[p].M.x = mx;
        particleCenter[p].M.y = my;
        particleCenter[p].M.z = mz;
    }
}

__host__ 
dfloat3 particleCollisionSoft(
    ParticleCenter* particleCenter,
    int particleIndex)
{

}


__global__
void particleMovement(
    ParticleCenter* particleCenter)
{

}