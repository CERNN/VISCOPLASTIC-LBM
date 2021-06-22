#include "ibm.h"

#ifdef IBM


__host__
void immersedBoundaryMethod(
    ParticlesSoA particles,
    Macroscopics* __restrict__ macr,
    dfloat3SoA* __restrict__ velsAuxIBM,
    Populations* const __restrict__ pop,
    dim3 gridLBM,
    dim3 threadsLBM,
    unsigned int gridNodesIBM,
    unsigned int threadsNodesIBM,
    cudaStream_t streamLBM[N_GPUS],
    cudaStream_t streamIBM[N_GPUS],
    unsigned int step,
    ParticleEulerNodesUpdate* pEulerNodes
    )
{
    // TODO: Update kernels to multi GPU

    // Update particle center position and its old values
    gpuUpdateParticleOldValues<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
        particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

    // Size of shared memory to use for optimization in interpolation/spread
    // const unsigned int sharedMemInterpSpread = threadsNodesIBM * sizeof(dfloat3) * 2;

    #if IBM_EULER_OPTIMIZATION
        // Grid size for euler nodes update
        dim3 currGrid(pEulerNodes->currEulerNodes/64+(pEulerNodes->currEulerNodes%64? 1 : 0), 1, 1);
        if(pEulerNodes->currEulerNodes > 0){
            // Update macroscopics post boundary conditions and reset forces
            gpuUpdateMacrIBM<<<currGrid, 64, 0, streamLBM[0]>>>(pop[0], macr[0], velsAuxIBM[0], 
                pEulerNodes->eulerIndexesUpdate, pEulerNodes->currEulerNodes);
            checkCudaErrors(cudaStreamSynchronize(streamLBM[0]));
        }
    #else //!IBM_EULER_OPTIMIZATION
        // Update macroscopics post boundary conditions and reset forces
        gpuUpdateMacrIBM<<<gridLBM, threadsLBM, 0, streamLBM[0]>>>(pop[0], macr[0], velsAuxIBM[0]);
        checkCudaErrors(cudaStreamSynchronize(streamLBM[0]));
    #endif//IBM_EULER_OPTIMIZATION

    // Reset forces in all IBM nodes
    gpuResetNodesForces<<<gridNodesIBM, threadsNodesIBM, 0, streamIBM[0]>>>(particles.nodesSoA);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    // Calculate collision force between particles

    gpuParticlesCollision<<<GRID_PCOLLISION_IBM, THREADS_PCOLLISION_IBM, 0, streamIBM[0]>>>(particles.nodesSoA,particles.pCenterArray,step);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));   

    // First update particle velocity using body center force and constant forces
    gpuUpdateParticleCenterVelocityAndRotation <<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0] >>>(
        particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

    for (int i = 0; i < IBM_MAX_ITERATION; i++)
    {
        // Make the interpolation of LBM and spreading of IBM forces
        gpuForceInterpolationSpread<<<gridNodesIBM, threadsNodesIBM, 
            0, streamIBM[0]>>>(
            particles.nodesSoA, particles.pCenterArray, macr[0], velsAuxIBM[0]);
        checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

        // Update particle velocity using body center force and constant forces
        gpuUpdateParticleCenterVelocityAndRotation<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
            particles.pCenterArray);

        #if IBM_EULER_OPTIMIZATION
            if(pEulerNodes->currEulerNodes > 0){
                ibmEulerCopyVelocities<<<currGrid, 64, 0, streamLBM[0]>>>(macr[0].u, velsAuxIBM[0], 
                    pEulerNodes->eulerIndexesUpdate, pEulerNodes->currEulerNodes);
            }
        #else //!IBM_EULER_OPTIMIZATION
            copyFromArray<<<gridLBM, threadsLBM, 0, streamLBM[0]>>>(macr[0].u, velsAuxIBM[0]);
        #endif //IBM_EULER_OPTIMIZATION

        checkCudaErrors(cudaStreamSynchronize(streamLBM[0]));
        checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    }
    // Update particle center position and its old values
    gpuParticleMovement<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
        particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    // Update particle nodes positions
    gpuParticleNodeMovement<<<gridNodesIBM, threadsNodesIBM, 0, streamIBM[0]>>>(
        particles.nodesSoA, particles.pCenterArray);

    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void gpuForceInterpolationSpread(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES],
    Macroscopics const macr,
    dfloat3SoA velAuxIBM)
{
    // TODO: update atomic double add to use only if is double
    const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    // Shared memory to sum particles values to particle center
    // __shared__ dfloat3 sumPC[2][64];

    if (i >= particlesNodes.numNodes)
        return;

    dfloat aux, aux1; // aux variable for many things
    size_t idx; // index for many things

    const dfloat xIBM = particlesNodes.pos.x[i];
    const dfloat yIBM = particlesNodes.pos.y[i];
    const dfloat zIBM = particlesNodes.pos.z[i];
    const dfloat pos[3] = {xIBM, yIBM, zIBM};

    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];
    // Base position for every index (leftest in x)
    int posBase[3] = { 
        int(xIBM - P_DIST + 1 -(xIBM < 1.0)), 
        int(yIBM - P_DIST + 1 - (yIBM < 1.0)), 
        int(zIBM - P_DIST + 1 - (zIBM < 1.0)) 
    };
    // Maximum stencil index for each direction xyz ("index" to stop)
    const int maxIdx[3] = {
        #ifdef IBM_BC_X_WALL
            (posBase[0]+P_DIST*2-1) < (int)NX? P_DIST*2-1 : ((int)NX-1-posBase[0])
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            3
        #endif //IBM_BC_X_PERIODIC
        ,
        #ifdef IBM_BC_Y_WALL 
            (posBase[1]+P_DIST*2-1) < (int)NY? P_DIST*2-1 : ((int)NY-1-posBase[1])
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            3
        #endif //IBM_BC_Y_PERIODIC
        , 
        #ifdef IBM_BC_Z_WALL 
            (posBase[2]+P_DIST*2-1) < (int)NZ? P_DIST*2-1 : ((int)NZ-1-posBase[2])
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            3
        #endif //IBM_BC_Z_PERIODIC
    };
    // Minimum stencil index for each direction xyz ("index" to start)
    const int minIdx[3] = {
        #ifdef IBM_BC_X_WALL
            posBase[0] >= 0? 0 : -posBase[0]
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            0
        #endif //IBM_BC_X_PERIODIC
        ,
        #ifdef IBM_BC_Y_WALL 
            posBase[1] >= 0? 0 : -posBase[1]
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            0
        #endif //IBM_BC_Y_PERIODIC
        , 
        #ifdef IBM_BC_Z_WALL 
            posBase[2] >= 0? 0 : -posBase[2]
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            0
        #endif //IBM_BC_Z_PERIODIC
    };

    // Particle stencil out of the domain
    if(maxIdx[0] <= 0 || maxIdx[1] <= 0 || maxIdx[2] <= 0)
        return;
    // Particle stencil out of the domain
    if(minIdx[0] >= P_DIST*2 || minIdx[1] >= P_DIST*2 || minIdx[2] >= P_DIST*2)
        return;

    for(int i = 0; i < 3; i++){
        for(int j=minIdx[i]; j <= maxIdx[i]; j++){
            stencilVal[i][j] = stencil(posBase[i]+j-pos[i]);
        }
    }

    dfloat rhoVar = 0;
    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

    // Interpolation (zyx for memory locality)
    for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];
                // same as aux = stencil(x - xIBM) * stencil(y - yIBM) * stencil(z - zIBM);

                idx = idxScalar(
                    #ifdef IBM_BC_X_WALL
                        posBase[0]+xi
                    #endif //IBM_BC_X_WALL
                    #ifdef IBM_BC_X_PERIODIC
                        (posBase[0]+xi + NX)%NX
                    #endif //IBM_BC_X_PERIODIC
                    ,
                    #ifdef IBM_BC_Y_WALL 
                        posBase[1]+yj
                    #endif //IBM_BC_Y_WALL
                    #ifdef IBM_BC_Y_PERIODIC
                        (posBase[1]+yj+NY)%NY
                    #endif //IBM_BC_Y_PERIODIC
                    , 
                    #ifdef IBM_BC_Z_WALL 
                        posBase[2]+zk
                    #endif //IBM_BC_Z_WALL
                    #ifdef IBM_BC_Z_PERIODIC
                        (posBase[2]+zk+NZ)%NZ
                    #endif //IBM_BC_Z_PERIODIC
                );

                rhoVar += macr.rho[idx] * aux;
                uxVar += macr.u.x[idx] * aux;
                uyVar += macr.u.y[idx] * aux;
                uzVar += macr.u.z[idx] * aux;
            }
        }
    }

    // Index of particle center of this particle node
    idx = particlesNodes.particleCenterIdx[i];

    // Velocity on node given the particle velocity and rotation
    dfloat ux_calc = 0;
    dfloat uy_calc = 0;
    dfloat uz_calc = 0;

    // Load position of particle center
    const dfloat x_pc = particleCenters[idx].pos.x;
    const dfloat y_pc = particleCenters[idx].pos.y;
    const dfloat z_pc = particleCenters[idx].pos.z;

    // Calculate velocity on node if particle is movable
    if(particleCenters[idx].movable){
        // Load velocity and rotation velocity of particle center
        const dfloat vx_pc = particleCenters[idx].vel.x;
        const dfloat vy_pc = particleCenters[idx].vel.y;
        const dfloat vz_pc = particleCenters[idx].vel.z;

        const dfloat wx_pc = particleCenters[idx].w.x;
        const dfloat wy_pc = particleCenters[idx].w.y;
        const dfloat wz_pc = particleCenters[idx].w.z;

        // velocity on node, given the center velocity and rotation
        // (i.e. no slip boundary condition velocity)
        ux_calc = vx_pc + (wy_pc * (zIBM - z_pc) - wz_pc * (yIBM - y_pc));
        uy_calc = vy_pc + (wz_pc * (xIBM - x_pc) - wx_pc * (zIBM - z_pc));
        uz_calc = vz_pc + (wx_pc * (yIBM - y_pc) - wy_pc * (xIBM - x_pc));
    }

    const dfloat dA = particlesNodes.S[i];
    aux = 2 * rhoVar * dA * IBM_THICKNESS;

    dfloat3 deltaF;
    deltaF.x = aux * (uxVar - ux_calc);
    deltaF.y = aux * (uyVar - uy_calc);
    deltaF.z = aux * (uzVar - uz_calc);

    // Calculate IBM forces
    const dfloat fxIBM = particlesNodes.f.x[i] + deltaF.x;
    const dfloat fyIBM = particlesNodes.f.y[i] + deltaF.y;
    const dfloat fzIBM = particlesNodes.f.z[i] + deltaF.z;

    // Spreading (zyx for memory locality)
    for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];
                // same as aux = stencil(x - xIBM) * stencil(y - yIBM) * stencil(z - zIBM);

                idx = idxScalar(
                    #ifdef IBM_BC_X_WALL
                        posBase[0]+xi
                    #endif //IBM_BC_X_WALL
                    #ifdef IBM_BC_X_PERIODIC
                        (posBase[0]+xi + NX)%NX
                    #endif //IBM_BC_X_PERIODIC
                    ,
                    #ifdef IBM_BC_Y_WALL 
                        posBase[1]+yj
                    #endif //IBM_BC_Y_WALL
                    #ifdef IBM_BC_Y_PERIODIC
                        (posBase[1]+yj+NY)%NY
                    #endif //IBM_BC_Y_PERIODIC
                    , 
                    #ifdef IBM_BC_Z_WALL 
                        posBase[2]+zk
                    #endif //IBM_BC_Z_WALL
                    #ifdef IBM_BC_Z_PERIODIC
                        (posBase[2]+zk+NZ)%NZ
                    #endif //IBM_BC_Z_PERIODIC
                );

                atomicAdd(&(macr.f.x[idx]), -deltaF.x * aux);
                atomicAdd(&(macr.f.y[idx]), -deltaF.y * aux);
                atomicAdd(&(macr.f.z[idx]), -deltaF.z * aux);

                // Update velocities field
                const dfloat inv_rho = 1 / macr.rho[idx];
                atomicAdd(&(velAuxIBM.x[idx]), 0.5 * -deltaF.x * aux * inv_rho);
                atomicAdd(&(velAuxIBM.y[idx]), 0.5 * -deltaF.y * aux * inv_rho);
                atomicAdd(&(velAuxIBM.z[idx]), 0.5 * -deltaF.z * aux * inv_rho);
            }
        }
    }

    // Update node force
    particlesNodes.f.x[i] = fxIBM;
    particlesNodes.f.y[i] = fyIBM;
    particlesNodes.f.z[i] = fzIBM;

    // Update node delta force
    particlesNodes.deltaF.x[i] = deltaF.x;
    particlesNodes.deltaF.y[i] = deltaF.y;
    particlesNodes.deltaF.z[i] = deltaF.z;

    // Particle node delta momentum
    idx = particlesNodes.particleCenterIdx[i];

    const dfloat3 deltaMomentum = dfloat3(
        (yIBM - y_pc) * deltaF.z - (zIBM - z_pc) * deltaF.y,
        (zIBM - z_pc) * deltaF.x - (xIBM - x_pc) * deltaF.z,
        (xIBM - x_pc) * deltaF.y - (yIBM - y_pc) * deltaF.x
    );

    atomicAdd(&(particleCenters[idx].f.x), deltaF.x);
    atomicAdd(&(particleCenters[idx].f.y), deltaF.y);
    atomicAdd(&(particleCenters[idx].f.z), deltaF.z);

    atomicAdd(&(particleCenters[idx].M.x), deltaMomentum.x);
    atomicAdd(&(particleCenters[idx].M.y), deltaMomentum.y);
    atomicAdd(&(particleCenters[idx].M.z), deltaMomentum.z);
}

__global__
void gpuUpdateMacrIBM(Populations pop, Macroscopics macr, dfloat3SoA velAuxIBM
    #if IBM_EULER_OPTIMIZATION
    , size_t* eulerIdxsUpdate, unsigned int currEulerNodes
    #endif//IBM_EULER_OPTIMIZATION
)
{
    #if IBM_EULER_OPTIMIZATION
        unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
        if(j >= currEulerNodes)
            return;
        size_t idx = eulerIdxsUpdate[j];

    #else //!IBM_EULER_OPTIMIZATION
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        int y = threadIdx.y + blockDim.y * blockIdx.y;
        int z = threadIdx.z + blockDim.z * blockIdx.z;
        if (x >= NX || y >= NY || z >= NZ)
            return;

        size_t idx = idxScalar(x, y, z);
    #endif //IBM_EULER_OPTIMIZATION
    
    // load populations
    dfloat fNode[Q];
    for (unsigned char i = 0; i < Q; i++)
        // fNode[i] = pop.pop[idxPop(x, y, z, i)];
        fNode[i] = pop.pop[idx+i*NUMBER_LBM_NODES];

    // Already reseted in LBM kernel, when using IBM
    // macr.f.x[idx] = FX;
    // macr.f.y[idx] = FY;
    // macr.f.z[idx] = FZ;

    const dfloat3 f = dfloat3(FX, FY, FZ);

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
    const dfloat invRho = 1 / rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15]) 
    - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16]) + 0.5 * f.x) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17])
     - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18]) + 0.5 * f.y) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18])
     - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17]) + 0.5 * f.z) * invRho;
#endif // !D3Q19
#ifdef D3Q27
    const dfloat rhoVar = fNode[0] + fNode[1] + fNode[2] + fNode[3] + fNode[4] 
        + fNode[5] + fNode[6] + fNode[7] + fNode[8] + fNode[9] + fNode[10] 
        + fNode[11] + fNode[12] + fNode[13] + fNode[14] + fNode[15] + fNode[16] 
        + fNode[17] + fNode[18] + fNode[19] + fNode[20] + fNode[21] + fNode[22] 
        + fNode[23] + fNode[24] + fNode[25] + fNode[26];
    const dfloat invRho = 1 / rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15] 
        + fNode[19] + fNode[21] + fNode[23] + fNode[26]) 
        - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16] + fNode[20] 
        + fNode[22] + fNode[24] + fNode[25]) + 0.5 * f.x) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17]
        + fNode[19] + fNode[21] + fNode[24] + fNode[25]) 
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18] + fNode[20] 
        + fNode[22] + fNode[23] + fNode[26]) + 0.5 * f.y) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18]
        + fNode[19] + fNode[22] + fNode[23] + fNode[25]) 
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17] + fNode[20] 
        + fNode[21] + fNode[24] + fNode[26]) + 0.5 * f.z) * invRho;
#endif // !D3Q27

    macr.rho[idx] = rhoVar;
    macr.u.x[idx] = uxVar;
    macr.u.y[idx] = uyVar;
    macr.u.z[idx] = uzVar;
    velAuxIBM.x[idx] = uxVar;
    velAuxIBM.y[idx] = uyVar;
    velAuxIBM.z[idx] = uzVar;
}

__global__ 
void gpuResetNodesForces(ParticleNodeSoA particlesNodes)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= particlesNodes.numNodes)
        return;

    const dfloat3SoA force = particlesNodes.f;
    const dfloat3SoA delta_force = particlesNodes.deltaF;

    force.x[idx] = 0;
    force.y[idx] = 0;
    force.z[idx] = 0;
    delta_force.x[idx] = 0;
    delta_force.y[idx] = 0;
    delta_force.z[idx] = 0;
}

__global__ 
void gpuUpdateParticleCenterVelocityAndRotation(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

    if(p >= NUM_PARTICLES)
        return;

    ParticleCenter *pc = &(particleCenters[p]);

    if(!pc->movable)
        return;

    const dfloat inv_volume = 1 / pc->volume;

    // Update particle center velocity using its surface forces and the body forces
    pc->vel.x = pc->vel_old.x + (( (pc->f_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION ) 
                + pc->f.x * IBM_MOVEMENT_DISCRETIZATION) + pc->dP_internal.x) * inv_volume 
                + (pc->density - FLUID_DENSITY) * GX) / (pc->density);
    pc->vel.y = pc->vel_old.y + (( (pc->f_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION )  
                + pc->f.y * IBM_MOVEMENT_DISCRETIZATION) + pc->dP_internal.y) * inv_volume 
                + (pc->density - FLUID_DENSITY) * GY) / (pc->density);
    pc->vel.z = pc->vel_old.z + (( (pc->f_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION )  
                + pc->f.z * IBM_MOVEMENT_DISCRETIZATION) + pc->dP_internal.z) * inv_volume 
                + (pc->density - FLUID_DENSITY) * GZ) / (pc->density);

    // Auxiliary variables for angular velocity update
    dfloat error = 1;
    dfloat3 wNew = dfloat3(), wAux;
    const dfloat3 M = pc->M;
    const dfloat3 M_old = pc->M_old;
    const dfloat3 w_old = pc->w_old;
    const dfloat3 I = pc->I;

    wAux.x = w_old.x;
    wAux.y = w_old.y;
    wAux.z = w_old.z;

    // Iteration process to upadate angular velocity 
    // (Crank-Nicolson implicit scheme)
    for (int i = 0; error > 1e-6; i++)
    {
        //TODO the last term should be present in dL equation, but since it does not affect spheres, it will stay for now.
        wNew.x = pc->w_old.x + (((M.x * IBM_MOVEMENT_DISCRETIZATION + M_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->dL_internal.x) 
                - (I.z - I.y)*(w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION ) 
                * (w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION))/I.x;
        wNew.y = pc->w_old.y + (((M.y * IBM_MOVEMENT_DISCRETIZATION + M_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->dL_internal.y) 
                - (I.x - I.z)*(w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION ) 
                * (w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION))/I.y;
        wNew.z = pc->w_old.z + (((M.z * IBM_MOVEMENT_DISCRETIZATION + M_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->dL_internal.z) 
                - (I.y - I.x)*(w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION ) 
                * (w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION))/I.z;

        error =  (wNew.x - wAux.x)*(wNew.x - wAux.x)/(wNew.x*wNew.x);
        error += (wNew.y - wAux.y)*(wNew.y - wAux.y)/(wNew.y*wNew.y);
        error += (wNew.z - wAux.z)*(wNew.z - wAux.z)/(wNew.z*wNew.z);

        wAux.x = wNew.x;
        wAux.y = wNew.y;
        wAux.z = wNew.z;
    }
    // Store new velocities in particle center
    pc->w.x = wNew.x;
    pc->w.y = wNew.y;
    pc->w.z = wNew.z;
}

__global__
void gpuParticleMovement(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

    if(p >= NUM_PARTICLES)
        return;

    ParticleCenter *pc = &(particleCenters[p]);
    if(!pc->movable)
        return;

    #ifdef IBM_BC_X_WALL
        pc->pos.x +=  (pc->vel.x * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    #endif //IBM_BC_X_WALL
    #ifdef IBM_BC_X_PERIODIC
        dfloat dx =  (pc->vel.x * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->pos.x = std::fmod((dfloat)(pc->pos.x + dx + NX) , (dfloat)NX); 
    #endif //IBM_BC_X_PERIODIC

    #ifdef IBM_BC_Y_WALL
        pc->pos.y +=  (pc->vel.y * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    #endif //IBM_BC_Y_WALL
    #ifdef IBM_BC_Y_PERIODIC
        dfloat dy =  (pc->vel.y * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->pos.y = std::fmod((dfloat)(pc->pos.y + dy + NY) , (dfloat)NY);
    #endif //IBM_BC_Y_PERIODIC

    #ifdef IBM_BC_Z_WALL
        pc->pos.z +=  (pc->vel.z * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    #endif //IBM_BC_Z_WALL
    #ifdef IBM_BC_Z_PERIODIC
        dfloat dz =  (pc->vel.z * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->pos.z = std::fmod((dfloat)(pc->pos.z + dz + NZ) , (dfloat)NZ); 
    #endif //IBM_BC_Z_PERIODIC



    pc->w_avg.x = (pc->w.x   * IBM_MOVEMENT_DISCRETIZATION + pc->w_old.x   * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    pc->w_avg.y = (pc->w.y   * IBM_MOVEMENT_DISCRETIZATION + pc->w_old.y   * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    pc->w_avg.z = (pc->w.z   * IBM_MOVEMENT_DISCRETIZATION + pc->w_old.z   * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
}


__global__
void gpuUpdateParticleOldValues(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

    if(p >= NUM_PARTICLES)
        return;

    ParticleCenter *pc = &(particleCenters[p]);

    // Internal linear momentum delta = rho*volume*delta(v)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    pc->dP_internal.x = 0.0; //RHO_0 * pc->volume * (pc->vel.x - pc->vel_old.x);
    pc->dP_internal.y = 0.0; //RHO_0 * pc->volume * (pc->vel.y - pc->vel_old.y);
    pc->dP_internal.z = 0.0; //RHO_0 * pc->volume * (pc->vel.z - pc->vel_old.z);

    // Internal angular momentum delta = (rho_f/rho_p)*I*delta(omega)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    pc->dL_internal.x = 0.0; //(RHO_0 / pc->density) * pc->I.x * (pc->w.x - pc->w_old.x);
    pc->dL_internal.y = 0.0; //(RHO_0 / pc->density) * pc->I.y * (pc->w.y - pc->w_old.y);
    pc->dL_internal.z = 0.0; //(RHO_0 / pc->density) * pc->I.z * (pc->w.z - pc->w_old.z);

    pc->pos_old.x = pc->pos.x;
    pc->pos_old.y = pc->pos.y;
    pc->pos_old.z = pc->pos.z;

    pc->vel_old.x = pc->vel.x;
    pc->vel_old.y = pc->vel.y;
    pc->vel_old.z = pc->vel.z;

    pc->w_old.x = pc->w.x;
    pc->w_old.y = pc->w.y;
    pc->w_old.z = pc->w.z;

    pc->f_old.x = pc->f.x;
    pc->f_old.y = pc->f.y;
    pc->f_old.z = pc->f.z;

    // Reset force, because kernel is always added
    pc->f.x = 0;
    pc->f.y = 0;
    pc->f.z = 0;

    pc->M.x = 0;
    pc->M.y = 0;
    pc->M.z = 0;
}

__global__
void gpuParticleNodeMovement(
    ParticleNodeSoA const particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
){
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i >= particlesNodes.numNodes)
        return;

    const ParticleCenter pc = particleCenters[particlesNodes.particleCenterIdx[i]];

    if(!pc.movable)
        return;

    // TODO: make the calculation of w_norm along with w_avg?
    const dfloat w_norm = sqrt((pc.w_avg.x * pc.w_avg.x) 
        + (pc.w_avg.y * pc.w_avg.y) 
        + (pc.w_avg.z * pc.w_avg.z));

    if(w_norm <= 1e-8)
    {
        #ifdef IBM_BC_X_WALL
         particlesNodes.pos.x[i] += pc.pos.x - pc.pos_old.x;
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
        dfloat dx_vec = pc.pos.x - pc.pos_old.x;
            if(abs(dx_vec)>1.0)
                dx_vec = std::fmod((dfloat)dx_vec,(dfloat)NX);
            particlesNodes.pos.x[i] = std::fmod((dfloat)(particlesNodes.pos.x[i] + dx_vec + NX) , (dfloat)NX);
        #endif //IBM_BC_X_PERIODIC


        #ifdef IBM_BC_Y_WALL
            particlesNodes.pos.y[i] += pc.pos.y - pc.pos_old.y;
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            dfloat dy_vec = pc.pos.y - pc.pos_old.y;
            if(abs(dy_vec)>1.0)
                dy_vec = std::fmod((dfloat)dy_vec,(dfloat)NY);
            particlesNodes.pos.y[i] = std::fmod((dfloat)(particlesNodes.pos.y[i] + dy_vec + NY) , (dfloat)NY);
        #endif //IBM_BC_Y_PERIODIC
        

        #ifdef IBM_BC_Z_WALL
            particlesNodes.pos.z[i] += pc.pos.z - pc.pos_old.z;
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            dfloat dz_vec = pc.pos.z - pc.pos_old.z;
            if(abs(dz_vec)>1.0)
                dz_vec = std::fmod((dfloat)dz_vec,(dfloat)NZ);
            particlesNodes.pos.z[i] = std::fmod((dfloat)(particlesNodes.pos.z[i] + dz_vec + NZ) , (dfloat)NZ);
        #endif //IBM_BC_Z_PERIODIC

        return;
    }

    // particlesNodes.pos.x[i] += (pc.pos.x - pc.pos_old.x);
    // particlesNodes.pos.y[i] += (pc.pos.y - pc.pos_old.y);
    // particlesNodes.pos.z[i] += (pc.pos.z - pc.pos_old.z);

    // TODO: these variables are the same for every particle center, optimize it
    const dfloat q0 = cos(0.5*w_norm);
    const dfloat qi = (pc.w_avg.x/w_norm) * sin (0.5*w_norm);
    const dfloat qj = (pc.w_avg.y/w_norm) * sin (0.5*w_norm);
    const dfloat qk = (pc.w_avg.z/w_norm) * sin (0.5*w_norm);

    const dfloat tq0m1 = (q0*q0) - 0.5;


    #ifdef IBM_BC_X_WALL    
        const dfloat x_vec = particlesNodes.pos.x[i] - pc.pos_old.x;
    #endif //IBM_BC_X_WALL
    #ifdef IBM_BC_X_PERIODIC
        dfloat x_vec = particlesNodes.pos.x[i] - pc.pos_old.x;
        if(abs(x_vec)>1.0)
            x_vec = std::fmod((dfloat)(NX - x_vec), (dfloat)NX);
    #endif //IBM_BC_X_PERIODIC


    #ifdef IBM_BC_Y_WALL    
        const dfloat y_vec = particlesNodes.pos.y[i] - pc.pos_old.y;
    #endif //IBM_BC_Y_WALL
    #ifdef IBM_BC_Y_PERIODIC
        dfloat y_vec = particlesNodes.pos.y[i] - pc.pos_old.y;
        if(abs(y_vec)>1.0)
            y_vec = std::fmod((dfloat)(NY - y_vec) , (dfloat)NY);
    #endif //IBM_BC_Y_PERIODIC


    #ifdef IBM_BC_Z_WALL   
        const dfloat z_vec = particlesNodes.pos.z[i] - pc.pos_old.z;
    #endif //IBM_BC_Z_WALL
    #ifdef IBM_BC_Z_PERIODIC
        dfloat z_vec = particlesNodes.pos.z[i] - pc.pos_old.z;
        if(abs(z_vec)>1.0)
            z_vec = std::fmod((dfloat)(NZ - z_vec) , (dfloat)NZ);
    #endif //IBM_BC_Z_PERIODIC


    particlesNodes.pos.x[i] = pc.pos.x + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec);
    particlesNodes.pos.y[i] = pc.pos.y + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec);
    particlesNodes.pos.z[i] = pc.pos.z + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec);

    dfloat x = pc.pos.x + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec);
    dfloat y = pc.pos.y + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec);
    dfloat z = pc.pos.z + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec);

    #ifdef IBM_BC_X_PERIODIC
        x = std::fmod((dfloat)(x+NX),(dfloat)NX);
    #endif //IBM_BC_X_PERIODIC

    #ifdef IBM_BC_Y_PERIODIC
        y = std::fmod((dfloat)(y+NY),(dfloat)NY);
    #endif //IBM_BC_Y_PERIODIC

    #ifdef IBM_BC_Z_PERIODIC
        z = std::fmod((dfloat)(z+NZ),(dfloat)NZ);
    #endif //IBM_BC_Z_PERIODIC

    particlesNodes.pos.x[i] = x;
    particlesNodes.pos.y[i] = y;
    particlesNodes.pos.z[i] = z;
}



#endif // !IBM