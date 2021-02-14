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
    #else
    // Update macroscopics post boundary conditions and reset forces
    gpuUpdateMacrIBM<<<gridLBM, threadsLBM, 0, streamLBM[0]>>>(pop[0], macr[0], velsAuxIBM[0]);
    checkCudaErrors(cudaStreamSynchronize(streamLBM[0]));
    #endif

    // Reset forces in all IBM nodes
    gpuResetNodesForces<<<gridNodesIBM, threadsNodesIBM, 0, streamIBM[0]>>>(particles.nodesSoA);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    // Calculate collision force between particles
    #if defined SOFT_SPHERE
    gpuParticlesCollisionSoft<<<GRID_PCOLLISION_IBM, THREADS_PCOLLISION_IBM, 0, streamIBM[0]>>>(particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    #endif
    #if defined HARD_SPHERE
    gpuParticlesCollisionHard<<<GRID_PCOLLISION_IBM, THREADS_PCOLLISION_IBM, 0, streamIBM[0]>>>(particles.nodesSoA,particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    #endif

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
        #else
        copyFromArray<<<gridLBM, threadsLBM, 0, streamLBM[0]>>>(macr[0].u, velsAuxIBM[0]);
        #endif

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
    const int posBase[3] = {int(xIBM-P_DIST+1), int(yIBM-P_DIST+1), int(zIBM-P_DIST+1)};
    // Maximum stencil index for each direction xyz ("index" to stop)
    const int maxIdx[3] = {
        (posBase[0]+P_DIST*2-1) < (int)NX? P_DIST*2-1 : ((int)NX-1-posBase[0]), 
        (posBase[1]+P_DIST*2-1) < (int)NY? P_DIST*2-1 : ((int)NY-1-posBase[1]), 
        (posBase[2]+P_DIST*2-1) < (int)NZ? P_DIST*2-1 : ((int)NZ-1-posBase[2])};
    // Minimum stencil index for each direction xyz ("index" to start)
    const int minIdx[3] = {
        posBase[0] >= 0? 0 : -posBase[0], 
        posBase[1] >= 0? 0 : -posBase[1], 
        posBase[2] >= 0? 0 : -posBase[2]};

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

                idx = idxScalar(posBase[0]+xi, posBase[1]+yj, posBase[2]+zk);

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

                idx = idxScalar(posBase[0]+xi, posBase[1]+yj, posBase[2]+zk);

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
    #endif
)
{
    #if IBM_EULER_OPTIMIZATION
    unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
    if(j >= currEulerNodes)
        return;
    size_t idx = eulerIdxsUpdate[j];

    #else
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
       return;

    size_t idx = idxScalar(x, y, z);
    #endif
    
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
    pc->vel.x = pc->vel_old.x + ((0.5 * (pc->f_old.x + pc->f.x) + pc->dP_internal.x) * inv_volume 
        + (pc->density - FLUID_DENSITY) * GX) / (pc->density);
    pc->vel.y = pc->vel_old.y + ((0.5 * (pc->f_old.y + pc->f.y) + pc->dP_internal.y) * inv_volume 
        + (pc->density - FLUID_DENSITY) * GY) / (pc->density);
    pc->vel.z = pc->vel_old.z + ((0.5 * (pc->f_old.z + pc->f.z) + pc->dP_internal.z) * inv_volume 
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
        wNew.x = pc->w_old.x + ((0.5*(M.x + M_old.x) + pc->dL_internal.x) - (I.z - I.y)*0.25*(w_old.y + wAux.y)*(w_old.z + wAux.z))/I.x;
        wNew.y = pc->w_old.y + ((0.5*(M.y + M_old.y) + pc->dL_internal.y) - (I.x - I.z)*0.25*(w_old.x + wAux.x)*(w_old.z + wAux.z))/I.y;
        wNew.z = pc->w_old.z + ((0.5*(M.z + M_old.z) + pc->dL_internal.z) - (I.y - I.x)*0.25*(w_old.x + wAux.x)*(w_old.y + wAux.y))/I.z;

        error = (wNew.x - wAux.x)*(wNew.x - wAux.x)/(wNew.x*wNew.x);
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

    pc->pos.x += 0.5 * (pc->vel.x + pc->vel_old.x);
    pc->pos.y += 0.5 * (pc->vel.y + pc->vel_old.y);
    pc->pos.z += 0.5 * (pc->vel.z + pc->vel_old.z);

    pc->w_avg.x = 0.5 * (pc->w.x + pc->w_old.x);
    pc->w_avg.y = 0.5 * (pc->w.y + pc->w_old.y);
    pc->w_avg.z = 0.5 * (pc->w.z + pc->w_old.z);
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

    // Internal angular momentum delta = I*delta(omega)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    pc->dL_internal.x = 0.0; //pc->I.x * (pc->w.x - pc->w_old.x);
    pc->dL_internal.y = 0.0; //pc->I.y * (pc->w.y - pc->w_old.y);
    pc->dL_internal.z = 0.0; //pc->I.z * (pc->w.z - pc->w_old.z);

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
        particlesNodes.pos.x[i] += pc.pos.x - pc.pos_old.x;
        particlesNodes.pos.y[i] += pc.pos.y - pc.pos_old.y;
        particlesNodes.pos.z[i] += pc.pos.z - pc.pos_old.z;
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

    const dfloat x_vec = particlesNodes.pos.x[i] - pc.pos_old.x;
    const dfloat y_vec = particlesNodes.pos.y[i] - pc.pos_old.y;
    const dfloat z_vec = particlesNodes.pos.z[i] - pc.pos_old.z;

    particlesNodes.pos.x[i] = pc.pos.x + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec);
    particlesNodes.pos.y[i] = pc.pos.y + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec);
    particlesNodes.pos.z[i] = pc.pos.z + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec);
}

#if defined SOFT_SPHERE
__global__
void gpuParticlesCollisionSoft(
    ParticleCenter particleCenters[NUM_PARTICLES]
){
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx > TOTAL_PCOLLISION_IBM_THREADS)
        return;
    
    /* Maps a 1D array to a Floyd triangle, where the last row is for checking
    collision against the wall and the other ones to check collision between 
    particles, with index given by row/column. Example for 7 particles:

    FLOYD TRIANGLE
        c0  c1  c2  c3  c4  c5  c6
    r0  0
    r1  1   2
    r2  3   4   5
    r3  6   7   8   9
    r4  10  11  12  13  14
    r5  15  16  17  18  19  20
    r6  21  22  23  24  25  26  27

    Index 7 is in r3, c1. It will compare p[1] (particle in index 1), from column,
    with p[4], from row (this is because for all rows one is added to its index)

    Index 0 will compare p[0] (column) and p[1] (row)
    Index 13 will compare p[3] (column) and p[5] (row)
    Index 20 will compare p[5] (column) and p[6] (row)

    For the last column, the particles check collision against the wall.
    Index 21 will check p[0] (column) collision against the wall
    Index 27 will check p[6] (column) collision against the wall
    Index 24 will check p[3] (column) collision against the wall

    FROM INDEX TO ROW/COLUMN
    Starting column/row from 1, the n'th row always ends (n)*(n+1)/2+1. So:

    k = (n)*(n+1)/2+1
    n^2 + n - (2k+1) = 0

    (with k=particle index)
    n_row = ceil((-1 + Sqrt(1 + 8(k+1))) / 2)
    n_column = k - n_row * (n_row - 1) / 2
    */

    const unsigned int row = ceil((-1.0+sqrt((float)1+8*(idx+1)))/2);
    const unsigned int column = idx - ((row-1)*row)/2;

    // Magnitude of gravity force
    const dfloat grav = sqrt(GX * GX + GY * GY + GZ * GZ);

    // Particle from column
    ParticleCenter* pc_i = &particleCenters[column];

    // Collision against walls
    if(row == NUM_PARTICLES){

        if(!pc_i->movable)
            return;

        dfloat3 f_normal = dfloat3();
        dfloat3 f_tang = dfloat3();
        dfloat3 t, G, G_ct,tang_disp;
        dfloat f_n,f_kn,displacement,mag;

        // Particle position
        const dfloat  m_i = pc_i ->volume * pc_i ->density;
        const dfloat3 pos_i = pc_i->pos;
        const dfloat r_i = pc_i->radius;

        dfloat3 v_i = pc_i->vel;
        dfloat3 w_i = pc_i->w;
        const dfloat min_dist = 2 * r_i;

        dfloat effective_radius = r_i;
        dfloat effective_mass = m_i;

        dfloat STIFFNESS_NORMAL;
        dfloat STIFFNESS_TANGENTIAL;
        dfloat damping_const;
        dfloat DAMPING_NORMAL;                                
        dfloat DAMPING_TANGENTIAL;
        dfloat3 n,f_dirs, m_dirs;

        dfloat pos_mirror, dist_abs;

        // East
        pos_mirror = -pos_i.x;
        dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            displacement = (2.0 * r_i - dist_abs)/2.0;
            //inverse normal surface vector
            
            n.x = -1.0;
            n.y = 0.0;
            n.z = 0.0;
            
            // relative velocity vector
            G.x = v_i.x;
            G.y = v_i.y;
            G.z = v_i.z;

            STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
            STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
            damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
            DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
            DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

            //normal force
            f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
            f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
            f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
            f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
            f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

            //tangential force       
            G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
            G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
            G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
        
            mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct.x/mag;
                t.y = G_ct.y/mag;
                t.z = G_ct.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }

            //TODO : this is not correct. it should take distance from impact point, not from previous time-step
            tang_disp.x = G_ct.x;
            tang_disp.y = G_ct.y;
            tang_disp.z = G_ct.z;

            f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
            f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
            f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

            if(  mag > FRICTION_COEF * abs(f_n) ){
                f_tang.x = - FRICTION_COEF * f_n * t.x;
                f_tang.y = - FRICTION_COEF * f_n * t.y;
                f_tang.z = - FRICTION_COEF * f_n * t.z;
            }

            // Force in each direction
            f_dirs = dfloat3(
                f_normal.x + f_tang.x,
                f_normal.y + f_tang.y,
                f_normal.z + f_tang.z
            );
            //Torque in each direction
            m_dirs = dfloat3(
                r_i * (n.y*f_tang.z - n.z*f_tang.y),
                r_i * (n.z*f_tang.x - n.x*f_tang.z),
                r_i * (n.x*f_tang.y - n.y*f_tang.x)
            );

            atomicAdd(&(pc_i->f.x), f_dirs.x);
            atomicAdd(&(pc_i->f.y), f_dirs.y);
            atomicAdd(&(pc_i->f.z), f_dirs.z);

            atomicAdd(&(pc_i->M.x), m_dirs.x);
            atomicAdd(&(pc_i->M.y), m_dirs.y);
            atomicAdd(&(pc_i->M.z), m_dirs.z);
        }

        // Weast
        pos_mirror = 2 * (NX - 1) - pos_i.x;
        dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            displacement = (2.0 * r_i - dist_abs)/2.0;
            //inverse normal surface vector
            
            n.x = 1.0;
            n.y = 0.0;
            n.z = 0.0;
            
            // relative velocity vector
            G.x = v_i.x;
            G.y = v_i.y;
            G.z = v_i.z;

            STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
            STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
            damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
            DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
            DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

            //normal force
            f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
            f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
            f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
            f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
            f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

            //tangential force       
            G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
            G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
            G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
        
            mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct.x/mag;
                t.y = G_ct.y/mag;
                t.z = G_ct.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }

            //TODO : this is not correct. it should take distance from impact point, not from previous time-step
            tang_disp.x = G_ct.x;
            tang_disp.y = G_ct.y;
            tang_disp.z = G_ct.z;

            f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
            f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
            f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

            if(  mag > FRICTION_COEF * abs(f_n) ){
                f_tang.x = - FRICTION_COEF * f_n * t.x;
                f_tang.y = - FRICTION_COEF * f_n * t.y;
                f_tang.z = - FRICTION_COEF * f_n * t.z;
            }

            // Force in each direction
            f_dirs = dfloat3(
                f_normal.x + f_tang.x,
                f_normal.y + f_tang.y,
                f_normal.z + f_tang.z
            );
            //Torque in each direction
            m_dirs = dfloat3(
                r_i * (n.y*f_tang.z - n.z*f_tang.y),
                r_i * (n.z*f_tang.x - n.x*f_tang.z),
                r_i * (n.x*f_tang.y - n.y*f_tang.x)
            );

            atomicAdd(&(pc_i->f.x), f_dirs.x);
            atomicAdd(&(pc_i->f.y), f_dirs.y);
            atomicAdd(&(pc_i->f.z), f_dirs.z);

            atomicAdd(&(pc_i->M.x), m_dirs.x);
            atomicAdd(&(pc_i->M.y), m_dirs.y);
            atomicAdd(&(pc_i->M.z), m_dirs.z);
        }

        // South
        pos_mirror = - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            displacement = (2.0 * r_i - dist_abs)/2.0;
            //inverse normal surface vector
            
            n.x = 0.0;
            n.y = -1.0;
            n.z = 0.0;
            
            // relative velocity vector
            G.x = v_i.x;
            G.y = v_i.y;
            G.z = v_i.z;

            STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
            STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
            damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
            DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
            DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

            //normal force
            f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
            f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
            f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
            f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
            f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

            //tangential force       
            G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
            G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
            G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
        
            mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct.x/mag;
                t.y = G_ct.y/mag;
                t.z = G_ct.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }

            //TODO : this is not correct. it should take distance from impact point, not from previous time-step
            tang_disp.x = G_ct.x;
            tang_disp.y = G_ct.y;
            tang_disp.z = G_ct.z;

            f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
            f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
            f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

            if(  mag > FRICTION_COEF * abs(f_n) ){
                f_tang.x = - FRICTION_COEF * f_n * t.x;
                f_tang.y = - FRICTION_COEF * f_n * t.y;
                f_tang.z = - FRICTION_COEF * f_n * t.z;
            }

            // Force in each direction
            f_dirs = dfloat3(
                f_normal.x + f_tang.x,
                f_normal.y + f_tang.y,
                f_normal.z + f_tang.z
            );
            //Torque in each direction
            m_dirs = dfloat3(
                r_i * (n.y*f_tang.z - n.z*f_tang.y),
                r_i * (n.z*f_tang.x - n.x*f_tang.z),
                r_i * (n.x*f_tang.y - n.y*f_tang.x)
            );

            atomicAdd(&(pc_i->f.x), f_dirs.x);
            atomicAdd(&(pc_i->f.y), f_dirs.y);
            atomicAdd(&(pc_i->f.z), f_dirs.z);

            atomicAdd(&(pc_i->M.x), m_dirs.x);
            atomicAdd(&(pc_i->M.y), m_dirs.y);
            atomicAdd(&(pc_i->M.z), m_dirs.z);
        }

        // North
        pos_mirror = 2 * (NY - 1) - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            displacement = (2.0 * r_i - dist_abs)/2.0;
            //inverse normal surface vector
            
            n.x = 0.0;
            n.y = 1.0;
            n.z = 0.0;
            
            // relative velocity vector
            G.x = v_i.x;
            G.y = v_i.y;
            G.z = v_i.z;

            STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
            STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
            damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
            DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
            DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

            //normal force
            f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
            f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
            f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
            f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
            f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

            //tangential force       
            G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
            G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
            G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
        
            mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct.x/mag;
                t.y = G_ct.y/mag;
                t.z = G_ct.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }

            //TODO : this is not correct. it should take distance from impact point, not from previous time-step
            tang_disp.x = G_ct.x;
            tang_disp.y = G_ct.y;
            tang_disp.z = G_ct.z;

            f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
            f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
            f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

            if(  mag > FRICTION_COEF * abs(f_n) ){
                f_tang.x = - FRICTION_COEF * f_n * t.x;
                f_tang.y = - FRICTION_COEF * f_n * t.y;
                f_tang.z = - FRICTION_COEF * f_n * t.z;
            }

            // Force in each direction
            f_dirs = dfloat3(
                f_normal.x + f_tang.x,
                f_normal.y + f_tang.y,
                f_normal.z + f_tang.z
            );
            //Torque in each direction
            m_dirs = dfloat3(
                r_i * (n.y*f_tang.z - n.z*f_tang.y),
                r_i * (n.z*f_tang.x - n.x*f_tang.z),
                r_i * (n.x*f_tang.y - n.y*f_tang.x)
            );

            atomicAdd(&(pc_i->f.x), f_dirs.x);
            atomicAdd(&(pc_i->f.y), f_dirs.y);
            atomicAdd(&(pc_i->f.z), f_dirs.z);

            atomicAdd(&(pc_i->M.x), m_dirs.x);
            atomicAdd(&(pc_i->M.y), m_dirs.y);
            atomicAdd(&(pc_i->M.z), m_dirs.z);
        }

        // Back
        pos_mirror = -pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist){
            displacement = (2.0 * r_i - dist_abs)/2.0;
            //inverse normal surface vector
            
            n.x = 0.0;
            n.y = 0.0;
            n.z = -1.0;
            
            // relative velocity vector
            G.x = v_i.x;
            G.y = v_i.y;
            G.z = v_i.z;

            STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
            STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
            damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
            DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
            DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

            //normal force
            f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
            f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
            f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
            f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
            f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

            //tangential force       
            G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
            G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
            G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
        
            mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct.x/mag;
                t.y = G_ct.y/mag;
                t.z = G_ct.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }

            //TODO : this is not correct. it should take distance from impact point, not from previous time-step
            tang_disp.x = G_ct.x;
            tang_disp.y = G_ct.y;
            tang_disp.z = G_ct.z;

            f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
            f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
            f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

            if(  mag > FRICTION_COEF * abs(f_n) ){
                f_tang.x = - FRICTION_COEF * f_n * t.x;
                f_tang.y = - FRICTION_COEF * f_n * t.y;
                f_tang.z = - FRICTION_COEF * f_n * t.z;
            }

            // Force in each direction
            f_dirs = dfloat3(
                f_normal.x + f_tang.x,
                f_normal.y + f_tang.y,
                f_normal.z + f_tang.z
            );
            //Torque in each direction
            m_dirs = dfloat3(
                r_i * (n.y*f_tang.z - n.z*f_tang.y),
                r_i * (n.z*f_tang.x - n.x*f_tang.z),
                r_i * (n.x*f_tang.y - n.y*f_tang.x)
            );

            atomicAdd(&(pc_i->f.x), f_dirs.x);
            atomicAdd(&(pc_i->f.y), f_dirs.y);
            atomicAdd(&(pc_i->f.z), f_dirs.z);

            atomicAdd(&(pc_i->M.x), m_dirs.x);
            atomicAdd(&(pc_i->M.y), m_dirs.y);
            atomicAdd(&(pc_i->M.z), m_dirs.z);

        }

        // Front
        pos_mirror = 2 * (NZ - 1) - pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist) {
            displacement = (2.0 * r_i - dist_abs)/2.0;
            //inverse normal surface vector
            
            n.x = 0.0;
            n.y = 0.0;
            n.z = 1.0;
            
            // relative velocity vector
            G.x = v_i.x;
            G.y = v_i.y;
            G.z = v_i.z;

            STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
            STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
            damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
            DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
            DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

            //normal force
            f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
            f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
            f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
            f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
            f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

            //tangential force       
            G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
            G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
            G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
        
            mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct.x/mag;
                t.y = G_ct.y/mag;
                t.z = G_ct.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }

            //TODO : this is not correct. it should take distance from impact point, not from previous time-step
            tang_disp.x = G_ct.x;
            tang_disp.y = G_ct.y;
            tang_disp.z = G_ct.z;

            f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
            f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
            f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

            if(  mag > FRICTION_COEF * abs(f_n) ){
                f_tang.x = - FRICTION_COEF * f_n * t.x;
                f_tang.y = - FRICTION_COEF * f_n * t.y;
                f_tang.z = - FRICTION_COEF * f_n * t.z;
            }

            // Force in each direction
            f_dirs = dfloat3(
                f_normal.x + f_tang.x,
                f_normal.y + f_tang.y,
                f_normal.z + f_tang.z
            );
            //Torque in each direction
            m_dirs = dfloat3(
                r_i * (n.y*f_tang.z - n.z*f_tang.y),
                r_i * (n.z*f_tang.x - n.x*f_tang.z),
                r_i * (n.x*f_tang.y - n.y*f_tang.x)
            );

            atomicAdd(&(pc_i->f.x), f_dirs.x);
            atomicAdd(&(pc_i->f.y), f_dirs.y);
            atomicAdd(&(pc_i->f.z), f_dirs.z);

            atomicAdd(&(pc_i->M.x), m_dirs.x);
            atomicAdd(&(pc_i->M.y), m_dirs.y);
            atomicAdd(&(pc_i->M.z), m_dirs.z);
            
        }
    }
    // Collision against particles
    else{
        ParticleCenter* pc_j = &particleCenters[row];

        if(!pc_i->movable && !pc_j->movable)
            return;

        // Particle i info (column)
        const dfloat3 pos_i = pc_i->pos;
        const dfloat r_i = pc_i->radius;

        // Particle j info (row)
        const dfloat3 pos_j = pc_j->pos;
        const dfloat r_j = pc_j->radius;

        // Particles position difference
        const dfloat3 diff_pos = dfloat3(
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z);

        // Magnitude of distance between particles
        const dfloat mag_dist = sqrt(
            diff_pos.x*diff_pos.x
            + diff_pos.y*diff_pos.y
            + diff_pos.z*diff_pos.z);

        //normal collision vector
        const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);
        
        // Force on particle
        dfloat3 f_dirs = dfloat3();
        dfloat3 m_dirs_i = dfloat3();
        dfloat3 m_dirs_j = dfloat3();


        // Hard collision (one particle inside another)
        if(mag_dist < r_i+r_j){
            dfloat3 f_normal = dfloat3();
            dfloat3 f_tang = dfloat3();
            dfloat3 t, G, G_ct,tang_disp;
            dfloat f_n,displacement,mag;

            const dfloat  m_i = pc_i ->volume * pc_i ->density;
            dfloat3 v_i = pc_i->vel;
            dfloat3 w_i = pc_i->w;
            
            const dfloat  m_j = pc_j ->volume * pc_j ->density;
            dfloat3 v_j = pc_j->vel;
            dfloat3 w_j = pc_j->w;

            // there is chance it is divided by the particle diameter
            displacement = r_i + r_j - mag_dist;

            // relative velocity vector
            G.x = v_i.x-v_j.x;
            G.y = v_i.y-v_j.y;
            G.z = v_i.z-v_j.z;

            //colision parameters

            dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
            dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

            const dfloat STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
            const dfloat STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
            dfloat damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
            const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
            const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);
            
            
            //normal force
            dfloat f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
            f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
            f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
            f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
            f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

            //tangential force       
            G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) + r_j*(w_j.y*n.z - w_j.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
            G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) + r_j*(w_j.z*n.x - w_j.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
            G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) + r_j*(w_j.x*n.y - w_j.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
        
            mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct.x/mag;
                t.y = G_ct.y/mag;
                t.z = G_ct.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }
            //tangential displacement = int_{t = t_0}^{t =t} G_ct dt, where t_0 is when occurred the contanct,
            //here it will be made a inteporlation based on the old position of the particles
            //const dfloat3 diff_pos_old = dfloat3(
            //    pos_old_i.x - pos_old_j.x,
            //    pos_old_i.y - pos_old_j.y,
            //    pos_old_i.z - pos_old_j.z); 
            //const dfloat mag_dist_old = sqrt(
            //      diff_pos_old.x*diff_pos_old.x
            //    + diff_pos_old.y*diff_pos_old.y
            //    + diff_pos_old.z*diff_pos_old.z);
    
            //dfloat partial_collition_time = (mag_dist_old - (r_i+r_j))/(mag_dist_old - mag_dist);
            dfloat partial_collition_time = 1.0;
            tang_disp.x = partial_collition_time * G_ct.x;
            tang_disp.y = partial_collition_time * G_ct.y;
            tang_disp.z = partial_collition_time * G_ct.z;

            f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
            f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
            f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

            mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

            if(  mag > FRICTION_COEF * abs(f_n) ){
                f_tang.x = - FRICTION_COEF * f_n * t.x;
                f_tang.y = - FRICTION_COEF * f_n * t.y;
                f_tang.z = - FRICTION_COEF * f_n * t.z;
            }


            // Force in each direction
            f_dirs = dfloat3(
                f_normal.x + f_tang.x,
                f_normal.y + f_tang.y,
                f_normal.z + f_tang.z
            );
            //Torque in each direction
            m_dirs_i = dfloat3(
                r_i * (n.y*f_tang.z - n.z*f_tang.y),
                r_i * (n.z*f_tang.x - n.x*f_tang.z),
                r_i * (n.x*f_tang.y - n.y*f_tang.x)
            );
            m_dirs_j = dfloat3(
                r_j * (n.y*f_tang.z - n.z*f_tang.y),
                r_j * (n.z*f_tang.x - n.x*f_tang.z),
                r_j * (n.x*f_tang.y - n.y*f_tang.x)
            );

            // Force positive in particle i (column)
            atomicAdd(&(pc_i->f.x), -f_dirs.x);
            atomicAdd(&(pc_i->f.y), -f_dirs.y);
            atomicAdd(&(pc_i->f.z), -f_dirs.z);

            atomicAdd(&(pc_i->M.x), m_dirs_i.x);
            atomicAdd(&(pc_i->M.y), m_dirs_i.y);
            atomicAdd(&(pc_i->M.z), m_dirs_i.z);

            // Force negative in particle j (row)
            atomicAdd(&(pc_j->f.x), f_dirs.x);
            atomicAdd(&(pc_j->f.y), f_dirs.y);
            atomicAdd(&(pc_j->f.z), f_dirs.z);

            atomicAdd(&(pc_j->M.x), m_dirs_j.x); //normal vector takes care of negative sign
            atomicAdd(&(pc_j->M.y), m_dirs_j.y);
            atomicAdd(&(pc_j->M.z), m_dirs_j.z);           
        }
    }
}
#endif //SOFT_SPHERE

#if defined HARD_SPHERE
__global__
void gpuParticlesCollisionHard(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
){
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx > TOTAL_PCOLLISION_IBM_THREADS)
        return;

    const unsigned int row = ceil((-1.0+sqrt((float)1+8*(idx+1)))/2);
    const unsigned int column = idx - ((row-1)*row)/2;


    // Particle from column
    ParticleCenter* pc_i = &particleCenters[column];



    // Collision against walls
    if(row == NUM_PARTICLES){

        if(!pc_i->movable)
            return;

        // Particle i info (column)
        const dfloat  m_i = pc_i ->volume * pc_i ->density;
        const dfloat3  I_i = pc_i ->I;
        const dfloat r_i = pc_i->radius;
        const dfloat3 pos_i = pc_i->pos;
        dfloat3 v_i = pc_i->vel;
        dfloat3 w_i = pc_i->w;
        dfloat dvx_i = 0;
        dfloat dvy_i = 0;
        dfloat dvz_i = 0;
        dfloat dwx_i = 0;
        dfloat dwy_i = 0;
        dfloat dwz_i = 0;

        //velocity mag
        const dfloat vel_mag = sqrt(v_i.x*v_i.x + v_i.y*v_i.y + v_i.z*v_i.z);
        const dfloat min_dist = 2 * r_i;
        dfloat ep_x, ep_y, ep_z, ep_mag;
        dfloat px, py,pz; //penetration
        px = 0;
        py = 0;
        pz = 0;

        //East x=0       
        dfloat pos_mirror = -pos_i.x;
        dfloat dist_abs = abs(pos_i.x - pos_mirror);     
        if (dist_abs <= min_dist){
            px = -(min_dist - dist_abs)/2;
            if ( (v_i.x / vel_mag) < -2 / (7*FRICTION_COEF*(REST_COEF+1)) && FRICTION_COEF != 0){
                dvy_i -= v_i.y - (5.0/7.0)*(v_i.y - 2*r_i*w_i.z/5);
                dvz_i -= v_i.z - (5.0/7.0)*(v_i.z - 2*r_i*w_i.y/5);

                dvx_i -= v_i.x + REST_COEF * v_i.x;

                dwy_i -= w_i.y - v_i.z/r_i;
                dwx_i += 0;
                dwz_i -= w_i.z + v_i.y/r_i;

            } else {
                if(v_i.y == 0 || v_i.z == 0){
                    ep_y = 0;
                    ep_z = 0;                    
                }else if(v_i.y == 0){
                    ep_z = 1;
                    ep_y = 0;
                }else if(v_i.z == 0){
                    ep_y = 1;
                    ep_z = 0;
                }else{
                    ep_mag = sqrt(v_i.y*v_i.y + v_i.z*v_i.z);
                    ep_y = v_i.y/ep_mag;
                    ep_z = v_i.z/ep_mag;
                }
                dvy_i += ep_y*FRICTION_COEF*(REST_COEF+1)*v_i.x;
                dvz_i += ep_z*FRICTION_COEF*(REST_COEF+1)*v_i.x;

                dvx_i -= v_i.x + REST_COEF * v_i.x;

                dwy_i += - (5.0/(2.0*r_i))*ep_z*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.x);
                dwz_i += + (5.0/(2.0*r_i))*ep_y*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.x);
                dwx_i += 0;
            }

        }
        //West x = NX-1
        pos_mirror = 2 * (NX - 1) - pos_i.x;
        dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            px = (min_dist - dist_abs)/2;
            if ( (v_i.x / vel_mag) < 2 / (7*FRICTION_COEF*(REST_COEF+1)) && FRICTION_COEF != 0){
                dvy_i -= v_i.y - (5.0/7.0)*(v_i.y - 2*r_i*w_i.z/5);
                dvz_i -= v_i.z - (5.0/7.0)*(v_i.z - 2*r_i*w_i.y/5);

                dvx_i -= v_i.x + REST_COEF * v_i.x;

                dwy_i -= w_i.y - v_i.z/r_i;
                dwx_i -= 0;
                dwz_i -= w_i.z + v_i.y/r_i;

            } else {
                if(v_i.y == 0 || v_i.z == 0){
                    ep_y = 0;
                    ep_z = 0;                    
                }else if(v_i.y == 0){
                    ep_z = 1;
                    ep_y = 0;
                }else if(v_i.z == 0){
                    ep_y = 1;
                    ep_z = 0;
                }else{
                    ep_mag = sqrt(v_i.y*v_i.y + v_i.z*v_i.z);
                    ep_y = v_i.y/ep_mag;
                    ep_z = v_i.z/ep_mag;
                }

                dvy_i += ep_y*FRICTION_COEF*(REST_COEF+1)*v_i.x;
                dvz_i += ep_z*FRICTION_COEF*(REST_COEF+1)*v_i.x;

                dvx_i -= v_i.x + REST_COEF * v_i.x;

                dwy_i += - (5.0/(2.0*r_i))*ep_z*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.x);
                dwz_i += + (5.0/(2.0*r_i))*ep_y*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.x);
                dwx_i += 0;
            }
        }

        //South y = 0
        pos_mirror = - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            py = -(min_dist - dist_abs)/2;
            if ( (v_i.y / vel_mag) < -2 / (7*FRICTION_COEF*(REST_COEF+1))  && FRICTION_COEF != 0){
                dvx_i -= v_i.x - (5.0/7.0)*(v_i.x - 2*r_i*w_i.z/5);
                dvz_i -= v_i.z - (5.0/7.0)*(v_i.z - 2*r_i*w_i.x/5);

                dvy_i -= v_i.y  + REST_COEF * v_i.y;


                dwx_i -= w_i.x - v_i.z/r_i;
                dwy_i += 0;
                dwz_i -= w_i.z + v_i.x/r_i;

            } else {
                if(v_i.x == 0 || v_i.z == 0){
                    ep_x = 0;
                    ep_z = 0;                    
                }else if(v_i.x == 0){
                    ep_z = 1;
                    ep_x = 0;
                }else if(v_i.z == 0){
                    ep_x = 1;
                    ep_z = 0;
                }else{
                    ep_mag = sqrt(v_i.x*v_i.x + v_i.z*v_i.z);
                    ep_x = v_i.x/ep_mag;
                    ep_z = v_i.z/ep_mag;
                }

                dvx_i += ep_x*FRICTION_COEF*(REST_COEF+1)*v_i.y;
                dvz_i += ep_z*FRICTION_COEF*(REST_COEF+1)*v_i.y;

                dvy_i -= v_i.y + REST_COEF * v_i.y;

                dwx_i += - (5.0/(2.0*r_i))*ep_z*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.y);
                dwz_i += + (5.0/(2.0*r_i))*ep_x*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.y);
                dwy_i += 0;
            }
        }

        //North y = NY -1
        pos_mirror = 2 * (NY - 1) - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            py = (min_dist - dist_abs)/2;
            if ( (v_i.y / vel_mag) < 2 / (7*FRICTION_COEF*(REST_COEF+1)) && FRICTION_COEF != 0){
                dvx_i -= v_i.x - (5.0/7.0)*(v_i.x - 2*r_i*w_i.z/5);
                dvz_i -= v_i.z - (5.0/7.0)*(v_i.z - 2*r_i*w_i.x/5);

                dvy_i -= v_i.y + REST_COEF * v_i.y;

                dwx_i -= w_i.x - v_i.z/r_i;
                dwy_i += 0;
                dwz_i -= w_i.z + v_i.x/r_i;
            } else {
                if(v_i.x == 0 || v_i.z == 0){
                    ep_x = 0;
                    ep_z = 0;                    
                }else if(v_i.x == 0){
                    ep_z = 1;
                    ep_x = 0;
                }else if(v_i.z == 0){
                    ep_x = 1;
                    ep_z = 0;
                }else{
                    ep_mag = sqrt(v_i.x*v_i.x + v_i.z*v_i.z);
                    ep_x = v_i.x/ep_mag;
                    ep_z = v_i.z/ep_mag;
                }
                dvx_i += + ep_x*FRICTION_COEF*(REST_COEF+1)*v_i.y;
                dvz_i += + ep_z*FRICTION_COEF*(REST_COEF+1)*v_i.y;

                dvy_i -= v_i.y + REST_COEF * v_i.y;

                dwx_i += - (5.0/(2.0*r_i))*ep_z*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.y);
                dwz_i += + (5.0/(2.0*r_i))*ep_x*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.y);
                dwy_i += 0;
            }
        }
        //Back z = 0
        pos_mirror = -pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist){
            pz = -(min_dist - dist_abs)/2;
            if ( (v_i.z / vel_mag) < -2 / (7*FRICTION_COEF*(REST_COEF+1)) && FRICTION_COEF != 0){
                dvx_i -= v_i.x - (5.0/7.0)*(v_i.x - 2*r_i*w_i.y/5);
                dvy_i -= v_i.y - (5.0/7.0)*(v_i.y - 2*r_i*w_i.x/5);

                dvz_i -= v_i.z + REST_COEF * v_i.z;

                dwx_i -= w_i.x - v_i.y/r_i;
                dwz_i += 0;
                dwy_i -= w_i.y + v_i.x/r_i;
            } else {
                if(v_i.x == 0 || v_i.y == 0){
                    ep_y = 0; 
                    ep_x = 0;
                }else if(v_i.x == 0){
                    ep_y = 1;
                    ep_x = 0;
                }else if(v_i.y == 0){
                    ep_x = 1;
                    ep_y = 0;
                }else{
                    ep_mag = sqrt(v_i.x*v_i.x + v_i.y*v_i.y);
                    ep_x = v_i.x/ep_mag;
                    ep_y = v_i.y/ep_mag;
                }    
                dvx_i += ep_x*FRICTION_COEF*(REST_COEF+1)*v_i.z;
                dvy_i += ep_y*FRICTION_COEF*(REST_COEF+1)*v_i.z;

                dvz_i -= v_i.z + REST_COEF * v_i.z;

                
                dwx_i += - (5.0/(2.0*r_i))*ep_y*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.z);
                dwy_i += + (5.0/(2.0*r_i))*ep_x*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.z);
                dwz_i += 0;
            }

        }
        
        //Front z = NZ -1
        pos_mirror = 2 * (NZ - 1) - pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist) {
            pz = (min_dist - dist_abs)/2;
            if ( (v_i.z / vel_mag) < 2 / (7*FRICTION_COEF*(REST_COEF+1)) && FRICTION_COEF != 0){
                dvx_i -= v_i.x - (5.0/7.0)*(v_i.x - 2*r_i*w_i.y/5);
                dvy_i -= v_i.y - (5.0/7.0)*(v_i.y - 2*r_i*w_i.x/5);

                dvz_i -= v_i.z + REST_COEF * v_i.z;

                dwx_i -= w_i.x - v_i.y/r_i;
                dwz_i += 0;
                dwy_i -= w_i.y + v_i.x/r_i;
            } else {
                if(v_i.x == 0 || v_i.y == 0){
                    ep_y = 0; 
                    ep_x = 0;
                }else if(v_i.x == 0){
                    ep_y = 1;
                    ep_x = 0;
                }else if(v_i.y == 0){
                    ep_x = 1;
                    ep_y = 0;
                }else{
                    ep_mag = sqrt(v_i.x*v_i.x + v_i.y*v_i.y);
                    ep_x = v_i.x/ep_mag;
                    ep_y = v_i.y/ep_mag;
                }

                dvx_i += ep_x*FRICTION_COEF*(REST_COEF+1)*v_i.z;
                dvy_i += ep_y*FRICTION_COEF*(REST_COEF+1)*v_i.z;

                dvz_i -= v_i.z + REST_COEF * v_i.z;

                dwx_i += - (5.0/(2.0*r_i))*ep_y*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.z);
                dwy_i += + (5.0/(2.0*r_i))*ep_x*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.z);
                dwz_i += 0;
            }
        }
        
        // Force positive in particle i (column)
        atomicAdd(&(pc_i->f.x), dvx_i * m_i);
        atomicAdd(&(pc_i->f.y), dvy_i * m_i);
        atomicAdd(&(pc_i->f.z), dvz_i * m_i);

        atomicAdd(&(pc_i->M.x), dwx_i * I_i.x);
        atomicAdd(&(pc_i->M.y), dwy_i * I_i.y);
        atomicAdd(&(pc_i->M.z), dwz_i * I_i.z);
        //update node velocities

        const dfloat add_dist = 1e-6;
        pc_i->pos.x -= px*(1.0 + add_dist);
        pc_i->pos.y -= py*(1.0 + add_dist);
        pc_i->pos.z -= pz*(1.0 + add_dist);

        dfloat xIBM,yIBM,zIBM;
        for(int i = 0; i < particlesNodes.numNodes; i++){
            if ( particlesNodes.particleCenterIdx[i] == column){
                xIBM = particlesNodes.pos.x[i];
                yIBM = particlesNodes.pos.y[i];
                zIBM = particlesNodes.pos.z[i];

                particlesNodes.vel.x[i] = v_i.x + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
                particlesNodes.vel.y[i] = v_i.y + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
                particlesNodes.vel.z[i] = v_i.z + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));

                particlesNodes.vel_old.x[i] = particlesNodes.vel_old.x[i] + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
                particlesNodes.vel_old.y[i] = particlesNodes.vel_old.y[i] + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
                particlesNodes.vel_old.z[i] = particlesNodes.vel_old.z[i] + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));
            }
        }
        
        



    } // Collision against particles
    else{
        dfloat3 n, t, G_0, G_c_0, G_ct_0;
        dfloat mag;
        dfloat px, py,pz; //penetration
        px = 0;
        py = 0;
        pz = 0;

        ParticleCenter* pc_j = &particleCenters[row];

        if(!pc_i->movable && !pc_j->movable)
            return;
        
        // Particle i info (column)
        const dfloat  m_i = pc_i ->volume * pc_i ->density;
        const dfloat3  I_i = pc_i ->I;
        const dfloat  r_i = pc_i->radius;
        const dfloat3 pos_i = pc_i->pos;
        dfloat3 v_i = pc_i->vel;
        dfloat3 w_i = pc_i->w;

        // Particle i info (column)
        const dfloat  m_j = pc_j ->volume * pc_j ->density;
        const dfloat3  I_j = pc_j ->I;
        const dfloat  r_j = pc_j->radius;
        const dfloat3 pos_j = pc_j->pos;
        dfloat3 v_j = pc_j->vel;
        dfloat3 w_j = pc_j->w;

        // determine normal vector
        n.x = pos_i.x-pos_j.x;
        n.y = pos_i.y-pos_j.y;
        n.z = pos_i.z-pos_j.z;

        mag = n.x*n.x+n.y*n.y+n.z*n.z;
        dfloat const dist_abs = sqrt(mag);

        if(dist_abs <= r_i+r_j){

            n.x = n.x/(dist_abs);
            n.y = n.y/(dist_abs);
            n.z = n.z/(dist_abs);

            px = -n.x*(r_i+r_j-dist_abs);
            py = -n.y*(r_i+r_j-dist_abs);
            pz = -n.z*(r_i+r_j-dist_abs);

            // relative velocity vector
            G_0.x = v_i.x-v_j.x;
            G_0.y = v_i.y-v_j.y;
            G_0.z = v_i.z-v_j.z;

            G_c_0.x = G_0.x + r_i*(w_i.y*n.z-w_i.z*n.y)+r_j*(w_j.y*n.z-w_j.z*n.y);
            G_c_0.y = G_0.y + r_i*(w_i.z*n.x-w_i.x*n.z)+r_j*(w_j.z*n.x-w_j.x*n.z);
            G_c_0.z = G_0.z + r_i*(w_i.x*n.y-w_i.y*n.x)+r_j*(w_j.x*n.y-w_j.y*n.x);
        
            G_ct_0.x = G_0.x + r_i*(w_i.y*n.z-w_i.z*n.y)+r_j*(w_j.y*n.z-w_j.z*n.y) - (G_c_0.x*n.x+G_c_0.y*n.y+G_c_0.z*n.z)*n.x;
            G_ct_0.y = G_0.y + r_i*(w_i.z*n.x-w_i.x*n.z)+r_j*(w_j.z*n.x-w_j.x*n.z) - (G_c_0.x*n.x+G_c_0.y*n.y+G_c_0.z*n.z)*n.y;
            G_ct_0.z = G_0.z + r_i*(w_i.x*n.y-w_i.y*n.x)+r_j*(w_j.x*n.y-w_j.y*n.x) - (G_c_0.x*n.x+G_c_0.y*n.y+G_c_0.z*n.z)*n.z;
        
            mag = G_ct_0.x*G_ct_0.x+G_ct_0.y*G_ct_0.y+G_ct_0.z*G_ct_0.z;
            mag=sqrt(mag);

            if (mag != 0){
                //tangential vector
                t.x = G_ct_0.x/mag;
                t.y = G_ct_0.y/mag;
                t.z = G_ct_0.z/mag;
            }else{
                t.x = 0.0;
                t.y = 0.0;
                t.z = 0.0;
            }
            dfloat nG_0;

            nG_0 = (n.x*G_0.x+n.y*G_0.y+n.z*G_0.z);

            // translational velocity change
            const dfloat dvx_i = - (n.x+FRICTION_COEF*t.x)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j));  
            const dfloat dvy_i = - (n.y+FRICTION_COEF*t.y)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j));
            const dfloat dvz_i = - (n.z+FRICTION_COEF*t.z)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j));

            const dfloat dvx_j = + (n.x+FRICTION_COEF*t.x)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j));
            const dfloat dvy_j = + (n.y+FRICTION_COEF*t.y)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j));
            const dfloat dvz_j = + (n.z+FRICTION_COEF*t.z)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j));

            //rotational velocity change
            const dfloat dwx_i = - (2.5/r_i)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j))*FRICTION_COEF*(n.y*t.z-n.z*t.y);
            const dfloat dwy_i = - (2.5/r_i)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j))*FRICTION_COEF*(n.z*t.x-n.x*t.z);
            const dfloat dwz_i = - (2.5/r_i)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j))*FRICTION_COEF*(n.x*t.y-n.y*t.x);
        
            const dfloat dwx_j = - (2.5/r_j)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j))*FRICTION_COEF*(n.y*t.z-n.z*t.y);
            const dfloat dwy_j = - (2.5/r_j)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j))*FRICTION_COEF*(n.z*t.x-n.x*t.z);
            const dfloat dwz_j = - (2.5/r_j)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j))*FRICTION_COEF*(n.x*t.y-n.y*t.x);

            // particle velocity update
            dfloat add_dist = 1e-3;
            if(pc_i->movable && pc_j->movable){
    
                atomicAdd(&(pc_i->f.x), (dvx_i*m_i));
                atomicAdd(&(pc_i->f.y), (dvy_i*m_i));
                atomicAdd(&(pc_i->f.z), (dvz_i*m_i));

                atomicAdd(&(pc_i->pos.x), -px*(0.5 + add_dist));
                atomicAdd(&(pc_i->pos.y), -py*(0.5 + add_dist));
                atomicAdd(&(pc_i->pos.z), -pz*(0.5 + add_dist));

                atomicAdd(&(pc_i->M.x), (dwx_i*I_i.x));
                atomicAdd(&(pc_i->M.y), (dwy_i*I_i.y));
                atomicAdd(&(pc_i->M.z), (dwz_i*I_i.z));


                // Force negative in particle j (row)
                atomicAdd(&(pc_j->f.x), dvx_j*m_j);
                atomicAdd(&(pc_j->f.y), dvy_j*m_j);
                atomicAdd(&(pc_j->f.z), dvz_j*m_j);

                atomicAdd(&(pc_j->pos.x), px*(0.5 + add_dist));
                atomicAdd(&(pc_j->pos.y), py*(0.5 + add_dist));
                atomicAdd(&(pc_j->pos.z), pz*(0.5 + add_dist));

                atomicAdd(&(pc_i->M.x), dwx_j* I_j.x);
                atomicAdd(&(pc_i->M.y), dwy_j* I_j.y);
                atomicAdd(&(pc_i->M.z), dwz_j* I_j.z);
            }                  

            //update node velocities
            dfloat xIBM,yIBM,zIBM;
            for(int i = 0; i < particlesNodes.numNodes; i++){
                if ( particlesNodes.particleCenterIdx[i] == column){
                    xIBM = particlesNodes.pos.x[i];
                    yIBM = particlesNodes.pos.y[i];
                    zIBM = particlesNodes.pos.z[i];

                    particlesNodes.vel.x[i] = v_i.x + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
                    particlesNodes.vel.y[i] = v_i.y + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
                    particlesNodes.vel.z[i] = v_i.z + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));

                    particlesNodes.vel_old.x[i] = particlesNodes.vel_old.x[i] + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
                    particlesNodes.vel_old.y[i] = particlesNodes.vel_old.y[i] + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
                    particlesNodes.vel_old.z[i] = particlesNodes.vel_old.z[i] + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));
                }
                if ( particlesNodes.particleCenterIdx[i] == row){
                    xIBM = particlesNodes.pos.x[i];
                    yIBM = particlesNodes.pos.y[i];
                    zIBM = particlesNodes.pos.z[i];

                    particlesNodes.vel.x[i] = v_j.x + (w_j.y * (zIBM - pos_j.z) - w_j.z * (yIBM - pos_j.y));
                    particlesNodes.vel.y[i] = v_j.y + (w_j.z * (xIBM - pos_j.x) - w_j.x * (zIBM - pos_j.z));
                    particlesNodes.vel.z[i] = v_j.z + (w_j.x * (yIBM - pos_j.y) - w_j.y * (xIBM - pos_j.x));

                    particlesNodes.vel_old.x[i] = particlesNodes.vel_old.x[i] + (w_j.y * (zIBM - pos_j.z) - w_j.z * (yIBM - pos_j.y));
                    particlesNodes.vel_old.y[i] = particlesNodes.vel_old.y[i] + (w_j.z * (xIBM - pos_j.x) - w_j.x * (zIBM - pos_j.z));
                    particlesNodes.vel_old.z[i] = particlesNodes.vel_old.z[i] + (w_j.x * (yIBM - pos_j.y) - w_j.y * (xIBM - pos_j.x));
                }
            }
        } //if mag dist < sum radius

    } //colision between particles

}
#endif //HARD_SPHERE

#endif // !IBM