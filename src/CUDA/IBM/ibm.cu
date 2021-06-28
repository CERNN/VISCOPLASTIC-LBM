#include "ibm.h"

#ifdef IBM


__host__
void immersedBoundaryMethod(
    ParticlesSoA particles,
    Macroscopics* __restrict__ macr,
    IBMMacrsAux ibmMacrsAux,
    Populations* const __restrict__ pop,
    dim3 gridLBM,
    dim3 threadsLBM,
    cudaStream_t streamLBM[N_GPUS],
    cudaStream_t streamIBM[N_GPUS],
    unsigned int step,
    ParticleEulerNodesUpdate* pEulerNodes
    )
{
    // TODO: Update kernels to multi GPU

    // Update particle center position and its old values
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    gpuUpdateParticleOldValues<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
        particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

    // Grid for only  z-borders
    dim3 copyMacrGrid = gridLBM;
    // Grid for full domain, including z-borders
    dim3 borderMacrGrid = gridLBM; 
    // Only 1 in z
    copyMacrGrid.z = MACR_BORDER_NODES;
    borderMacrGrid.z += MACR_BORDER_NODES*2;

    unsigned int gridNodesIBM[N_GPUS];
    unsigned int threadsNodesIBM[N_GPUS];
    for(int i = 0; i < N_GPUS; i++){
        threadsNodesIBM[i] = 64;
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        unsigned int pNumNodes = particles.nodesSoA[i].numNodes;
        gridNodesIBM[i] = pNumNodes % threadsNodesIBM[i] ? pNumNodes / threadsNodesIBM[i] + 1 : pNumNodes / threadsNodesIBM[i];
    }
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));

    // Size of shared memory to use for optimization in interpolation/spread
    // const unsigned int sharedMemInterpSpread = threadsNodesIBM * sizeof(dfloat3) * 2;
    #if IBM_EULER_OPTIMIZATION

    // Grid size for euler nodes update
    
    for(int i = 0; i < N_GPUS; i++){
        if(pEulerNodes->currEulerNodes[i] > 0){
            dim3 currGrid(pEulerNodes->currEulerNodes[i]/64+(pEulerNodes->currEulerNodes[i]%64? 1 : 0), 1, 1);
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            // Update macroscopics post boundary conditions and reset forces
            gpuUpdateMacrIBM<<<currGrid, 64, 0, streamIBM[i]>>>(pop[i], macr[i], ibmMacrsAux, i,
                pEulerNodes->eulerIndexesUpdate[i], pEulerNodes->currEulerNodes[i]);
            checkCudaErrors(cudaStreamSynchronize(streamIBM[i]));
            getLastCudaError("IBM update macr euler error\n");
        }
    }
    #else
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        // Update macroscopics post boundary conditions and reset forces
        gpuUpdateMacrIBM<<<gridLBM, threadsLBM, 0, streamLBM[i]>>>(pop[i], macr[i], ibmMacrsAux, i);
        checkCudaErrors(cudaStreamSynchronize(streamLBM[i]));
        gpuResetBorderMacrAuxIBM<<<copyMacrGrid, threadsLBM, 0, streamLBM[i]>>>(ibmMacrsAux, i);
        checkCudaErrors(cudaStreamSynchronize(streamLBM[i]));
        getLastCudaError("IBM update macr error\n");
    }
    #endif


    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        int nxt = (i+1) % N_GPUS;
        // Copy macroscopics
        gpuCopyBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[i]>>>(macr[i], macr[nxt]);
        checkCudaErrors(cudaStreamSynchronize(streamLBM[i]));
        getLastCudaError("Copy macroscopics border error\n");
        // If GPU has nodes in it
        if(particles.nodesSoA[i].numNodes > 0){
            // Reset forces in all IBM nodes;
            gpuResetNodesForces<<<gridNodesIBM[i], threadsNodesIBM[i], 0, streamIBM[i]>>>(particles.nodesSoA[i]);
            checkCudaErrors(cudaStreamSynchronize(streamIBM[i]));
            getLastCudaError("Reset IBM nodes forces error\n");
        }
    }


    // Calculate collision force between particles
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    gpuParticlesCollision<<<GRID_PCOLLISION_IBM, THREADS_PCOLLISION_IBM, 0, streamIBM[0]>>>(particles.pCenterArray,step);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0])); 

    // First update particle velocity using body center force and constant forces
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    gpuUpdateParticleCenterVelocityAndRotation <<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0] >>>(
        particles.pCenterArray);
    getLastCudaError("IBM update particle center velocity error\n");
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

    for (int i = 0; i < IBM_MAX_ITERATION; i++)
    {
        for(int j = 0; j < N_GPUS; j++){
            // If GPU has nodes in it
            if(particles.nodesSoA[j].numNodes > 0){
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
                // Make the interpolation of LBM and spreading of IBM forces
                gpuForceInterpolationSpread<<<gridNodesIBM[j], threadsNodesIBM[j], 
                    0, streamIBM[j]>>>(
                    particles.nodesSoA[j], particles.pCenterArray, macr[j], ibmMacrsAux, j);
                checkCudaErrors(cudaStreamSynchronize(streamIBM[j]));
                getLastCudaError("IBM interpolation spread error\n");
            }
        }

        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
        // Update particle velocity using body center force and constant forces
        gpuUpdateParticleCenterVelocityAndRotation<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
            particles.pCenterArray);
        checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
        getLastCudaError("IBM update particle center velocity error\n");

        // Sum border macroscopics
        for(int j = 0; j < N_GPUS; j++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
            int nxt = j+1;
            int prv = j-1;
            if(nxt < N_GPUS)
                gpuSumBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[nxt], ibmMacrsAux, j, 1);
            if(prv >= 0)
                gpuSumBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[prv], ibmMacrsAux, j, -1);
            checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
            getLastCudaError("Sum border macroscopics error\n");
        }

        #if IBM_EULER_OPTIMIZATION

        for(int j = 0; j < N_GPUS; j++){
            if(pEulerNodes->currEulerNodes[j] > 0){
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
                dim3 currGrid(pEulerNodes->currEulerNodes[j]/64+(pEulerNodes->currEulerNodes[j]%64? 1 : 0), 1, 1);
                gpuEulerSumIBMAuxsReset<<<currGrid, 64, 0, streamLBM[j]>>>(macr[j], ibmMacrsAux,
                    pEulerNodes->eulerIndexesUpdate[j], pEulerNodes->currEulerNodes[j], j);
                checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
                getLastCudaError("IBM sum auxiliary values error\n");
            }
        }
        #else
        for(int j = 0; j < N_GPUS; j++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
            gpuEulerSumIBMAuxsReset<<<borderMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[j], ibmMacrsAux, j);
            checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
        }
        #endif

    }

    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    // Update particle center position and its old values
    gpuParticleMovement<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
        particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    getLastCudaError("IBM particle movement error\n");

    for(int i = 0; i < N_GPUS; i++){
        // If GPU has nodes in it
        if(particles.nodesSoA[i].numNodes > 0){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            // Update particle nodes positions
            gpuParticleNodeMovement<<<gridNodesIBM[i], threadsNodesIBM[i], 0, streamIBM[i]>>>(
                particles.nodesSoA[i], particles.pCenterArray);
            checkCudaErrors(cudaStreamSynchronize(streamIBM[i]));
            getLastCudaError("IBM particle movement error\n");
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void gpuForceInterpolationSpread(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES],
    Macroscopics const macr,
    IBMMacrsAux ibmMacrsAux,
    const int n_gpu)
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

   // Base position is memory, so it discount the nodes in Z in others gpus
   /*const int posBase[3] = {
       (int)(xIBM+0.5)-P_DIST+1, 
       (int)(yIBM+0.5)-P_DIST+1, 
       (int)(zIBM+0.5)-P_DIST+1-n_gpu*NZ}
   ;*/
    int posBase[3] = { 
        int(xIBM - P_DIST + 0.5 - (xIBM < 1.0)), 
        int(yIBM - P_DIST + 0.5 - (yIBM < 1.0)), 
        int(zIBM - P_DIST + 0.5 - (zIBM < 1.0)-n_gpu*NZ) 
    };
    // Maximum stencil index for each direction xyz ("index" to stop)
    const int maxIdx[3] = {
        #ifdef IBM_BC_X_WALL
            ((posBase[0]+P_DIST*2-1) < (int)NX)? P_DIST*2-1 : ((int)NX-1-posBase[0])
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            P_DIST*2-1
        #endif //IBM_BC_X_PERIODIC
        ,
        #ifdef IBM_BC_Y_WALL 
             ((posBase[1]+P_DIST*2-1) < (int)NY)? P_DIST*2-1 : ((int)NY-1-posBase[1])
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            P_DIST*2-1
        #endif //IBM_BC_Y_PERIODIC
        , 
        #ifdef IBM_BC_Z_WALL 
            ((posBase[2]+P_DIST*2-1) < (int)(n_gpu != N_GPUS-1? NZ : NZ+P_DIST))? P_DIST*2-1 : ((int)NZ-1-posBase[2])
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            P_DIST*2-1
        #endif //IBM_BC_Z_PERIODIC
    };
    // Minimum stencil index for each direction xyz ("index" to start)
    const int minIdx[3] = {
        #ifdef IBM_BC_X_WALL
            (posBase[0] >= 0)? 0 : -posBase[0]
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            0
        #endif //IBM_BC_X_PERIODIC
        ,
        #ifdef IBM_BC_Y_WALL 
            (posBase[1] >= 0)? 0 : -posBase[1]
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            0
        #endif //IBM_BC_Y_PERIODIC
        , 
        #ifdef IBM_BC_Z_WALL 
            (posBase[2] >= (int)(n_gpu != 0? 0 : -P_DIST))? 0 : -posBase[2]
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

    // printf("posN %.2f %.2f %.2f posBase %d %d %d minIdx %d %d %d maxIdx %d %d %d pNodes %d i %d\n", 
    //     pos[0], pos[1], pos[2], posBase[0], posBase[1], posBase[2], 
    //     minIdx[0], minIdx[1], minIdx[2], maxIdx[0], maxIdx[1], maxIdx[2],
    //     particlesNodes.numNodes, i);


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
                        IBM_BC_X_0 + (posBase[0]+xi + IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0)%(IBM_BC_X_E - IBM_BC_X_0)
                    #endif //IBM_BC_X_PERIODIC
                    ,
                    #ifdef IBM_BC_Y_WALL 
                        posBase[1]+yj
                    #endif //IBM_BC_Y_WALL
                    #ifdef IBM_BC_Y_PERIODIC
                        IBM_BC_Y_0 + (posBase[1]+yj + IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0)%(IBM_BC_Y_E - IBM_BC_Y_0)
                    #endif //IBM_BC_Y_PERIODIC
                    , 
                    #ifdef IBM_BC_Z_WALL  // +MACR_BORDER_NODES in z because of the ghost nodes
                        posBase[2]+zk +MACR_BORDER_NODES
                    #endif //IBM_BC_Z_WALL
                    #ifdef IBM_BC_Z_PERIODIC
                        IBM_BC_Z_0 + (posBase[2]+zk + IBM_BC_Z_E - IBM_BC_Z_0-IBM_BC_Z_0)%(IBM_BC_Z_E - IBM_BC_Z_0)
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

    dfloat dx = xIBM - x_pc;
    dfloat dy = yIBM - y_pc;
    dfloat dz = zIBM - z_pc; 

    #ifdef IBM_BC_X_PERIODIC
    if(abs(dx) > (dfloat)((IBM_BC_X_E - IBM_BC_X_0))/2.0){
        if(dx < 0)
            dx = (xIBM + (IBM_BC_X_E - IBM_BC_X_0)) - x_pc;
        else
            dx = (xIBM - (IBM_BC_X_E - IBM_BC_X_0)) - x_pc;
    }
    #endif //IBM_BC_X_PERIODIC
    
    #ifdef IBM_BC_Y_PERIODIC
    if(abs(dy) > (dfloat)((IBM_BC_Y_E - IBM_BC_Y_0))/2.0){
        if(dy < 0)
            dy = (yIBM + (IBM_BC_Y_E - IBM_BC_Y_0)) - y_pc;
        else
            dy = (yIBM - (IBM_BC_Y_E - IBM_BC_Y_0)) - y_pc;
    }
    #endif //IBM_BC_Y_PERIODIC

    #ifdef IBM_BC_Z_PERIODIC
    if(abs(dz) > (dfloat)((IBM_BC_Z_E - IBM_BC_Z_0))/2.0){
        if(dz < 0)
            dz = (zIBM + (IBM_BC_Z_E - IBM_BC_Z_0)) - z_pc;
        else
            dz = (zIBM - (IBM_BC_Z_E - IBM_BC_Z_0)) - z_pc;
    }
    #endif //IBM_BC_Z_PERIODIC

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
        ux_calc = vx_pc + (wy_pc * (dz) - wz_pc * (dy));
        uy_calc = vy_pc + (wz_pc * (dx) - wx_pc * (dz));
        uz_calc = vz_pc + (wx_pc * (dy) - wy_pc * (dx));
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
                        IBM_BC_X_0 + (posBase[0]+xi + IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0)%(IBM_BC_X_E - IBM_BC_X_0)
                    #endif //IBM_BC_X_PERIODIC
                    ,
                    #ifdef IBM_BC_Y_WALL 
                        posBase[1]+yj
                    #endif //IBM_BC_Y_WALL
                    #ifdef IBM_BC_Y_PERIODIC
                        IBM_BC_Y_0 + (posBase[1]+yj + IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0)%(IBM_BC_Y_E - IBM_BC_Y_0)
                    #endif //IBM_BC_Y_PERIODIC
                    , 
                    #ifdef IBM_BC_Z_WALL  // +MACR_BORDER_NODES in z because of the ghost nodes
                        posBase[2]+zk +MACR_BORDER_NODES
                    #endif //IBM_BC_Z_WALL
                    #ifdef IBM_BC_Z_PERIODIC
                        IBM_BC_Z_0 + (posBase[2]+zk + IBM_BC_Z_E - IBM_BC_Z_0-IBM_BC_Z_0)%(IBM_BC_Z_E - IBM_BC_Z_0)
                    #endif //IBM_BC_Z_PERIODIC
                );


                atomicAdd(&(ibmMacrsAux.fAux[n_gpu].x[idx]), -deltaF.x * aux);
                atomicAdd(&(ibmMacrsAux.fAux[n_gpu].y[idx]), -deltaF.y * aux);
                atomicAdd(&(ibmMacrsAux.fAux[n_gpu].z[idx]), -deltaF.z * aux);

                // Update velocities field
                const dfloat inv_rho = 1 / macr.rho[idx];
                atomicAdd(&(ibmMacrsAux.velAux[n_gpu].x[idx]), 0.5 * -deltaF.x * aux * inv_rho);
                atomicAdd(&(ibmMacrsAux.velAux[n_gpu].y[idx]), 0.5 * -deltaF.y * aux * inv_rho);
                atomicAdd(&(ibmMacrsAux.velAux[n_gpu].z[idx]), 0.5 * -deltaF.z * aux * inv_rho);
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
        (dy) * deltaF.z - (dz) * deltaF.y,
        (dz) * deltaF.x - (dx) * deltaF.z,
        (dx) * deltaF.y - (dy) * deltaF.x
    );

    atomicAdd(&(particleCenters[idx].f.x), deltaF.x);
    atomicAdd(&(particleCenters[idx].f.y), deltaF.y);
    atomicAdd(&(particleCenters[idx].f.z), deltaF.z);

    atomicAdd(&(particleCenters[idx].M.x), deltaMomentum.x);
    atomicAdd(&(particleCenters[idx].M.y), deltaMomentum.y);
    atomicAdd(&(particleCenters[idx].M.z), deltaMomentum.z);
}

__global__
void gpuUpdateMacrIBM(Populations pop, Macroscopics macr, IBMMacrsAux ibmMacrsAux, int n_gpu
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

    int x = idx % NX;
    int y = (idx / NX) % NY;
    // remove border nodes in z
    int z = (idx / NX) / NY - MACR_BORDER_NODES;

    #else

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
       return;

    // +MACR_BORDER_NODES because of the ghost nodes in z
    size_t idx = idxScalar(x, y, z+MACR_BORDER_NODES);
    #endif

    // Reset values from auxiliary vectors
    ibmMacrsAux.velAux[n_gpu].x[idx] = 0;
    ibmMacrsAux.velAux[n_gpu].y[idx] = 0;
    ibmMacrsAux.velAux[n_gpu].z[idx] = 0;
    ibmMacrsAux.fAux[n_gpu].x[idx] = 0;
    ibmMacrsAux.fAux[n_gpu].y[idx] = 0;
    ibmMacrsAux.fAux[n_gpu].z[idx] = 0;

    
    // check if it is some kind of border, if so, just not update
    if(z < 0 || z >= NZ)
        return;

    // load populations
    dfloat fNode[Q];
    for (unsigned char i = 0; i < Q; i++)
        fNode[i] = pop.pop[idxPop(x, y, z, i)];

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
}

__global__
void gpuResetBorderMacrAuxIBM(IBMMacrsAux ibmMacrsAux, int n_gpu){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= MACR_BORDER_NODES)
       return;

    const size_t idx_left = idxScalar(x, y, z);
    const size_t idx_right = idxScalar(x, y, z+MACR_BORDER_NODES+NZ);

    dfloat3SoA vel = ibmMacrsAux.velAux[n_gpu];
    dfloat3SoA f = ibmMacrsAux.fAux[n_gpu];

    vel.x[idx_left] = 0; vel.x[idx_right] = 0;
    vel.y[idx_left] = 0; vel.y[idx_right] = 0;
    vel.z[idx_left] = 0; vel.z[idx_right] = 0;

    f.x[idx_left] = 0; f.x[idx_right] = 0;
    f.y[idx_left] = 0; f.y[idx_right] = 0;
    f.z[idx_left] = 0; f.z[idx_right] = 0;
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
void gpuCopyBorderMacr(Macroscopics macrBase, Macroscopics macrNext)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= MACR_BORDER_NODES)
       return;

    // values to read from base and write to next
    const int zm_w = -z-1, zm_r = NZ-1-z;
    // values to read from next and write to base
    const int zp_w = NZ+z, zp_r = z;

    // FIXME: wrong, it needs to consider the added values in ghost nodes. 
    // This may be used from the auxiliary velocities vectors 
    // write to next
    size_t idx_m_w = idxScalar(x, y, zm_w+MACR_BORDER_NODES);
    size_t idx_m_r = idxScalar(x, y, zm_r+MACR_BORDER_NODES);
    macrNext.u.x[idx_m_w] = macrBase.u.x[idx_m_r];
    macrNext.u.y[idx_m_w] = macrBase.u.y[idx_m_r];
    macrNext.u.z[idx_m_w] = macrBase.u.z[idx_m_r];
    macrNext.f.x[idx_m_w] = macrBase.f.x[idx_m_r];
    macrNext.f.y[idx_m_w] = macrBase.f.y[idx_m_r];
    macrNext.f.z[idx_m_w] = macrBase.f.z[idx_m_r];

    // write to base
    size_t idx_p_w = idxScalar(x, y, zp_w+MACR_BORDER_NODES);
    size_t idx_p_r = idxScalar(x, y, zp_r+MACR_BORDER_NODES);
    macrBase.u.x[idx_p_w] = macrNext.u.x[idx_p_r];
    macrBase.u.y[idx_p_w] = macrNext.u.y[idx_p_r];
    macrBase.u.z[idx_p_w] = macrNext.u.z[idx_p_r];
    macrBase.f.x[idx_p_w] = macrNext.f.x[idx_p_r];
    macrBase.f.y[idx_p_w] = macrNext.f.y[idx_p_r];
    macrBase.f.z[idx_p_w] = macrNext.f.z[idx_p_r];
}


__global__
void gpuSumBorderMacr(Macroscopics macr, IBMMacrsAux ibmMacrsAux, int n_gpu, int borders){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= MACR_BORDER_NODES)
       return;
    
    size_t read, write;
    // macr to the right of ibmMacrsAux
    if(borders == 1){
        // read from right ghost nodes of ibmMacrsAux
        read = idxScalar(x, y, NZ+z+MACR_BORDER_NODES);
        // write to left ghost nodes of macr
        write = idxScalar(x, y, MACR_BORDER_NODES-z);
    }
    // macr to the left of ibmMacrsAux
    else if(borders == -1){
        // read from left ghost nodes of ibmMacrsAux
        read = idxScalar(x, y, MACR_BORDER_NODES-z);
        // write to right ghost nodes of macr
        write = idxScalar(x, y, NZ+z+MACR_BORDER_NODES);
    }
    // invalid
    else{
        return;
    }

    // Sum velocities
    macr.u.x[write] += ibmMacrsAux.velAux[n_gpu].x[read];
    macr.u.y[write] += ibmMacrsAux.velAux[n_gpu].y[read];
    macr.u.z[write] += ibmMacrsAux.velAux[n_gpu].z[read];
    // Sum forces
    macr.f.x[write] += ibmMacrsAux.fAux[n_gpu].x[read];
    macr.f.y[write] += ibmMacrsAux.fAux[n_gpu].y[read];
    macr.f.z[write] += ibmMacrsAux.fAux[n_gpu].z[read];
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
        pc->pos.x = IBM_BC_X_0 + std::fmod((dfloat)(pc->pos.x + dx + IBM_BC_X_E - IBM_BC_X_0 - IBM_BC_X_0) , (dfloat)(IBM_BC_X_E - IBM_BC_X_0)); 
    #endif //IBM_BC_X_PERIODIC

    #ifdef IBM_BC_Y_WALL
        pc->pos.y +=  (pc->vel.y * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    #endif //IBM_BC_Y_WALL
    #ifdef IBM_BC_Y_PERIODIC
        dfloat dy =  (pc->vel.y * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->pos.y = IBM_BC_Y_0 + std::fmod((dfloat)(pc->pos.y + dy + IBM_BC_Y_E - IBM_BC_Y_0 - IBM_BC_Y_0) , (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0));
    #endif //IBM_BC_Y_PERIODIC

    #ifdef IBM_BC_Z_WALL
        pc->pos.z +=  (pc->vel.z * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    #endif //IBM_BC_Z_WALL
    #ifdef IBM_BC_Z_PERIODIC
        dfloat dz =  (pc->vel.z * IBM_MOVEMENT_DISCRETIZATION + pc->vel_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->pos.z = IBM_BC_Z_0 + std::fmod((dfloat)(pc->pos.z + dz + IBM_BC_Z_E - IBM_BC_Z_0 - IBM_BC_Z_0) , (dfloat)(IBM_BC_Z_E - IBM_BC_Z_0)); 
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
        dfloat dx,dy,dz;
        dfloat new_pos_x,new_pos_y,new_pos_z;

        dx = pc.pos.x - pc.pos_old.x;
        dy = pc.pos.y - pc.pos_old.y;
        dz = pc.pos.z - pc.pos_old.z;

        #ifdef IBM_BC_X_WALL
            particlesNodes.pos.x[i] += dx;
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            if(abs(dx) > (dfloat)(IBM_BC_X_E - IBM_BC_X_0)/2.0){
                if(pc.pos.x < pc.pos_old.x )
                    dx = (pc.pos.x  + (IBM_BC_X_E - IBM_BC_X_0)) - pc.pos_old.x;
                else
                    dx = (pc.pos.x  - (IBM_BC_X_E - IBM_BC_X_0)) - pc.pos_old.x;
            }
            particlesNodes.pos.x[i] = IBM_BC_X_0 + std::fmod((dfloat)(particlesNodes.pos.x[i] + dx + (IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0)),(dfloat)(IBM_BC_X_E - IBM_BC_X_0));
        #endif //IBM_BC_X_PERIODIC


        #ifdef IBM_BC_Y_WALL
            particlesNodes.pos.y[i] += dy;
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            if(abs(dy) > (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0)/2.0){
                if(pc.pos.y < pc.pos_old.y )
                    dy = (pc.pos.y  + (IBM_BC_Y_E - IBM_BC_Y_0)) - pc.pos_old.y;
                else
                    dy = (pc.pos.y  - (IBM_BC_Y_E - IBM_BC_Y_0)) - pc.pos_old.y;
            }
            particlesNodes.pos.y[i] = IBM_BC_Y_0 + std::fmod((dfloat)(particlesNodes.pos.y[i] + dy + (IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0)),(dfloat)(IBM_BC_Y_E - IBM_BC_Y_0));
        #endif // IBM_BC_Y_PERIODIC


        #ifdef IBM_BC_Z_WALL
            particlesNodes.pos.z[i] += dz;
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            if(abs(dz) > (dfloat)(IBM_BC_Z_E - IBM_BC_Z_0)/2.0){
                if(pc.pos.y < pc.pos_old.y )
                    dz = (pc.pos.z  + (IBM_BC_Z_E - IBM_BC_Z_0)) - pc.pos_old.z;
                else
                    dz = (pc.pos.z  - (IBM_BC_Z_E - IBM_BC_Z_0)) - pc.pos_old.z;
            }
            particlesNodes.pos.z[i] = IBM_BC_Z_0 + std::fmod((dfloat)(particlesNodes.pos.z[i] + dz + (IBM_BC_Z_E - IBM_BC_Z_0-IBM_BC_Z_0)),(dfloat)(IBM_BC_Z_E - IBM_BC_Z_0));
        #endif //IBM_BC_Z_PERIODIC

        return;
    }

    // TODO: these variables are the same for every particle center, optimize it
    
    const dfloat q0 = cos(0.5*w_norm);
    const dfloat qi = (pc.w_avg.x/w_norm) * sin (0.5*w_norm);
    const dfloat qj = (pc.w_avg.y/w_norm) * sin (0.5*w_norm);
    const dfloat qk = (pc.w_avg.z/w_norm) * sin (0.5*w_norm);

    const dfloat tq0m1 = (q0*q0) - 0.5;

    dfloat x_vec = particlesNodes.pos.x[i] - pc.pos_old.x;
    dfloat y_vec = particlesNodes.pos.y[i] - pc.pos_old.y;
    dfloat z_vec = particlesNodes.pos.z[i] - pc.pos_old.z;


    #ifdef IBM_BC_X_PERIODIC
        if(abs(x_vec) > (dfloat)(IBM_BC_X_E - IBM_BC_X_0)/2.0){
            if(particlesNodes.pos.x[i] < pc.pos_old.x )
                x_vec = (particlesNodes.pos.x[i]  + (IBM_BC_X_E - IBM_BC_X_0)) - pc.pos_old.x;
            else
                x_vec = (particlesNodes.pos.x[i]  - (IBM_BC_X_E - IBM_BC_X_0)) - pc.pos_old.x;
        }
    #endif //IBM_BC_X_PERIODIC


    #ifdef IBM_BC_Y_PERIODIC
        if(abs(y_vec) > (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0)/2.0){
            if(pc.pos.y < pc.pos_old.y )
                y_vec = (particlesNodes.pos.y[i]  + (IBM_BC_Y_E - IBM_BC_Y_0)) - pc.pos_old.y;
            else
                y_vec = (particlesNodes.pos.y[i]  - (IBM_BC_Y_E - IBM_BC_Y_0)) - pc.pos_old.y;
        }
    #endif //IBM_BC_Y_PERIODIC


    #ifdef IBM_BC_Z_PERIODIC
        if(abs(z_vec) > (dfloat)(IBM_BC_Z_E - IBM_BC_Z_0)/2.0){
            if(pc.pos.y < pc.pos_old.y )
                z_vec = ( particlesNodes.pos.z[i]  + (IBM_BC_Z_E - IBM_BC_Z_0)) - pc.pos_old.z;
            else
                z_vec = ( particlesNodes.pos.z[i]  - (IBM_BC_Z_E - IBM_BC_Z_0)) - pc.pos_old.z;
        }
    #endif //IBM_BC_Z_PERIODIC


    
    dfloat new_pos_x = pc.pos.x + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec);
    dfloat new_pos_y = pc.pos.y + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec);
    dfloat new_pos_z = pc.pos.z + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec);

    #ifdef  IBM_BC_X_WALL
    particlesNodes.pos.x[i] =  new_pos_x;
    #endif //IBM_BC_X_WALL
    #ifdef  IBM_BC_Y_WALL
    particlesNodes.pos.y[i] =  new_pos_y;
    #endif //IBM_BC_Y_WALL
    #ifdef  IBM_BC_Z_WALL
    particlesNodes.pos.z[i] =  new_pos_z;
    #endif //IBM_BC_Z_WALL



    #ifdef  IBM_BC_X_PERIODIC
    particlesNodes.pos.x[i] =  IBM_BC_X_0 + std::fmod((dfloat)(new_pos_x + IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0),(dfloat)(IBM_BC_X_E - IBM_BC_X_0));
    #endif //IBM_BC_X_PERIODIC
    #ifdef  IBM_BC_Y_PERIODIC
    particlesNodes.pos.y[i] =  IBM_BC_Y_0 + std::fmod((dfloat)(new_pos_y + IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0),(dfloat)(IBM_BC_Y_E - IBM_BC_Y_0));
    #endif //IBM_BC_Y_PERIODIC
    #ifdef  IBM_BC_Z_PERIODIC
    particlesNodes.pos.z[i] =  IBM_BC_Z_0 + std::fmod((dfloat)(new_pos_z + IBM_BC_Z_E - IBM_BC_Z_0-IBM_BC_Z_0),(dfloat)(IBM_BC_Z_E - IBM_BC_Z_0));
    #endif //IBM_BC_Z_PERIODIC



    
}



#endif // !IBM