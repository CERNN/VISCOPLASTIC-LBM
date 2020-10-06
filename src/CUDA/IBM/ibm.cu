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
    cudaStream_t __restrict__ streamLBM[N_GPUS],
    cudaStream_t __restrict__ streamIBM[N_GPUS],
    unsigned int step)
{
    // Update particle center position and its old values
    gpuUpdateParticleOldValues<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
        particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

    if(!(step % 100)){
        ParticleCenter pc = particles.pCenterArray[0];
        printf("step %d\n", step); 
        printf("pos (%f, %f, %f)\n", pc.pos.x, pc.pos.y, pc.pos.z);
        printf("pos_old (%f, %f, %f)\n", pc.pos_old.x, pc.pos_old.y, pc.pos_old.z);
        printf("vel (%e, %e, %e)\n", pc.vel.x, pc.vel.y, pc.vel.z);
        printf("accel (%e, %e, %e)\n", pc.vel.x-pc.vel_old.x, pc.vel.y-pc.vel_old.y, pc.vel.z-pc.vel_old.z);
        std::string file_part;
        file_part += "parts";
        file_part += std::to_string(step);
        file_part += ".csv";
        FILE* outFile = fopen(file_part.c_str(), "w");
        if(outFile != nullptr)
        {
            fprintf(outFile, "x, y, z\n");
            for(int idx=0; idx < particles.nodesSoA.numNodes; idx++){
                fprintf(outFile, "%.3e, %.3e, %.3e\n", 
                    particles.nodesSoA.pos.x[idx],
                    particles.nodesSoA.pos.y[idx],
                    particles.nodesSoA.pos.z[idx]);
            }
            fclose(outFile);
        }
    }
    // TODO: Update it to multi GPU
    // Size of shared memory to use for optimization in interpolation/spread
    const unsigned int sharedMemInterpSpread = threadsNodesIBM * sizeof(dfloat3);

    // Update macroscopics post boundary conditions and reset forces
    gpuUpdateMacrResetForces<<<gridLBM, threadsLBM, 0, streamLBM[0]>>>(pop[0], macr[0], velsAuxIBM[0]);
    checkCudaErrors(cudaStreamSynchronize(streamLBM[0]));

    // Reset forces in all IBM nodes
    gpuResetNodesForces<<<gridNodesIBM, threadsNodesIBM, 0, streamIBM[0]>>>(particles.nodesSoA);
    // Calculate collision force between particles
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

    gpuParticlesCollision<<<GRID_PCOLLISION_IBM, THREADS_PCOLLISION_IBM, 0, streamIBM[0]>>>(particles.pCenterArray);
    checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));

    for (int i = 0; i < IBM_MAX_ITERATION; i++)
    {
        // Make the interpolation of LBM and spreading of IBM forces
        gpuForceInterpolationSpread<<<gridNodesIBM, threadsNodesIBM, 
            sharedMemInterpSpread, streamIBM[0]>>>(
            particles.nodesSoA, particles.pCenterArray, macr[0], velsAuxIBM[0]);
        checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
        
        // Swapping velocity vectors
        dfloat* tmp = macr[0].ux;
        macr[0].ux = velsAuxIBM[0].x;
        velsAuxIBM[0].x = tmp;

        tmp = macr[0].uy;
        macr[0].uy = velsAuxIBM[0].y;
        velsAuxIBM[0].y = tmp;

        tmp = macr[0].uz;
        macr[0].uz = velsAuxIBM[0].z;
        velsAuxIBM[0].z = tmp;

        // Update particle velocity using body center force and constant forces
        gpuUpdateParticleCenterVelocityAndRotation<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
            particles.pCenterArray);

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

    // Synchronize and swap populations
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

    if (i >= particlesNodes.numNodes)
        return;

    dfloat aux; // aux variable for many things
    size_t idx; // index for many things

    const dfloat xIBM = particlesNodes.pos.x[i];
    const dfloat yIBM = particlesNodes.pos.y[i];
    const dfloat zIBM = particlesNodes.pos.z[i];

    // Minimum number of xyz for LBM interpolation
    const unsigned int xMin = (((int)xIBM - P_DIST) < 0) ? 0 : (int)xIBM - P_DIST;
    const unsigned int yMin = (((int)yIBM - P_DIST) < 0) ? 0 : (int)yIBM - P_DIST;
    const unsigned int zMin = (((int)zIBM - P_DIST) < 0) ? 0 : (int)zIBM - P_DIST;

    // Maximum number of xyz for LBM interpolation, excluding last
    // (e.g. NX goes just until NX-1)
    const unsigned int xMax = (((int)xIBM + 1 + P_DIST) > NX) ? NX : (int)xIBM + P_DIST + 1;
    const unsigned int yMax = (((int)yIBM + 1 + P_DIST) > NY) ? NY : (int)yIBM + P_DIST + 1;
    const unsigned int zMax = (((int)zIBM + 1 + P_DIST) > NZ) ? NZ : (int)zIBM + P_DIST + 1;

    if(xMin >= NX || yMin >= NY || zMin >= NZ || xMax <= 0 || yMax <= 0 || zMax <= 0)
        return;

    dfloat rhoVar = 0;
    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

    dfloat sumAux_interp = 0;
    //  Interpolation
    for (int z = zMin; z < zMax; z++)
    {
        for (int y = yMin; y < yMax; y++)
        {
            for (int x = xMin; x < xMax; x++)
            {
                idx = idxScalar(x, y, z);
                // Dirac delta (kernel)
                aux = stencil(x - xIBM) * stencil(y - yIBM) * stencil(z - zIBM);

                sumAux_interp += aux;

                rhoVar += macr.rho[idx] * aux;
                uxVar += macr.ux[idx] * aux;
                uyVar += macr.uy[idx] * aux;
                uzVar += macr.uz[idx] * aux;
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

    const dfloat3 deltaF = dfloat3(
        aux * (uxVar - ux_calc),
        aux * (uyVar - uy_calc),
        aux * (uzVar - uz_calc));

    // Calculate IBM forces
    const dfloat fxIBM = particlesNodes.f.x[i] + deltaF.x;
    const dfloat fyIBM = particlesNodes.f.y[i] + deltaF.y;
    const dfloat fzIBM = particlesNodes.f.z[i] + deltaF.z;

    dfloat sumAux_spread = 0;
    // Spreading
    for (int z = zMin; z < zMax; z++)
    {
        for (int y = yMin; y < yMax; y++)
        {
            for (int x = xMin; x < xMax; x++)
            {
                idx = idxScalar(x, y, z);

                // Dirac delta (kernel)
                aux = stencil(x - xIBM) * stencil(y - yIBM) * stencil(z - zIBM);
                sumAux_spread += aux;

                // TODO: update rho and velocities of LBM here, but with 
                // different array to not have concurrent problems with loading 
                // the velocities
                atomicAdd(&(macr.fx[idx]), -deltaF.x * aux);
                atomicAdd(&(macr.fy[idx]), -deltaF.y * aux);
                atomicAdd(&(macr.fz[idx]), -deltaF.z * aux);

                const dfloat inv_rho = 1 / macr.rho[idx];
                atomicAdd(&(velAuxIBM.x[idx]), 0.5 * -deltaF.x * aux * inv_rho);
                atomicAdd(&(velAuxIBM.y[idx]), 0.5 * -deltaF.y * aux * inv_rho);
                atomicAdd(&(velAuxIBM.z[idx]), 0.5 * -deltaF.z * aux * inv_rho);
            }
        }
    }
    // if(i == 0)
    //     printf("id %d xMin %d xMax %d xIBM %.2e deltaFz %.2e aux_spread %.2e aux_interp %.2e\n", 
    //     i, xMin, xMax, xIBM, deltaF.z, sumAux_spread, sumAux_interp);

    // Update node velocity
    particlesNodes.vel.x[i] = ux_calc;
    particlesNodes.vel.y[i] = uy_calc;
    particlesNodes.vel.z[i] = uz_calc;

    // Update node force
    particlesNodes.f.x[i] = fxIBM;
    particlesNodes.f.y[i] = fyIBM;
    particlesNodes.f.z[i] = fzIBM;

    // Update node delta force
    particlesNodes.deltaF.x[i] = deltaF.x;
    particlesNodes.deltaF.y[i] = deltaF.y;
    particlesNodes.deltaF.z[i] = deltaF.z;

    // Particle node delta momentum
    const dfloat3 deltaMomentum = dfloat3(
        (yIBM - y_pc) * deltaF.z - (zIBM - z_pc) * deltaF.y,
        (zIBM - z_pc) * deltaF.x - (xIBM - x_pc) * deltaF.z,
        (xIBM - x_pc) * deltaF.y - (yIBM - y_pc) * deltaF.x
    );

    // Add node force to particle center
    idx = particlesNodes.particleCenterIdx[i];

    // TODO: check if shared memory is more efficient
    atomicAdd(&(particleCenters[idx].f.x), deltaF.x);
    atomicAdd(&(particleCenters[idx].f.y), deltaF.y);
    atomicAdd(&(particleCenters[idx].f.z), deltaF.z);

    atomicAdd(&(particleCenters[idx].M.x), deltaMomentum.x);
    atomicAdd(&(particleCenters[idx].M.y), deltaMomentum.y);
    atomicAdd(&(particleCenters[idx].M.z), deltaMomentum.z);
}

__global__
void gpuUpdateMacrResetForces(Populations pop, Macroscopics macr, dfloat3SoA velAuxIBM)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t idx = idxScalar(x, y, z);

    // load populations
    dfloat fNode[Q];
    for (unsigned char i = 0; i < Q; i++)
        fNode[i] = pop.pop[idxPop(x, y, z, i)];

    macr.fx[idx] = FX;
    macr.fy[idx] = FY;
    macr.fz[idx] = FZ;

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
    macr.ux[idx] = uxVar;
    macr.uy[idx] = uyVar;
    macr.uz[idx] = uzVar;
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
    pc->vel.x = pc->vel_old.x + (0.5 * (pc->f_old.x + pc->f.x) * inv_volume + (PARTICLE_DENSITY - FLUID_DENSITY) * GX) / (PARTICLE_DENSITY);
    pc->vel.y = pc->vel_old.y + (0.5 * (pc->f_old.y + pc->f.y) * inv_volume + (PARTICLE_DENSITY - FLUID_DENSITY) * GY) / (PARTICLE_DENSITY);
    pc->vel.z = pc->vel_old.z + (0.5 * (pc->f_old.z + pc->f.z) * inv_volume + (PARTICLE_DENSITY - FLUID_DENSITY) * GZ) / (PARTICLE_DENSITY);

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
        wNew.x = pc->w_old.x + (0.5*(M.x + M_old.x) - (I.z - I.y)*0.25*(w_old.y + wAux.y)*(w_old.z + wAux.z))/I.x;
        wNew.y = pc->w_old.y + (0.5*(M.y + M_old.y) - (I.x - I.z)*0.25*(w_old.x + wAux.x)*(w_old.z + wAux.z))/I.y;
        wNew.z = pc->w_old.z + (0.5*(M.z + M_old.z) - (I.y - I.x)*0.25*(w_old.x + wAux.x)*(w_old.y + wAux.y))/I.z;

        error = (wNew.x - wAux.x)*(wNew.x - wAux.x)/(wNew.x*wNew.x);
        error += (wNew.y - wAux.y)*(wNew.y - wAux.y)/(wNew.y*wNew.y);
        error += (wNew.z - wAux.z)*(wNew.z - wAux.z)/(wNew.z*wNew.z);

        wAux.x = wNew.x;
        wAux.y = wNew.y;
        wAux.z = wNew.z;
    }
    // Store new velocities in particle center
    pc->w.x = 0; //wNew.x;
    pc->w.y = 0; //wNew.y;
    pc->w.z = 0; //wNew.z;
}

__global__
void gpuParticleMovement(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

    if(p >= NUM_PARTICLES)
        return;

    ParticleCenter *pc = &(particleCenters[p]);

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

    // if(w_norm <= 1e-8)
    // {
    //     particlesNodes.pos.x[i] += pc.pos.x - pc.pos_old.x;
    //     particlesNodes.pos.y[i] += pc.pos.y - pc.pos_old.y;
    //     particlesNodes.pos.z[i] += pc.pos.z - pc.pos_old.z;
    //     return;
    // }

    particlesNodes.pos.x[i] += (pc.pos.x - pc.pos_old.x);
    particlesNodes.pos.y[i] += (pc.pos.y - pc.pos_old.y);
    particlesNodes.pos.z[i] += (pc.pos.z - pc.pos_old.z);
    return;
    

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

__global__
void gpuParticlesCollision(
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
    c6  21  22  23  24  25  26  27

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
    const unsigned int column = idx - (row-1)*row/2;

    // Auxiliary variables, for many things
    dfloat aux;

    // Magnitude of gravity force
    const dfloat grav = sqrt(GX * GX + GY * GY + GZ * GZ);

    // Particle from column
    ParticleCenter* pc_i = &particleCenters[column];

    // Collision against walls
    if(row == NUM_PARTICLES){

        if(!pc_i->movable)
            return;

        // Particle position
        const dfloat3 pos_i = pc_i->pos;
        const dfloat radius_i = pc_i->radius;

        const dfloat min_dist = 2 * radius_i + ZETA;

        // Buoyancy force
        const dfloat b_force = grav * (PARTICLE_DENSITY - FLUID_DENSITY) * pc_i->volume;

        // West
        dfloat pos_mirror = -pos_i.x;
        dfloat dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            atomicAdd(&(pc_i->f.x), (b_force / STIFF_WALL) * aux * aux);
        }

        // East
        pos_mirror = 2 * (NX - 1) - pos_i.x;
        dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            atomicAdd(&(pc_i->f.x), -(b_force / STIFF_WALL) * aux * aux);
        }

        // South
        pos_mirror = - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            atomicAdd(&(pc_i->f.y), (b_force / STIFF_WALL) * aux * aux);
        }

        // North
        pos_mirror = 2 * (NY - 1) - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            atomicAdd(&(pc_i->f.y), -(b_force / STIFF_WALL) * aux * aux);
        }

        // Back
        pos_mirror = -pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            atomicAdd(&(pc_i->f.z), (b_force / STIFF_WALL) * aux * aux);
        }

        // Front
        pos_mirror = 2 * (NZ - 1) - pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist) {
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            atomicAdd(&(pc_i->f.z), -(b_force / STIFF_WALL) * aux * aux);
        }
    }
    // Collision against particles
    else{
        ParticleCenter* pc_j = &particleCenters[row];

        if(!pc_i->movable && !pc_j->movable)
            return;

        // Particle i info (column)
        const dfloat3 pos_i = pc_i->pos;
        const dfloat radius_i = pc_i->radius;

        // Particle j info (row)
        const dfloat3 pos_j = pc_j->pos;
        const dfloat radius_j = pc_j->radius;

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

        // Force on particle
        dfloat f = 0;
        dfloat3 f_dirs = dfloat3();

        // Buoyancy force
        const dfloat b_force = grav * (PARTICLE_DENSITY - FLUID_DENSITY) * pc_i->volume;

        // Hard collision (one particle inside another)
        if(mag_dist < radius_i+radius_j){
            aux = (mag_dist - radius_i - radius_j - ZETA) / ZETA;
            // Force to act on particles
            f = ((b_force / STIFF_SOFT) * aux * aux 
                + (b_force / STIFF_HARD) * 
                ((radius_i + radius_j - mag_dist) / ZETA)) / mag_dist;

            // Force in each direction
            f_dirs = dfloat3(
                f * diff_pos.x,
                f * diff_pos.y,
                f * diff_pos.z
            );
        }
        // Soft collision (one particle close to another)
        else if (mag_dist < radius_i+radius_j+ZETA){
            aux = (mag_dist - radius_i - radius_j - ZETA) / ZETA;
            // Force to act on particles
            f = (b_force / STIFF_SOFT) * aux*aux / mag_dist;

            // Force in each direction
            f_dirs = dfloat3(
                f * diff_pos.x,
                f * diff_pos.y,
                f * diff_pos.z
            );
        }
        // Add force on particles
        if(f != 0){
            // Both particles are movable
            if(pc_i->movable && pc_j->movable){
                // Force positive in particle i (column)
                atomicAdd(&(pc_i->f.x), f_dirs.x);
                atomicAdd(&(pc_i->f.y), f_dirs.y);
                atomicAdd(&(pc_i->f.z), f_dirs.z);
                // Force negative in particle j (row)
                atomicAdd(&(pc_j->f.x), -f_dirs.x);
                atomicAdd(&(pc_j->f.y), -f_dirs.y);
                atomicAdd(&(pc_j->f.z), -f_dirs.z);
            }
            // Only particle i is movable
            else if(pc_i->movable && !pc_j->movable){
                // Force positive in particle i (column)
                atomicAdd(&(pc_i->f.x), 2*f_dirs.x);
                atomicAdd(&(pc_i->f.y), 2*f_dirs.y);
                atomicAdd(&(pc_i->f.z), 2*f_dirs.z);
            }
            // Only particle j is movable
            else{
                // Force positive in particle i (column)
                atomicAdd(&(pc_j->f.x), -2*f_dirs.x);
                atomicAdd(&(pc_j->f.y), -2*f_dirs.y);
                atomicAdd(&(pc_j->f.z), -2*f_dirs.z);
            }
        }
    }
}

#endif // !IBM