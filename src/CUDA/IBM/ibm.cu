#include "ibm.h"

#ifdef IBM

__host__ void immersedBoundaryMethod(
    ParticlesSoA particles,
    Macroscopics * __restrict__ macr,
    Populations *const __restrict__ pop,
    dim3 gridLBM,
    dim3 threadsLBM,
    unsigned int gridIBM,
    unsigned int threadsIBM,
    cudaStream_t *__restrict__ stream)
{
    // TODO: Update it to multi GPU
    // Update macroscopics post streaming and reset forces
    gpuUpdateMacrResetForces<<<gridLBM, threadsLBM, 0, stream[0]>>>(pop[0], macr[0]);
    gpuResetNodesForces<<<gridIBM, threadsIBM, 0, stream[0]>>>(particles.nodesSoA);

    checkCudaErrors(cudaStreamSynchronize(stream[0]));

    // Test if this can be done before iterations
    // Calculate collision force between particles
    particlesCollision(particles.pCenterArray);

    for (int i = 0; i < IBM_MAX_ITERATION; i++)
    {
        gpuForceInterpolationSpread<<<gridIBM, threadsIBM, 0, stream[0]>>>(
            particles.nodesSoA, particles.pCenterArray, macr[0]);
        checkCudaErrors(cudaStreamSynchronize(stream[0]));

        // Update LBM velcoties and density
        // TODO: update it in kernel with interpolation/spread
        // TODO: update only velocities, since density is constant
        gpuUpdateMacr<<<gridLBM, threadsLBM, 0, stream[0]>>>(pop[0], macr[0]);
        checkCudaErrors(cudaStreamSynchronize(stream[0]));

        // Update particle velocity using body center force and constant forces
        updateParticleCenterVelocityAndRotation(particles.pCenterArray);
    }

    // Update particle center position and its old values
    particleMovement(particles.pCenterArray);
    // Update particle nodes positions
    gpuParticleNodeMovement<<<gridIBM, threadsIBM, 0, stream[0]>>>(
        particles.nodesSoA, particles.pCenterArray);

    checkCudaErrors(cudaStreamSynchronize(stream[0]));
    ParticleCenter* pc = &particles.pCenterArray[0];
    printf("position %f %f %f\n", pc->pos.x, pc->pos.y, pc->pos.z);
}

__global__ void gpuForceInterpolationSpread(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES],
    Macroscopics const macr)
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
    const unsigned int xMin = (((int)xIBM - P_DIST) < 0) ? 0 : xIBM - P_DIST;
    const unsigned int yMin = (((int)yIBM - P_DIST) < 0) ? 0 : yIBM - P_DIST;
    const unsigned int zMin = (((int)zIBM - P_DIST) < 0) ? 0 : zIBM - P_DIST;

    // Maximum number of xyz for LBM interpolation, excluding last
    // (e.g. NX goes just until NX-1)
    const unsigned int xMax = (((int)xIBM + P_DIST + 1) > NX) ? NX : xIBM + P_DIST + 1;
    const unsigned int yMax = (((int)yIBM + P_DIST + 1) > NY) ? NY : yIBM + P_DIST + 1;
    const unsigned int zMax = (((int)zIBM + P_DIST + 1) > NZ) ? NZ : zIBM + P_DIST + 1;

    dfloat rhoVar = 0;
    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

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
                if(aux == 0)
                    continue;

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

    const dfloat deltaFx = aux * (uxVar - ux_calc);
    const dfloat deltaFy = aux * (uyVar - uy_calc);
    const dfloat deltaFz = aux * (uzVar - uz_calc);

    // Calculate IBM forces
    const dfloat fxIBM = particlesNodes.f.x[i] + deltaFx;
    const dfloat fyIBM = particlesNodes.f.y[i] + deltaFy;
    const dfloat fzIBM = particlesNodes.f.z[i] + deltaFz;

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

                // TODO: update rho and velocities of LBM here, but with 
                // different array to not have concurrent problems with loading 
                // the velocities
                atomicAdd(&(macr.fx[idx]), -deltaFx * aux);
                atomicAdd(&(macr.fy[idx]), -deltaFy * aux);
                atomicAdd(&(macr.fz[idx]), -deltaFz * aux);
            }
        }
    }

    // Update node velocity
    particlesNodes.vel.x[i] = ux_calc;
    particlesNodes.vel.y[i] = uy_calc;
    particlesNodes.vel.z[i] = uz_calc;

    // Update node force
    particlesNodes.f.x[i] = fxIBM;
    particlesNodes.f.y[i] = fyIBM;
    particlesNodes.f.z[i] = fzIBM;

    // Update node delta force
    particlesNodes.deltaF.x[i] = deltaFx;
    particlesNodes.deltaF.y[i] = deltaFy;
    particlesNodes.deltaF.z[i] = deltaFz;

    // Particle node delta momentum
    dfloat3 deltaMomentum = dfloat3();

    deltaMomentum.x = (yIBM - y_pc) * deltaFz - (zIBM - z_pc) * deltaFy;
    deltaMomentum.y = (zIBM - z_pc) * deltaFx - (xIBM - x_pc) * deltaFz;
    deltaMomentum.z = (xIBM - x_pc) * deltaFy - (yIBM - y_pc) * deltaFx;

    // Add node force to particle center
    idx = particlesNodes.particleCenterIdx[i];

    // TODO: check if shared memory is more efficient
    atomicAdd(&(particleCenters[idx].f.x), deltaFx);
    atomicAdd(&(particleCenters[idx].f.y), deltaFy);
    atomicAdd(&(particleCenters[idx].f.z), deltaFz);

    atomicAdd(&(particleCenters[idx].M.x), deltaMomentum.x);
    atomicAdd(&(particleCenters[idx].M.y), deltaMomentum.y);
    atomicAdd(&(particleCenters[idx].M.z), deltaMomentum.z);
}

__global__ void gpuUpdateMacrResetForces(Populations pop, Macroscopics macr)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t idx = idxScalar(x, y, z);

    // Reset forces
    macr.fx[idx] = FX;
    macr.fy[idx] = FY;
    macr.fz[idx] = FZ;

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
    const dfloat invRho = 1 / rhoVar;
    const dfloat uxVar = ((fNode[1] + fNode[7] + fNode[9] + fNode[13] + fNode[15]) 
    - (fNode[2] + fNode[8] + fNode[10] + fNode[14] + fNode[16]) + 0.5 * FX) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17])
     - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18]) + 0.5 * FY) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18])
     - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17]) + 0.5 * FZ) * invRho;
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
        + fNode[22] + fNode[24] + fNode[25]) + 0.5 * FX) * invRho;
    const dfloat uyVar = ((fNode[3] + fNode[7] + fNode[11] + fNode[14] + fNode[17]
        + fNode[19] + fNode[21] + fNode[24] + fNode[25]) 
        - (fNode[4] + fNode[8] + fNode[12] + fNode[13] + fNode[18] + fNode[20] 
        + fNode[22] + fNode[23] + fNode[26]) + 0.5 * FY) * invRho;
    const dfloat uzVar = ((fNode[5] + fNode[9] + fNode[11] + fNode[16] + fNode[18]
        + fNode[19] + fNode[22] + fNode[23] + fNode[25]) 
        - (fNode[6] + fNode[10] + fNode[12] + fNode[15] + fNode[17] + fNode[20] 
        + fNode[21] + fNode[24] + fNode[26]) + 0.5 * FZ) * invRho;
#endif // !D3Q27

    macr.rho[idx] = rhoVar;
    macr.ux[idx] = uxVar;
    macr.uy[idx] = uyVar;
    macr.uz[idx] = uzVar;
}

__global__ void gpuResetNodesForces(ParticleNodeSoA particlesNodes)
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

__host__ void updateParticleCenterVelocityAndRotation(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        ParticleCenter *pc = &(particleCenters[p]);
        
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
        pc->w.x = wNew.x;
        pc->w.y = wNew.y;
        pc->w.z = wNew.z;
    }
}


__host__
void particleMovement(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    for (int p = 0; p < NUM_PARTICLES; p++){
        ParticleCenter *pc = &(particleCenters[p]);

        pc->pos_old.x = pc->pos.x;
        pc->pos_old.y = pc->pos.y;
        pc->pos_old.z = pc->pos.z;

        pc->pos.x += 0.5 * (pc->vel.x + pc->vel_old.x);
        pc->pos.y += 0.5 * (pc->vel.y + pc->vel_old.y);
        pc->pos.z += 0.5 * (pc->vel.z + pc->vel_old.z);

        pc->vel_old.x = pc->vel.x;
        pc->vel_old.y = pc->vel.y;
        pc->vel_old.z = pc->vel.z;

        pc->w_avg.x = 0.5 * (pc->w.x + pc->w_old.x);
        pc->w_avg.y = 0.5 * (pc->w.y + pc->w_old.y);
        pc->w_avg.z = 0.5 * (pc->w.z + pc->w_old.z);

        pc->w_old.x = pc->w.x;
        pc->w_old.y = pc->w.y;
        pc->w_old.z = pc->w.z;

        pc->f_old.x = pc->f.x;
        pc->f_old.y = pc->f.y;
        pc->f_old.z = pc->f.z;
    }
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


__host__
void particlesCollision(
    ParticleCenter particleCenters[NUM_PARTICLES]
){
    // Auxiliary variable for many things
    dfloat aux;

    for(int i = 0; i < NUM_PARTICLES; i++){
        const dfloat3 pos_i = particleCenters[i].pos;
        const dfloat radius_i = particleCenters[i].radius;

        const dfloat grav = sqrt(GX * GX + GY * GY + GZ * GZ);
        // Buoyancy force
        const dfloat b_force = grav * (PARTICLE_DENSITY - FLUID_DENSITY) * particleCenters[i].volume;
        
        // Collision against walls
        const dfloat min_dist = 2 * radius_i + ZETA;

        // West
        dfloat pos_mirror = -pos_i.x;
        dfloat dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            particleCenters[i].f.x += (b_force / STIFF_WALL) * aux * aux;
        }

        // East
        pos_mirror = 2 * (NX - 1) - pos_i.x;
        dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            particleCenters[i].f.x -= (b_force / STIFF_WALL) * aux * aux;
        }

        // South
        pos_mirror = - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            particleCenters[i].f.y += (b_force / STIFF_WALL) * aux * aux;
        }

        // North
        pos_mirror = 2 * (NY - 1) - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            particleCenters[i].f.y -= (b_force / STIFF_WALL) * aux * aux;
        }

        // Back
        pos_mirror = -pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist){
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            particleCenters[i].f.y += (b_force / STIFF_WALL) * aux * aux;
        }

        // Front
        pos_mirror = 2 * (NZ - 1) - pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist) {
            aux = (dist_abs - 2 * radius_i - ZETA) / ZETA;
            particleCenters[i].f.y -= (b_force / STIFF_WALL) * aux * aux;
        }


        for(int j = i+1; j < NUM_PARTICLES; j++){
            const dfloat3 pos_j = particleCenters[j].pos;
            const dfloat radius_j = particleCenters[j].radius;

            dfloat3 diff_pos;
            diff_pos.x = pos_i.x - pos_j.x;
            diff_pos.y = pos_i.y - pos_j.y;
            diff_pos.z = pos_i.z - pos_j.z;

            const dfloat mag_dist = sqrt(
                diff_pos.x*diff_pos.x
                + diff_pos.y*diff_pos.y
                + diff_pos.z*diff_pos.z);

            // Hard collision (one particle inside another)
            if(mag_dist < radius_i+radius_j){
                aux = (mag_dist - radius_i - radius_j - ZETA) / ZETA;
                // Force to act on particles
                const dfloat f = ((b_force / STIFF_SOFT) * aux * aux 
                    + (b_force / STIFF_HARD) * 
                    ((radius_i + radius_j - mag_dist) / ZETA)) / mag_dist;

                // Force in each direction
                const dfloat3 f_dirs = dfloat3(
                    f * diff_pos.x,
                    f * diff_pos.y,
                    f * diff_pos.z
                );

                // Force positive in particle i
                particleCenters[i].f.x += f_dirs.x;
                particleCenters[i].f.y += f_dirs.y;
                particleCenters[i].f.z += f_dirs.z;
                // Force negative in particle j
                particleCenters[j].f.x -= f_dirs.x;
                particleCenters[j].f.y -= f_dirs.y;
                particleCenters[j].f.z -= f_dirs.z;
            }
            // Soft collision (one particle close to another)
            else if (mag_dist < radius_i+radius_j+ZETA){
                aux = (mag_dist - radius_i - radius_j - ZETA) / ZETA;
                // Force to act on particles
                const dfloat f = (b_force / STIFF_SOFT) * aux*aux / mag_dist;

                // Force in each direction
                const dfloat3 f_dirs = dfloat3(
                    f * diff_pos.x,
                    f * diff_pos.y,
                    f * diff_pos.z
                );

                // Force positive in particle i
                particleCenters[i].f.x += f_dirs.x;
                particleCenters[i].f.y += f_dirs.y;
                particleCenters[i].f.z += f_dirs.z;
                // Force negative in particle j
                particleCenters[j].f.x -= f_dirs.x;
                particleCenters[j].f.y -= f_dirs.y;
                particleCenters[j].f.z -= f_dirs.z;
            }
        }
    }
}

#endif // !IBM