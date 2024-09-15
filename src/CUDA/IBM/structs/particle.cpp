#include "particle.h"

#ifdef IBM

void ParticlesSoA::updateParticlesAsSoA(Particle* particles){
    unsigned int totalIbmNodes = 0;

    // Determine the total number of nodes
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        totalIbmNodes += particles[p].numNodes;
    }

    printf("Total number of nodes: %u\n", totalIbmNodes);
    printf("Total memory used for Particles: %lu Mb\n",
           (unsigned long)((totalIbmNodes * sizeof(particleNode) * N_GPUS + NUM_PARTICLES * sizeof(particleCenter)) / BYTES_PER_MB));
    fflush(stdout);

    printf("Allocating particles in GPU... \t"); fflush(stdout);
    // Allocate nodes in each GPU
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        this->nodesSoA[i].allocateMemory(totalIbmNodes);
    }

    // Allocate particle center array
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    checkCudaErrors(
        cudaMallocManaged((void**)&(this->pCenterArray), sizeof(ParticleCenter) * NUM_PARTICLES));
    // Allocate array of last positions for Particles
    this->pCenterLastPos = (dfloat3*)malloc(sizeof(dfloat3)*NUM_PARTICLES);
    this->pCenterLastWPos = (dfloat3*)malloc(sizeof(dfloat3) * NUM_PARTICLES);
    printf("Particles allocated in GPU!\n"); fflush(stdout);

    printf("Optimizig memory layout of particles for GPU... \t"); fflush(stdout);

    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
        this->pCenterArray[p] = particles[p].pCenter;
        this->pCenterLastPos[p] = particles[p].pCenter.pos;
        this->pCenterLastWPos[p] = particles[p].pCenter.w_pos;
        for(int i = 0; i < N_GPUS; i++){
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            this->nodesSoA[i].copyNodesFromParticle(particles[p], p, i);
        }
    }
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));

    printf("Optimized memory layout for GPU!\n"); fflush(stdout);

    printf("Particles positions:\n");
    for(int p = 0; p < NUM_PARTICLES; p++)
    {
        printf("p[%u] -- x: %f - y: %f - z: %f \n",
               p, this->pCenterArray[p].pos.x, 
               this->pCenterArray[p].pos.y, 
               this->pCenterArray[p].pos.z);
    }

    fflush(stdout);
}


void ParticlesSoA::updateNodesGPUs(){
    // No need to update for 1 GPU
    if(N_GPUS == 1)
        return;

    for(int i = 0; i < NUM_PARTICLES; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
        if(!this->pCenterArray[i].movable)
            continue;

        dfloat3 pos_p = this->pCenterArray[i].pos;
        dfloat3 last_pos = this->pCenterLastPos[i];
        dfloat3 w_pos = this->pCenterArray[i].w_pos;
        dfloat3 last_w_pos = this->pCenterLastWPos[i];
        dfloat3 diff_w_pos = dfloat3(w_pos.x-last_w_pos.x, w_pos.y-last_w_pos.y, w_pos.z-last_w_pos.z);
        dfloat radius = this->pCenterArray[i].radius;

        dfloat min_pos = myMin(pos_p.z,last_pos.z) - (radius - BREUGEM_PARAMETER);
        dfloat max_pos = myMax(pos_p.z,last_pos.z) + (radius - BREUGEM_PARAMETER);
        int min_gpu = (int)(min_pos+NZ_TOTAL)/(NZ)-N_GPUS;
        int max_gpu = (int)(max_pos/NZ);

        // Translation
        dfloat diff_z = (pos_p.z-last_pos.z);
        if (diff_z < 0)
            diff_z = -diff_z;
        // Maximum rotation
        diff_z += (radius-BREUGEM_PARAMETER)*sqrt(diff_w_pos.x*diff_w_pos.x+diff_w_pos.y*diff_w_pos.y);

        // Particle has not moved enoush and nodes that needs to be 
        // updated/synchronized are already considering that
        if(diff_z < IBM_PARTICLE_UPDATE_DIST)
            continue;

        // Update particle's last position
        this->pCenterLastPos[i] = this->pCenterArray[i].pos;
        this->pCenterLastWPos[i] = this->pCenterArray[i].w_pos;
        if(min_gpu == max_gpu)
            continue;

        for(int n = min_gpu; n <= max_gpu; n++){
            // Set current device
            int real_gpu = (n+N_GPUS)%N_GPUS;
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[real_gpu]));
            int left_shift = 0;
            for(int p = 0; p < this->nodesSoA[real_gpu].numNodes; p++){
                // Shift left nodes, if a node was already removed
                if(left_shift != 0){
                    this->nodesSoA[real_gpu].leftShiftNodesSoA(p, left_shift);
                }

                // Node is from another particle
                if(this->nodesSoA[real_gpu].particleCenterIdx[p] != i)
                    continue;

                // Check in what GPU node is
                dfloat pos_z = this->nodesSoA[real_gpu].pos.z[p];
                int node_gpu = (int) (pos_z/NZ);
                // If node is still in same GPU, continues
                if(node_gpu == real_gpu)
                    continue;
                //printf("b");
                // to not raise any error when setting up device
                #ifdef IBM_BC_Z_PERIODIC
                    node_gpu = (node_gpu+N_GPUS)%N_GPUS;
                #else
                    if(node_gpu < 0)
                        node_gpu = 0;
                    else if(node_gpu >= N_GPUS)
                        node_gpu = N_GPUS-1;
                #endif

                // Nodes will have to be shifted
                left_shift += 1;

                // Get values to move
                ParticleNodeSoA nSoA = this->nodesSoA[real_gpu];
                dfloat copy_S = nSoA.S[p];
                unsigned int copy_pIdx = nSoA.particleCenterIdx[p];
                dfloat3 copy_pos = nSoA.pos.getValuesFromIdx(p);
                dfloat3 copy_vel = nSoA.vel.getValuesFromIdx(p);
                dfloat3 copy_vel_old = nSoA.vel_old.getValuesFromIdx(p);
                dfloat3 copy_f = nSoA.f.getValuesFromIdx(p);
                dfloat3 copy_deltaF = nSoA.deltaF.getValuesFromIdx(p);

                // Set device to move to (unnecessary)
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[node_gpu]));
                nSoA = this->nodesSoA[node_gpu];
                // Copy values to last position in nodesSoA
                size_t idxMove = nSoA.numNodes;
                nSoA.S[idxMove] = copy_S;
                nSoA.particleCenterIdx[idxMove] = copy_pIdx;
                nSoA.pos.copyValuesFromFloat3(copy_pos, idxMove);
                nSoA.vel.copyValuesFromFloat3(copy_vel, idxMove);
                nSoA.vel_old.copyValuesFromFloat3(copy_vel_old, idxMove);
                nSoA.f.copyValuesFromFloat3(copy_f, idxMove);
                nSoA.deltaF.copyValuesFromFloat3(copy_deltaF, idxMove);
                // Added one node to it
                this->nodesSoA[node_gpu].numNodes += 1;
                // Set back particle device (unnecessary)
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[real_gpu]));
                // printf("idx %d  gpu curr %d  ", p, n);
                // for(int nnn = 0; nnn < N_GPUS; nnn++){
                //     printf("Nodes GPU %d: %d\t", nnn, this->nodesSoA[nnn].numNodes);
                // }
                // printf("\n");
            }
            // Remove nodes that were added
            this->nodesSoA[real_gpu].numNodes -= left_shift;
        }
    }
    // int sum = 0;
    // for(int nnn = 0; nnn < N_GPUS; nnn++){
    //     sum += this->nodesSoA[nnn].numNodes;
    // }
    // printf("sum nodes %d\n\n", sum);
}


void ParticlesSoA::freeNodesAndCenters(){
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        this->nodesSoA[i].freeMemory();
    }
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    cudaFree(this->pCenterArray);
    free(this->pCenterLastPos);
    free(this->pCenterLastWPos);
    this->pCenterArray = nullptr;
}

void Particle::makeSphereIco(dfloat diameter, dfloat3 center, bool move,
    dfloat density, dfloat3 vel, dfloat3 w)
{

    constexpr dfloat golden = 1.618033988749895;     // golden number - CONSTANT
    constexpr dfloat nref = 5;    // IBM mesh refinment
    constexpr unsigned int Faces = 20 * (1 + 2 * nref + (nref*nref));
    constexpr unsigned int Edges = 30 * (nref + 1) + 10 * 3 * (nref + nref * nref);
    constexpr unsigned int Vert = 2 + Edges - Faces;

    constexpr dfloat A_AXIS = 0.5*PARTICLE_DIAMETER;
    constexpr dfloat B_AXIS = 0.5*PARTICLE_DIAMETER;
    constexpr dfloat C_AXIS = 0.5*PARTICLE_DIAMETER;

    constexpr unsigned int IT_REF = 0;
    
    // Define the properties of the particle
    unsigned int i, l, r;
    dfloat radius = diameter/2;
    dfloat scale;
    dfloat rc = radius - BREUGEM_PARAMETER;
    dfloat dA = 4*M_PI*(((rc + 0.5)*(rc + 0.5)*(rc + 0.5)) - ((rc - 0.5)*(rc - 0.5)*(rc - 0.5)))/(3*Vert);

    this->pCenter.radius = radius;

    this->pCenter.volume = radius*radius*radius*4*M_PI/3;
    // Particle area
    this->pCenter.S = 4.0 * M_PI * radius * radius;
    // Particle density
    this->pCenter.density = density;

    // Particle center position
    this->pCenter.pos = center;
    this->pCenter.pos_old = center;

    // Particle velocity
    this->pCenter.vel = vel;
    this->pCenter.vel_old = vel;

    // Particle rotation
    this->pCenter.w = w;
    this->pCenter.w_avg = w;
    this->pCenter.w_old = w;

    this->pCenter.q_pos.w = 0;
    this->pCenter.q_pos.x = 1;
    this->pCenter.q_pos.y = 0;
    this->pCenter.q_pos.z = 0;

    this->pCenter.q_pos_old.w = 0.0;
    this->pCenter.q_pos_old.x = 1.0;
    this->pCenter.q_pos_old.y = 0.0;
    this->pCenter.q_pos_old.z = 0.0;

    // Innertia momentum
    this->pCenter.I.xx = 1.0 * this->pCenter.volume * this->pCenter.density * ((B_AXIS * B_AXIS) + (C_AXIS * C_AXIS)) / 5.0;
    this->pCenter.I.yy = 1.0 * this->pCenter.volume * this->pCenter.density * ((A_AXIS * A_AXIS) + (C_AXIS * C_AXIS)) / 5.0;
    this->pCenter.I.zz = 1.0 * this->pCenter.volume * this->pCenter.density * ((A_AXIS * A_AXIS) + (B_AXIS * B_AXIS)) / 5.0;

    this->pCenter.I.xy = 0.0;
    this->pCenter.I.xz = 0.0;
    this->pCenter.I.yz = 0.0;
    
    this->pCenter.f.x = 0.0;
    this->pCenter.f.y = 0.0;
    this->pCenter.f.z = 0.0;

    this->pCenter.f_old.x = 0.0;
    this->pCenter.f_old.y = 0.0;
    this->pCenter.f_old.z = 0.0;

    this->pCenter.M.x = 0.0;
    this->pCenter.M.y = 0.0;
    this->pCenter.M.z = 0.0;

    this->pCenter.M_old.x = 0.0;
    this->pCenter.M_old.y = 0.0;
    this->pCenter.M_old.z = 0.0;


    this->pCenter.movable = move;

    this->nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * Vert);
    this->numNodes = Vert;

    i = 1;
    scale = radius/sqrt(1 + (golden*golden));

    ParticleNode* first_node = &(this->nodes[0]);

    //  Icosahedron vertices:
    // Node 0:
    first_node->pos.x = -1.0*scale;
    first_node->pos.y = golden*scale;
    first_node->pos.z = 0.0*scale;
    first_node->S = dA;
    // Define node velocity
    first_node->vel.x = vel.x + w.y * first_node->pos.z - w.z * first_node->pos.y;
    first_node->vel.y = vel.y + w.z * first_node->pos.x - w.x * first_node->pos.z;
    first_node->vel.z = vel.z + w.x * first_node->pos.y - w.y * first_node->pos.x;

    first_node->vel_old.x = vel.x + w.y * first_node->pos.z - w.z * first_node->pos.y;
    first_node->vel_old.y = vel.y + w.z * first_node->pos.x - w.x * first_node->pos.z;
    first_node->vel_old.z = vel.z + w.x * first_node->pos.y - w.y * first_node->pos.x;
    //  Node 1:
    this->nodes[i].pos.x = 1.0*scale;
	this->nodes[i].pos.y = golden*scale;
    this->nodes[i].pos.z = 0.0*scale;
    i++;
    //  Node 2:
    this->nodes[i].pos.x = -1.0*scale;
	this->nodes[i].pos.y = -golden*scale;
    this->nodes[i].pos.z = 0.0*scale;
    i++;
    //  Node 3:
    this->nodes[i].pos.x = 1.0*scale;
	this->nodes[i].pos.y = -golden*scale;
    this->nodes[i].pos.z = 0.0*scale;
    i++;
    //  Node 4:
    this->nodes[i].pos.x = 0.0*scale;
	this->nodes[i].pos.y = -1.0*scale;
    this->nodes[i].pos.z = golden*scale;
    i++;
    //  Node 5:
    this->nodes[i].pos.x = 0.0*scale;
	this->nodes[i].pos.y = 1.0*scale;
    this->nodes[i].pos.z = golden*scale;
    i++;
	//  Node 6:
    this->nodes[i].pos.x = 0.0*scale;
	this->nodes[i].pos.y = -1.0*scale;
    this->nodes[i].pos.z = -golden*scale;
    i++;
	//  Node 7:
    this->nodes[i].pos.x = 0.0*scale;
	this->nodes[i].pos.y = 1.0*scale;
    this->nodes[i].pos.z = -golden*scale;
    i++;
    //  Node 8:
    this->nodes[i].pos.x = golden*scale;
	this->nodes[i].pos.y = 0.0*scale;
    this->nodes[i].pos.z = -1.0*scale;
    i++;
    //  Node 9:
    this->nodes[i].pos.x = golden*scale;
	this->nodes[i].pos.y = 0.0*scale;
    this->nodes[i].pos.z = 1.0*scale;
    i++;
    //  Node 10:
    this->nodes[i].pos.x = -golden*scale;
	this->nodes[i].pos.y = 0.0*scale;
    this->nodes[i].pos.z = -1.0*scale;
    i++;
    //  Node 11:
    this->nodes[i].pos.x = -golden*scale;
	this->nodes[i].pos.y = 0.0*scale;
    this->nodes[i].pos.z = 1.0*scale;
    i++;
    //  Edges Nodes:
    for (r = 1; r < nref + 1; r++) // Edge 0 - 5
	{
		this->nodes[i].pos.x = -1.0 + (r*(0.0 - (-1.0)) / (1 + nref));
		this->nodes[i].pos.y = golden + (r*(1.0 - golden) / (1 + nref));
        this->nodes[i].pos.z = 0.0 + (r*(golden - 0.0) / (1 + nref));

        scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

        this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
        this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
        this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 1].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 2].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 2].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 2].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 3].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 3].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 3].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 4].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 4].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 4].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 5].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 5].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 5].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 6].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 6].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 6].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 7].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 7].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 7].pos.z;
		i++;
    }
    for (l = 1; l < nref + 1; l++) // Edge 0 - 11
	{
        this->nodes[i].pos.x = -1.0 + (l*(-golden - (-1.0)) / (1 + nref));
		this->nodes[i].pos.y = golden + (l*(0.0 - golden) / (1 + nref));
        this->nodes[i].pos.z = 0.0 + (l*(1.0 - 0.0) / (1 + nref));

        scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

        this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
        this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
        this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
		i++;
		this->nodes[i].pos.x = - this->nodes[i - 1].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 2].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 2].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 2].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 3].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 3].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 3].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 4].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 4].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 4].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 5].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 5].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 5].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 6].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 6].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 6].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 7].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 7].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 7].pos.z;
		i++;
    }
    for (r = 1; r < nref + 1; r++) // Edge 11 - 5
	{
        this->nodes[i].pos.x = -golden + (r*(0.0 - (-golden)) / (1 + nref));
		this->nodes[i].pos.y = 0.0 + (r*(1.0 - 0.0) / (1 + nref));
        this->nodes[i].pos.z = 1.0 + (r*(golden - 1.0) / (1 + nref));

        scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

        this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
        this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
        this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
		i++;
		this->nodes[i].pos.x = - this->nodes[i - 1].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 2].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 2].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 2].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 3].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 3].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 3].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 4].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 4].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 4].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 5].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 5].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 5].pos.z;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 6].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 6].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 6].pos.z;
        i++;
        this->nodes[i].pos.x = - this->nodes[i - 7].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 7].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 7].pos.z;
		i++;
    }
    for (r = 1; r < nref + 1; r++) // Edge 10 - 11
	{
        this->nodes[i].pos.x = -golden + (r*(-golden - (-golden)) / (1 + nref));
		this->nodes[i].pos.y = 0.0 + (r*(0.0 - 0.0) / (1 + nref));
        this->nodes[i].pos.z = -1.0 + (r*(1.0 - (-1.0)) / (1 + nref));

        scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

        this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
        this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
        this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
		i++;
		this->nodes[i].pos.x = - this->nodes[i - 1].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
        i++;
    }
    for (r = 1; r < nref + 1; r++) // Edge 0 - 1
	{
        this->nodes[i].pos.x = -1.0 + (r*(1.0 - (-1.0)) / (1 + nref));
		this->nodes[i].pos.y = golden + (r*(golden - golden) / (1 + nref));
        this->nodes[i].pos.z = 0.0 + (r*(0.0 - 0.0) / (1 + nref));

        scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

        this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
        this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
        this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 1].pos.x;
	    this->nodes[i].pos.y = - this->nodes[i - 1].pos.y;
        this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
		i++;
    }
    for (r = 1; r < nref + 1; r++) // Edge 4 - 5
	{
        this->nodes[i].pos.x = 0.0 + (r*(0.0 - (0.0)) / (1 + nref));
		this->nodes[i].pos.y = -1.0 + (r*(1.0 - (-1.0)) / (1 + nref));
        this->nodes[i].pos.z = golden + (r*(golden - golden) / (1 + nref));

        scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

        this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
        this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
        this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
        i++;
        this->nodes[i].pos.x =   this->nodes[i - 1].pos.x;
	    this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
        this->nodes[i].pos.z = - this->nodes[i - 1].pos.z;
		i++;
    }
    //  Internal Nodes
    for (r = 1; r < nref; r++) // Triangle 0 - 5 - 11
	{
		for (l = 1; l < nref + 1 - r; l++)
		{
            this->nodes[i].pos.x = -1.0 + (r*(0.0 - (-1.0)) / (1 + nref)) + (l*(-golden - (-1.0)) / (1 + nref));
		    this->nodes[i].pos.y = golden + (r*(1.0 - golden) / (1 + nref)) + (l*(0.0 - golden) / (1 + nref));
            this->nodes[i].pos.z = 0.0 + (r*(golden - 0.0) / (1 + nref)) + (l*(1.0 - 0.0) / (1 + nref));

            scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

            this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
            this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
            this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
			i++;
			this->nodes[i].pos.x = - this->nodes[i - 1].pos.x;
	        this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 2].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 2].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 2].pos.z;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 3].pos.x;
	        this->nodes[i].pos.y =   this->nodes[i - 3].pos.y;
            this->nodes[i].pos.z = - this->nodes[i - 3].pos.z;
            i++;
            this->nodes[i].pos.x = - this->nodes[i - 4].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 4].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 4].pos.z;
            i++;
            this->nodes[i].pos.x = - this->nodes[i - 5].pos.x;
	        this->nodes[i].pos.y =   this->nodes[i - 5].pos.y;
            this->nodes[i].pos.z = - this->nodes[i - 5].pos.z;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 6].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 6].pos.y;
            this->nodes[i].pos.z = - this->nodes[i - 6].pos.z;
            i++;
            this->nodes[i].pos.x = - this->nodes[i - 7].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 7].pos.y;
            this->nodes[i].pos.z = - this->nodes[i - 7].pos.z;
            i++;
		}
    }
    for (r = 1; r < nref; r++) // Triangle 0 - 11 - 10
	{
		for (l = 1; l < nref + 1 - r; l++)
		{
            this->nodes[i].pos.x = -1.0 + (r*(-golden - (-1.0)) / (1 + nref)) + (l*(-golden - (-1.0)) / (1 + nref));
		    this->nodes[i].pos.y = golden + (r*(0.0 - golden) / (1 + nref)) + (l*(0.0 - golden) / (1 + nref));
            this->nodes[i].pos.z = 0.0 + (r*(1.0 - 0.0) / (1 + nref)) + (l*(-1.0 - (0.0)) / (1 + nref));

            scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

            this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
            this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
            this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
            i++;
            this->nodes[i].pos.x = - this->nodes[i - 1].pos.x;
	        this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 2].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 2].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 2].pos.z;
            i++;
            this->nodes[i].pos.x = - this->nodes[i - 3].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 3].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 3].pos.z;
			i++;
		}
	}
	for (r = 1; r < nref; r++) // Triangle 0 - 5 - 1
	{
		for (l = 1; l < nref + 1 - r; l++)
		{
            this->nodes[i].pos.x = -1.0 + (r*(1.0 - (-1.0)) / (1 + nref)) + (l*(0.0 - (-1.0)) / (1 + nref));
		    this->nodes[i].pos.y = golden + (r*(golden - golden) / (1 + nref)) + (l*(1.0 - golden) / (1 + nref));
            this->nodes[i].pos.z = 0.0 + (r*(0.0 - 0.0) / (1 + nref)) + (l*(golden - (0.0)) / (1 + nref));

            scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

            this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
            this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
            this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 1].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 1].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 2].pos.x;
	        this->nodes[i].pos.y =   this->nodes[i - 2].pos.y;
            this->nodes[i].pos.z = - this->nodes[i - 2].pos.z;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 3].pos.x;
	        this->nodes[i].pos.y = - this->nodes[i - 3].pos.y;
            this->nodes[i].pos.z = - this->nodes[i - 3].pos.z;
			i++;
		}
	}
	for (r = 1; r < nref; r++) // Triangle 5 - 4 - 11
	{
		for (l = 1; l < nref + 1 - r; l++)
		{
            this->nodes[i].pos.x = 0.0 + (r*(0.0 - (0.0)) / (1 + nref)) + (l*(-golden - (0.0)) / (1 + nref));
		    this->nodes[i].pos.y = 1.0 + (r*(-1.0 - 1.0) / (1 + nref)) + (l*(0.0 - 1.0) / (1 + nref));
            this->nodes[i].pos.z = golden + (r*(golden - golden) / (1 + nref)) + (l*(1.0 - golden) / (1 + nref));

            scale = radius / sqrt((this->nodes[i].pos.x * this->nodes[i].pos.x) + (this->nodes[i].pos.y * this->nodes[i].pos.y) + (this->nodes[i].pos.z * this->nodes[i].pos.z));

            this->nodes[i].pos.x =   this->nodes[i].pos.x * scale;
            this->nodes[i].pos.y =   this->nodes[i].pos.y * scale;
            this->nodes[i].pos.z =   this->nodes[i].pos.z * scale;
            i++;
            this->nodes[i].pos.x = - this->nodes[i - 1].pos.x;
	        this->nodes[i].pos.y =   this->nodes[i - 1].pos.y;
            this->nodes[i].pos.z =   this->nodes[i - 1].pos.z;
            i++;
            this->nodes[i].pos.x =   this->nodes[i - 2].pos.x;
	        this->nodes[i].pos.y =   this->nodes[i - 2].pos.y;
            this->nodes[i].pos.z = - this->nodes[i - 2].pos.z;
            i++;
            if (r > nref - 2)
            {
                ParticleNode* last_node = &(this->nodes[this->numNodes-1]);
                last_node->pos.x = - this->nodes[i - 3].pos.x;
                last_node->pos.y =   this->nodes[i - 3].pos.y;
                last_node->pos.z = - this->nodes[i - 3].pos.z;
                last_node->S = dA;
                // define last node velocity
                last_node->vel.x = vel.x + w.y * last_node->pos.z - w.z * last_node->pos.y;
                last_node->vel.y = vel.y + w.z * last_node->pos.x - w.x * last_node->pos.z;
                last_node->vel.z = vel.z + w.x * last_node->pos.y - w.y * last_node->pos.x;

                last_node->vel_old.x = vel.x + w.y * last_node->pos.z  - w.z * last_node->pos.y;
                last_node->vel_old.y = vel.y + w.z * last_node->pos.x  - w.x * last_node->pos.z;
                last_node->vel_old.z = vel.z + w.x * last_node->pos.y  - w.y * last_node->pos.x;
                i++;
            }
            else
            {
                this->nodes[i].pos.x = - this->nodes[i - 3].pos.x;
	            this->nodes[i].pos.y =   this->nodes[i - 3].pos.y;
                this->nodes[i].pos.z = - this->nodes[i - 3].pos.z;
			    i++;
            }
		}
    }
    
    ParticleNode* node_i;
    for (int i = 0; i < this->numNodes; i++) 
    {
        node_i = &(this->nodes[i]);
        node_i->pos.x += center.x;
        node_i->pos.y += center.y;
        node_i->pos.z += center.z;
    }
    // Update old position value
    this->pCenter.pos_old = this->pCenter.pos;

    for(int ii = 0;ii<this->numNodes;ii++){
        ParticleNode* node_j = &(this->nodes[ii]);
        printf("%f %f %f \n",node_j->pos.x,node_j->pos.y,node_j->pos.z );
    }
}


void Particle::makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move,
    dfloat density, dfloat3 vel, dfloat3 w)
{
    // Maximum number of layer of sphere
    //unsigned int maxNumLayers = 5000;
    // Number of layers in sphere
    unsigned int nLayer;
    // Number of nodes per layer in sphere
    unsigned int* nNodesLayer;
    // Angles in polar coordinates and node area
    dfloat *theta, *zeta, *S;

    dfloat phase = 0.0;

    // this->pCenter = ParticleCenter();

    // Define the properties of the particle
    dfloat r = diameter / 2.0;
    dfloat volume = r*r*r*4*M_PI/3;

    this->pCenter.radius = r;
    this->pCenter.volume = r*r*r*4*M_PI/3;
    // Particle area
    this->pCenter.S = 4.0 * M_PI * r * r;
    // Particle density
    this->pCenter.density = density;

    // Particle center position
    this->pCenter.pos = center;
    this->pCenter.pos_old = center;

    // Particle velocity
    this->pCenter.vel = vel;
    this->pCenter.vel_old = vel;

    // Particle rotation
    this->pCenter.w = w;
    this->pCenter.w_avg = w;
    this->pCenter.w_old = w;

    this->pCenter.q_pos.w = 0;
    this->pCenter.q_pos.x = 1;
    this->pCenter.q_pos.y = 0;
    this->pCenter.q_pos.z = 0;

    this->pCenter.q_pos_old.w = 0.0;
    this->pCenter.q_pos_old.x = 1.0;
    this->pCenter.q_pos_old.y = 0.0;
    this->pCenter.q_pos_old.z = 0.0;

    // Innertia momentum
    this->pCenter.I.xx = 2.0 * volume * this->pCenter.density * r * r / 5.0;
    this->pCenter.I.yy = 2.0 * volume * this->pCenter.density * r * r / 5.0;
    this->pCenter.I.zz = 2.0 * volume * this->pCenter.density * r * r / 5.0;

    this->pCenter.I.xy = 0.0;
    this->pCenter.I.xz = 0.0;
    this->pCenter.I.yz = 0.0;

    this->pCenter.f.x = 0.0;
    this->pCenter.f.y = 0.0;
    this->pCenter.f.z = 0.0;

    this->pCenter.f_old.x = 0.0;
    this->pCenter.f_old.y = 0.0;
    this->pCenter.f_old.z = 0.0;

    this->pCenter.M.x = 0.0;
    this->pCenter.M.y = 0.0;
    this->pCenter.M.z = 0.0;

    this->pCenter.M_old.x = 0.0;
    this->pCenter.M_old.y = 0.0;
    this->pCenter.M_old.z = 0.0;


    this->pCenter.movable = move;


    this->pCenter.collision.shape = SPHERE;
    this->pCenter.collision.semiAxis = dfloat3(r,r,r);
    for(int i = 0; i <MAX_ACTIVE_COLLISIONS;i++){
        this->pCenter.collision.collisionPartnerIDs[i] = -1;
        this->pCenter.collision.tangentialDisplacements[i] = dfloat3(0,0,0);
        this->pCenter.collision.lastCollisionStep[i] = -1;
    }

    //breugem correction
    r -= BREUGEM_PARAMETER;

    // Number of layers in the sphere
    nLayer = (unsigned int)(2.0 * sqrt(2) * r / MESH_SCALE + 1.0); 

    nNodesLayer = (unsigned int*)malloc((nLayer+1) * sizeof(unsigned int));
    theta = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));
    zeta = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));
    S = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));

    this->numNodes = 0;
    for (int i = 0; i <= nLayer; i++) {
        // Angle of each layer
        theta[i] = M_PI * ((dfloat)i / (dfloat)nLayer - 0.5); 
        // Determine the number of node per layer
        nNodesLayer[i] = (unsigned int)(1.5 + cos(theta[i]) * nLayer * sqrt(3)); 
        // Total number of nodes on the sphere
        this->numNodes += nNodesLayer[i]; 
        zeta[i] = r * sin(theta[i]); // Height of each layer
    }

    
    for (int i = 0; i < nLayer; i++) {
        // Calculate the distance to the south pole to the mid distance of the layer and previous layer
        S[i] = (zeta[i] + zeta[i + 1]) / 2.0 - zeta[0]; 
    }
    S[nLayer] = 2 * r;
    for (int i = 0; i <= nLayer; i++) {
        // Calculate the area of sphere segment since the south pole
        S[i] = 2 * M_PI * r * S[i]; 
    }
    for (int i = nLayer; i > 0; i--) {
        // Calculate the area of the layer
        S[i] = S[i] - S[i - 1];
    }
    S[0] = S[nLayer];
    

    this->nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * this->numNodes);

    ParticleNode* first_node = &(this->nodes[0]);

    // South node - define all properties
    first_node->pos.x = 0;
    first_node->pos.y = 0;
    first_node->pos.z = r * sin(theta[0]);

    first_node->S = S[0];

    // Define node velocity
    first_node->vel.x = vel.x + w.y * first_node->pos.z - w.z * first_node->pos.y;
    first_node->vel.y = vel.y + w.z * first_node->pos.x - w.x * first_node->pos.z;
    first_node->vel.z = vel.z + w.x * first_node->pos.y - w.y * first_node->pos.x;

    first_node->vel_old.x = vel.x + w.y * first_node->pos.z - w.z * first_node->pos.y;
    first_node->vel_old.y = vel.y + w.z * first_node->pos.x - w.x * first_node->pos.z;
    first_node->vel_old.z = vel.z + w.x * first_node->pos.y - w.y * first_node->pos.x;

    int nodeIndex = 1;
    for (int i = 1; i < nLayer; i++) {
        if (i % 2 == 1) {
            // Calculate the phase of the segmente to avoid a straight point line
            phase = phase + M_PI / nNodesLayer[i];
        }

        for (int j = 0; j < nNodesLayer[i]; j++) {
            // Determine the properties of each node in the mid layers
            this->nodes[nodeIndex].pos.x = r * cos(theta[i]) * cos((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            this->nodes[nodeIndex].pos.y = r * cos(theta[i]) * sin((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            this->nodes[nodeIndex].pos.z = r * sin(theta[i]);

            // The area of sphere segment is divided by the number of node
            // in the layer, so all nodes have the same area in the layer
            this->nodes[nodeIndex].S = S[i] / nNodesLayer[i];

            // Define node velocity
            this->nodes[nodeIndex].vel.x = vel.x + w.y * this->nodes[nodeIndex].pos.z - w.z * this->nodes[nodeIndex].pos.y;
            this->nodes[nodeIndex].vel.y = vel.y + w.z * this->nodes[nodeIndex].pos.x - w.x * this->nodes[nodeIndex].pos.z;
            this->nodes[nodeIndex].vel.z = vel.z + w.x * this->nodes[nodeIndex].pos.y - w.y * this->nodes[nodeIndex].pos.x;

            this->nodes[nodeIndex].vel_old.x = vel.x + w.y * this->nodes[nodeIndex].pos.z - w.z * this->nodes[nodeIndex].pos.y;
            this->nodes[nodeIndex].vel_old.y = vel.y + w.z * this->nodes[nodeIndex].pos.x - w.x * this->nodes[nodeIndex].pos.z;
            this->nodes[nodeIndex].vel_old.z = vel.z + w.x * this->nodes[nodeIndex].pos.y - w.y * this->nodes[nodeIndex].pos.x;

            // Add one node
            nodeIndex++;
        }
    }

    // North pole -define all properties
    ParticleNode* last_node = &(this->nodes[this->numNodes-1]);
    
    last_node->pos.x = 0;
    last_node->pos.y = 0;
    last_node->pos.z = r * sin(theta[nLayer]);
    last_node->S = S[nLayer];
    // define last node velocity
    last_node->vel.x = vel.x + w.y * last_node->pos.z - w.z * last_node->pos.y;
    last_node->vel.y = vel.y + w.z * last_node->pos.x - w.x * last_node->pos.z;
    last_node->vel.z = vel.z + w.x * last_node->pos.y - w.y * last_node->pos.x;

    last_node->vel_old.x = vel.x + w.y * last_node->pos.z  - w.z * last_node->pos.y;
    last_node->vel_old.y = vel.y + w.z * last_node->pos.x  - w.x * last_node->pos.z;
    last_node->vel_old.z = vel.z + w.x * last_node->pos.y  - w.y * last_node->pos.x;

    unsigned int numNodes = this->numNodes;

    // Coulomb node positions distribution
    if (coulomb != 0) {
        dfloat3 dir;
        dfloat mag;
        dfloat scaleF;
        dfloat3* cForce;
        cForce = (dfloat3*)malloc(numNodes * sizeof(dfloat3));


        scaleF = 0.001;

        dfloat fx, fy, fz;

        for (unsigned int c = 0; c < coulomb; c++) {
            for (int i = 0; i < numNodes; i++) {
                cForce[i].x = 0;
                cForce[i].y = 0;
                cForce[i].z = 0;
            }

            for (int i = 0; i < numNodes; i++) {
                ParticleNode* node_i = &(this->nodes[i]);

                for (int j = i+1; j < numNodes; j++) {

                    ParticleNode* node_j = &(this->nodes[j]);

                    dir.x = node_j->pos.x - node_i->pos.x;
                    dir.y = node_j->pos.y - node_i->pos.y;
                    dir.z = node_j->pos.z - node_i->pos.z;

                    mag = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

                    cForce[i].x -= dir.x / mag;
                    cForce[i].y -= dir.y / mag;
                    cForce[i].z -= dir.z / mag;

                    cForce[j].x -= -dir.x / mag;
                    cForce[j].y -= -dir.y / mag;
                    cForce[j].z -= -dir.z / mag;
                }
            }
            for (int i = 0; i < numNodes; i++) {
                // Move particle
                fx = cForce[i].x / scaleF;
                fy = cForce[i].y / scaleF;
                fz = cForce[i].z / scaleF;
                
                ParticleNode* node_i = &(this->nodes[i]);

                node_i->pos.x += fx;
                node_i->pos.y += fy;
                node_i->pos.z += fz;

                // Return to sphere
                mag = sqrt(node_i->pos.x * node_i->pos.x 
                    + node_i->pos.y * node_i->pos.y 
                    + node_i->pos.z * node_i->pos.z);

                node_i->pos.x *= r / mag;
                node_i->pos.y *= r / mag;
                node_i->pos.z *= r / mag;
            }
        }

        // Area fix
        for (int i = 0; i < numNodes; i++) {
            ParticleNode* node_i = &(this->nodes[i]);

            node_i->S = this->pCenter.S / (numNodes);
        }

        // Free coulomb force
        free(cForce);

        dfloat dA =  this->pCenter.S/this->numNodes;
        for(int i = 0; i < numNodes; i++){
            this->nodes[i].S = dA;
        }
    }
    ParticleNode* node_i;
    for (int i = 0; i < numNodes; i++) {
        node_i = &(this->nodes[i]);
        node_i->pos.x += center.x;
        node_i->pos.y += center.y;
        node_i->pos.z += center.z;
    }

    /*for(int ii = 0;ii<nodeIndex;ii++){
        ParticleNode* node_j = &(this->nodes[ii]);
        printf("%f %f %f \n",node_j->pos.x,node_j->pos.y,node_j->pos.z );
    }*/

    // Free allocated variables
    free(nNodesLayer);
    free(theta);
    free(zeta);
    free(S);

    // Update old position value
    this->pCenter.pos_old = this->pCenter.pos;
}


void Particle::makeOpenCylinder(dfloat diameter, dfloat3 baseOneCenter, dfloat3 baseTwoCenter, bool pattern)
{
    // Define the properties of the cylinder
    dfloat r = diameter / 2.0;
    dfloat x = baseTwoCenter.x - baseOneCenter.x;
    dfloat y = baseTwoCenter.y - baseOneCenter.y;
    dfloat z = baseTwoCenter.z - baseOneCenter.z;
    dfloat length = sqrt (x*x +y*y+z*z);
    dfloat volume = r*r*M_PI*length;

    this->pCenter.radius = r;
    // Particle volume
    this->pCenter.volume = volume;
    // Particle area
    this->pCenter.S = 2.0*M_PI*r*length;

    // Particle center position
    this->pCenter.pos.x = (baseOneCenter.x + baseTwoCenter.x)/2;
    this->pCenter.pos.y = (baseOneCenter.y + baseTwoCenter.y)/2;
    this->pCenter.pos.z = (baseOneCenter.z + baseTwoCenter.z)/2;

    this->pCenter.movable = false;


    int nLayer, nNodesLayer;

    dfloat3* centerLayer;
    

    dfloat scale = MESH_SCALE;

    dfloat layerDistance = scale;
    if (pattern)
        layerDistance = scale*scale*sqrt(3)/4;

    //number of layers
    nLayer = (int)(length/layerDistance);
    //number of nodes per layer
    nNodesLayer = (int)(M_PI * 2.0 *r / scale);

    //total number of nodes
    this->numNodes = nLayer * nNodesLayer;

    this->nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * this->numNodes);

    //layer center position step
    dfloat dx = x / nLayer;
    dfloat dy = y / nLayer;
    dfloat dz = z / nLayer;

    centerLayer = (dfloat3*)malloc((nLayer) * sizeof(dfloat3));
    for (int i = 0; i < nLayer ; i++) {
        centerLayer[i].x = baseOneCenter.x + dx * ((dfloat)i+0.5*scale);
        centerLayer[i].y = baseOneCenter.y + dy * ((dfloat)i+0.5*scale);
        centerLayer[i].z = baseOneCenter.z + dz * ((dfloat)i+0.5*scale);
    }

    //adimensionalise direction vector
    x = x / length;
    y = y / length;
    z = z / length;

    dfloat3 a1;//randon vector for perpicular direction
    a1.x = 0.9; 
    a1.y = 0.8; 
    a1.z = 0.7; 
    //TODO: it will work as long the created cyclinder does not have the same direction
    //      make it can work in any direction
    dfloat a1l = sqrt(a1.x * a1.x + a1.y * a1.y + a1.z * a1.z);
    a1.x = a1.x / a1l; 
    a1.y = a1.y / a1l; 
    a1.z = a1.z / a1l;

    //product of x with a1, v1 is the first axis in the layer plane
    dfloat3 v1;
    v1.x = (y * a1.z - z * a1.y);
    v1.y = (z * a1.x - x * a1.z);
    v1.z = (x * a1.y - y * a1.x);

    //product of x with v1, v2 is perpendicular to v1, creating the second axis in the layer plane
    dfloat3 v2;
    v2.x = (y * v1.z - z * v1.y);
    v2.y = (z * v1.x - x * v1.z);
    v2.z = (x * v1.y - y * v1.x);

    //calculate length and make v1 and v2 unitary
    dfloat v1l = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    dfloat v2l = sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

    v1.x = v1.x / v1l; 
    v1.y = v1.y / v1l; 
    v1.z = v1.z / v1l;

    v2.x = v2.x / v2l; 
    v2.y = v2.y / v2l; 
    v2.z = v2.z / v2l;

    int nodeIndex = 0;
    dfloat phase = 0;
    dfloat angle;
    for (int i = 0; i < nLayer; i++) {
        if (pattern) {
            phase = (i % 2) * M_PI / nNodesLayer;
        }
        for (int j = 0; j < nNodesLayer; j++) {
            angle = (dfloat)j * 2.0 * M_PI / nNodesLayer + phase;
            this->nodes[nodeIndex].pos.x = centerLayer[i].x + r * cos(angle) * v1.x + r * sin(angle) * v2.x; 
            this->nodes[nodeIndex].pos.y = centerLayer[i].y + r * cos(angle) * v1.y + r * sin(angle) * v2.y;
            this->nodes[nodeIndex].pos.z = centerLayer[i].z + r * cos(angle) * v1.z + r * sin(angle) * v2.z;

            this->nodes[i].S = this->pCenter.S/((dfloat)nNodesLayer * nLayer);

            nodeIndex++;
        }
    }
}


void Particle::makeCapsule(dfloat diameter, dfloat3 point1, dfloat3 point2, bool move, dfloat density, dfloat3 vel, dfloat3 w){

    //radius
    dfloat r = diameter/2.00;
    //length cylinder
    dfloat length = sqrt((point2.x - point1.x)*(point2.x - point1.x) + 
                         (point2.y - point1.y)*(point2.y - point1.y) + 
                         (point2.z - point1.z)*(point2.z - point1.z));
    //unit vector of capsule direction
    dfloat3 vec = dfloat3((point2.x - point1.x)/length,
                          (point2.y - point1.y)/length,
                          (point2.z - point1.z)/length);
    //number of slices
    int nSlices = (int)(length/MESH_SCALE);
    //array location of height of slices
    dfloat* z_cylinder = (dfloat*)malloc(nSlices * sizeof(dfloat));
    for (int i = 0; i < nSlices; i++)
        z_cylinder[i] = length * i / (nSlices - 1);

    // Calculate the number of nodes in each circle
    int nCirclePoints = (int)(2 * M_PI * r / MESH_SCALE);
    //array location of theta
    dfloat* theta = (dfloat*)malloc(nCirclePoints * sizeof(dfloat));
    for (int i = 0; i < nCirclePoints; i++)
        theta[i] = 2 * M_PI * i / (nCirclePoints - 1);

    // Cylinder surface nodes
    dfloat3* cylinder = (dfloat3*)malloc(nSlices * nCirclePoints * sizeof(dfloat3));

    for (int i = 0; i < nSlices; i++) {
        for (int j = 0; j < nCirclePoints; j++) {
            cylinder[i * nCirclePoints + j].x = r * cos(theta[j]);
            cylinder[i * nCirclePoints + j].y = r * sin(theta[j]);
            cylinder[i * nCirclePoints + j].z = z_cylinder[i];
        }
    }

    //spherical caps
    //calculate the number of slices and the angle of the slices
    int nSlices_sphere = (int)(M_PI * r / (2 * MESH_SCALE));
    dfloat* phi = (dfloat*)malloc(nSlices_sphere * sizeof(dfloat));
    for (int i = 0; i < nSlices_sphere; i++)
        phi[i] = (M_PI / 2) * i / (nSlices_sphere - 1);

    //calculate the number of nodes in the cap
    int cap_count = 0;
    for (int i = 0; i < nSlices_sphere; i++) {
        dfloat rSlice = r * cos(phi[i]);
        int nCirclePoints_sphere = (int)(2 * M_PI * rSlice / MESH_SCALE);
        for (int j = 0; j < nCirclePoints_sphere; j++) 
            cap_count++;
    }

    dfloat3* cap1 = (dfloat3*)malloc(cap_count * sizeof(dfloat3));
    dfloat3* cap2 = (dfloat3*)malloc(cap_count * sizeof(dfloat3));

    cap_count = 0;
    dfloat Xs, Ys, Zs;
    for (int i = 0; i < nSlices_sphere; i++) {
        dfloat rSlice = r * cos(phi[i]);
        int nCirclePoints_sphere = floor(2 * M_PI * rSlice / MESH_SCALE);
        
        for (int j = 0; j < nCirclePoints_sphere; j++)
            theta[j] = 2 * M_PI * j / (nCirclePoints_sphere - 1);


        for (int j = 0; j < nCirclePoints_sphere; j++) {
            Xs = rSlice * cos(theta[j]);
            Ys = rSlice * sin(theta[j]);
            Zs = r * sin(phi[i]);

            cap1[cap_count].x = Xs;
            cap1[cap_count].y = Ys;
            cap1[cap_count].z = -Zs;

            cap2[cap_count].x = Xs;
            cap2[cap_count].y = Ys;
            cap2[cap_count].z = Zs + length;
            cap_count++;
        }
    }

    dfloat3* cap1_filtered = (dfloat3*)malloc(cap_count * sizeof(dfloat3));
    dfloat3* cap2_filtered = (dfloat3*)malloc(cap_count * sizeof(dfloat3));

    //remove the nodes from inside, if necessary

    int cap_filtered_count = 0;
    for (int i = 0; i < cap_count; i++) {
        if (!(cap1[i].z >= 0 && cap1[i].z <= length)) {
            cap1_filtered[cap_filtered_count].x = cap1[i].x;
            cap1_filtered[cap_filtered_count].y = cap1[i].y;
            cap1_filtered[cap_filtered_count].z = cap1[i].z;
            cap_filtered_count++;
        }
    }

    cap_filtered_count = 0;
    for (int i = 0; i < cap_count; i++) {
        if (!(cap2[i].z >= 0 && cap2[i].z <= length)) {
            cap2_filtered[cap_filtered_count].x = cap2[i].x;
            cap2_filtered[cap_filtered_count].y = cap2[i].y;
            cap2_filtered[cap_filtered_count].z = cap2[i].z;
            cap_filtered_count++;
        }
    }

    //combine caps and cylinder
    int nCylinderPoints = nSlices * nCirclePoints;
    int nTotalPoints = nCylinderPoints + 2*cap_filtered_count;

    dfloat3* nodesTotal = (dfloat3*)malloc(nTotalPoints * sizeof(dfloat3));
    for (int i = 0; i < nCylinderPoints; i++) {
        nodesTotal[i].x = cylinder[i].x;
        nodesTotal[i].y = cylinder[i].y;
        nodesTotal[i].z = cylinder[i].z;
    }
    for (int i = 0; i < cap_filtered_count; i++) {
        nodesTotal[nCylinderPoints + i].x = cap1_filtered[i].x;
        nodesTotal[nCylinderPoints + i].y = cap1_filtered[i].y;
        nodesTotal[nCylinderPoints + i].z = cap1_filtered[i].z;
    }
    for (int i = 0; i < cap_filtered_count; i++) {
        nodesTotal[nCylinderPoints + cap_filtered_count + i].x = cap2_filtered[i].x;
        nodesTotal[nCylinderPoints + cap_filtered_count + i].y = cap2_filtered[i].y;
        nodesTotal[nCylinderPoints + cap_filtered_count + i].z = cap2_filtered[i].z;
    }

    //rotation
    //determine the quartetion which has to be used to rotate the vector (0,0,1) to vec
    //calculate dot product
    dfloat4 qf = compute_rotation_quart(dfloat3(0,0,1),vec);

    dfloat3 new_pos;
    for (int i = 0; i < nTotalPoints; i++) {
        new_pos = dfloat3(nodesTotal[i].x,nodesTotal[i].y,nodesTotal[i].z);
        
        new_pos = rotate_vector_by_quart_R(new_pos,qf);

        nodesTotal[i].x = new_pos.x + point1.x;
        nodesTotal[i].y = new_pos.y + point1.y;
        nodesTotal[i].z = new_pos.z + point1.z;

    }


    //DEFINITIONS
    
    dfloat volume = r*r*r*4*M_PI/3 + M_PI*r*r*length;
    dfloat sphereVol = r*r*r*4*M_PI/3;
    dfloat cylinderVol = M_PI*r*r*length;
    dfloat3 center = (point2 + point1)/2.0;

    this->pCenter.radius = r;
    this->pCenter.volume = sphereVol + cylinderVol;
    // Particle area
    this->pCenter.S = 4.0 * M_PI * r * r + 2*M_PI*r*length;
    // Particle density
    this->pCenter.density = density;

    // Particle center position
    this->pCenter.pos = center;
    //printf("pos center x %f y %f z %f \n",center.x,center.y,center.z);
    this->pCenter.pos_old = center;

    // Particle velocity
    this->pCenter.vel = vel;
    this->pCenter.vel_old = vel;

    // Particle rotation
    this->pCenter.w = w;
    this->pCenter.w_avg = w;
    this->pCenter.w_old = w;

    this->pCenter.q_pos.w = qf.w;
    this->pCenter.q_pos.x = qf.x;
    this->pCenter.q_pos.y = qf.y;
    this->pCenter.q_pos.z = qf.z;

    this->pCenter.q_pos_old.w = this->pCenter.q_pos.w;
    this->pCenter.q_pos_old.x = this->pCenter.q_pos.x;
    this->pCenter.q_pos_old.y = this->pCenter.q_pos.y;
    this->pCenter.q_pos_old.z = this->pCenter.q_pos.z;

    // Innertia momentum
    dfloat6 In;
    In.xx = this->pCenter.density * (cylinderVol*(r*r/2) + sphereVol * (2*r*r/5));
    In.yy = this->pCenter.density * (cylinderVol*(length*length/12 + r*r/4 ) + sphereVol * (2*r*r/5 + length*length/2 + 3*length*r/8));
    In.zz = this->pCenter.density * (cylinderVol*(length*length/12 + r*r/4 ) + sphereVol * (2*r*r/5 + length*length/2 + 3*length*r/8));
    In.xy = 0.0;
    In.xz = 0.0;
    In.yz = 0.0;

    dfloat4 q1 = compute_rotation_quart(dfloat3(1,0,0),vec);
    //rotate inertia 
    this->pCenter.I = rotate_inertia_by_quart(q1,In);

    this->pCenter.q_pos.w = qf.w;
    this->pCenter.q_pos.x = qf.x;
    this->pCenter.q_pos.y = qf.y;
    this->pCenter.q_pos.z = qf.z;

    this->pCenter.q_pos_old.w = this->pCenter.q_pos.w;
    this->pCenter.q_pos_old.x = this->pCenter.q_pos.x;
    this->pCenter.q_pos_old.y = this->pCenter.q_pos.y;
    this->pCenter.q_pos_old.z = this->pCenter.q_pos.z; 



    this->pCenter.f.x = 0.0;
    this->pCenter.f.y = 0.0;
    this->pCenter.f.z = 0.0;

    this->pCenter.f_old.x = 0.0;
    this->pCenter.f_old.y = 0.0;
    this->pCenter.f_old.z = 0.0;

    this->pCenter.M.x = 0.0;
    this->pCenter.M.y = 0.0;
    this->pCenter.M.z = 0.0;

    this->pCenter.M_old.x = 0.0;
    this->pCenter.M_old.y = 0.0;
    this->pCenter.M_old.z = 0.0;


    this->pCenter.movable = move;


    this->pCenter.collision.shape = CAPSULE;
    this->pCenter.collision.semiAxis = point1 - center;
    for(int i = 0; i <MAX_ACTIVE_COLLISIONS;i++){
        this->pCenter.collision.collisionPartnerIDs[i] = -1;
        this->pCenter.collision.tangentialDisplacements[i] = dfloat3(0,0,0);
        this->pCenter.collision.lastCollisionStep[i] = -1;
    }

    
    this->numNodes = nTotalPoints;

    this->nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * this->numNodes);

    //convert nodes info

    for (int nodeIndex = 0; nodeIndex < nTotalPoints; nodeIndex++) {
        this->nodes[nodeIndex].pos.x = nodesTotal[nodeIndex].x ;
        this->nodes[nodeIndex].pos.y = nodesTotal[nodeIndex].y ;
        this->nodes[nodeIndex].pos.z = nodesTotal[nodeIndex].z ;

        this->nodes[nodeIndex].vel.x = vel.x + w.y * this->nodes[nodeIndex].pos.z - w.z * this->nodes[nodeIndex].pos.y;
        this->nodes[nodeIndex].vel.y = vel.y + w.z * this->nodes[nodeIndex].pos.x - w.x * this->nodes[nodeIndex].pos.z;
        this->nodes[nodeIndex].vel.z = vel.z + w.x * this->nodes[nodeIndex].pos.y - w.y * this->nodes[nodeIndex].pos.x;

        this->nodes[nodeIndex].vel_old.x = vel.x + w.y * this->nodes[nodeIndex].pos.z - w.z * this->nodes[nodeIndex].pos.y;
        this->nodes[nodeIndex].vel_old.y = vel.y + w.z * this->nodes[nodeIndex].pos.x - w.x * this->nodes[nodeIndex].pos.z;
        this->nodes[nodeIndex].vel_old.z = vel.z + w.x * this->nodes[nodeIndex].pos.y - w.y * this->nodes[nodeIndex].pos.x;

        if(nodeIndex < nCylinderPoints)
            this->nodes[nodeIndex].S = (length*2*M_PI*r)/nCylinderPoints;
        else
            this->nodes[nodeIndex].S = (4*M_PI*r*r)/(2*cap_filtered_count);
    }

    //free
    free(z_cylinder);
    free(theta);
    free(cylinder);
    free(phi);
    free(cap1);
    free(cap2);
    free(cap1_filtered);
    free(cap2_filtered);
    free(nodesTotal);
 
     // Update old position value
     this->pCenter.pos_old = this->pCenter.pos;   

    for(int ii = 0;ii<nTotalPoints;ii++){
         ParticleNode* node_j = &(this->nodes[ii]);
         printf("%f,%f,%f \n",node_j->pos.x,node_j->pos.y,node_j->pos.z );
    }

}

void Particle::makeEllipsoid(dfloat3 diameter, dfloat3 center, dfloat3 vec, dfloat angleMag, bool move, dfloat density, dfloat3 vel, dfloat3 w)
{
    
    dfloat a, b, c;  // principal radius

    unsigned int i, j;

    a = diameter.x / 2.0;
    b = diameter.y / 2.0;
    c = diameter.z / 2.0;

    this->pCenter.radius = POW_FUNCTION(a*b*c,1.0/3.0);
    this->pCenter.volume = a*b*c*4*M_PI/3;

    // Particle density
    this->pCenter.density = density;

    // Particle center position
    this->pCenter.pos = center;
    this->pCenter.pos_old = center;

    // Particle velocity
    this->pCenter.vel = vel;
    this->pCenter.vel_old = vel;

    // Particle rotation
    this->pCenter.w = w;
    this->pCenter.w_avg = w;
    this->pCenter.w_old = w;

    // Innertia momentum
    dfloat6 In;
    In.xx = 0.2 * this->pCenter.volume * this->pCenter.density * (b*b + c*c);
    In.yy = 0.2 * this->pCenter.volume * this->pCenter.density * (a*a + c*c);
    In.zz = 0.2 * this->pCenter.volume * this->pCenter.density * (a*a + b*b);
    In.xy = 0.0;
    In.xz = 0.0;
    In.yz = 0.0;

    this->pCenter.f.x = 0.0;
    this->pCenter.f.y = 0.0;
    this->pCenter.f.z = 0.0;

    this->pCenter.f_old.x = 0.0;
    this->pCenter.f_old.y = 0.0;
    this->pCenter.f_old.z = 0.0;

    this->pCenter.M.x = 0.0;
    this->pCenter.M.y = 0.0;
    this->pCenter.M.z = 0.0;

    this->pCenter.M_old.x = 0.0;
    this->pCenter.M_old.y = 0.0;
    this->pCenter.M_old.z = 0.0;


    this->pCenter.movable = move;

    //printf("%f %f %f %f %f \n", this->pCenter.radius,this->pCenter.volume,this->pCenter.I.xx ,this->pCenter.I.yy,this->pCenter.I.zz);

    // Particle area
    dfloat p = 1.6075; //aproximation
    dfloat ab = POW_FUNCTION(a*b,p); 
    dfloat ac = POW_FUNCTION(a*c,p); 
    dfloat bc = POW_FUNCTION(b*c,p); 

    this->pCenter.S = 4*M_PI*POW_FUNCTION((ab + ac + bc)/3,1.0/p);



    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    dfloat scaling = myMin(myMin(a, b), c);
    
    a /= scaling;
    b /= scaling;
    c /= scaling;
    ab = POW_FUNCTION(a*b,p); 
    ac = POW_FUNCTION(a*c,p); 
    bc = POW_FUNCTION(b*c,p); 

    dfloat SS =  4*M_PI*POW_FUNCTION((ab + ac + bc)/3.0,1.0/p);
    int numberNodes = (int)(SS * POW_FUNCTION(scaling, 3.0) /(4*M_PI/MESH_SCALE));

    // Particle num nodes
    this->numNodes = numberNodes;

    //#############################################################################

    // Allocate memory for positions and forces
    dfloat *phi = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat *theta = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat* posx = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat* posy = (dfloat *)malloc(numberNodes* sizeof(dfloat));
    dfloat* posz = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat* fx = (dfloat *)malloc(numberNodes* sizeof(dfloat));
    dfloat* fy = (dfloat *)malloc(numberNodes* sizeof(dfloat));
    dfloat* fz = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    
    

    // Initialize random positions of charges on the ellipsoid surface
    for (int i = 0; i < numberNodes; i++) {
        phi[i] = 2 * M_PI * ((dfloat)rand() / (dfloat)RAND_MAX);   // Angle in XY plane
        theta[i] = M_PI * ((dfloat)rand() / (dfloat)RAND_MAX);     // Angle from Z axis

        posx[i] = 1 * sin(theta[i]) * cos(phi[i]); // x coordinate
        posy[i] = 1 * sin(theta[i]) * sin(phi[i]); // y coordinate
        posz[i] = 1 * cos(theta[i]);               // z coordinate
    }


    // Constants
    dfloat base_k = 1.0; // Base Coulomb's constant (assuming unit charge)
    dfloat rij[3];
    dfloat r;
    dfloat F;
    dfloat unit_rij[3];
    dfloat force_scale_factor;

    
    for (int iter = 0; iter < 300; iter++) {
        // Initialize force accumulator
        for (int i = 0; i < numberNodes; i++) {
            fx[i] = 0.0;
            fy[i] = 0.0;
            fz[i] = 0.0;
        }

        // Compute pairwise forces and update positions
        for (int i = 0; i < numberNodes; i++) {
            for (int j = 0; j < numberNodes; j++) {
                if (i != j) { // not the same node
                    // Vector from node j to node i

                    rij[0] = posx[i] - posx[j];
                    rij[1] = posy[i] - posy[j];
                    rij[2] = posz[i] - posz[j];

                    // Distance between node i and node j
                    r = sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);

                    // Coulomb's force magnitude
                    F = base_k / (r * r);

                    // Direction of force
                    unit_rij[0] = rij[0] / r;
                    unit_rij[1] = rij[1] / r;
                    unit_rij[2] = rij[2] / r;

                    // Accumulate force on nodes i
                    fx[i] += F * unit_rij[0];
                    fy[i] += F * unit_rij[1];
                    fz[i] += F * unit_rij[2];
                }
            }
        }
        // Update positions of nodes
        for (int i = 0; i < numberNodes; i++) {
            posx[i] += 10 * fx[i] / (numberNodes * numberNodes);
            posy[i] += 10 * fy[i] / (numberNodes * numberNodes);
            posz[i] += 10 * fz[i] / (numberNodes * numberNodes);
        }

        // Project updated positions back onto the ellipsoid surface
        for (int i = 0; i < numberNodes; i++) {
            // Calculate the current point's distance to the center along each axis
            force_scale_factor = sqrt(posx[i]*posx[i] +
                                posy[i]*posy[i] +
                                posz[i]*posz[i]);
            // Rescale to ensure it lies on the ellipsoid surface
            posx[i] /= force_scale_factor;
            posy[i] /= force_scale_factor;
            posz[i] /= force_scale_factor;
        }
    }
    //convert into elipsoid 
    for (int i = 0; i < numberNodes; i++) {
    posx[i] *= a*scaling;
    posy[i] *= b*scaling;
    posz[i] *= c*scaling;
    }
    
 
    /*
    for(int ii = 0;ii<N;ii++){
         printf("%f %f %f \n",posx[ii],posy[ii],posz[ii] );
     }*/
   

    this->nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * this->numNodes);

    for (int nodeIndex = 0; nodeIndex < numberNodes; nodeIndex++) {
        this->nodes[nodeIndex].pos.x = posx[nodeIndex];
        this->nodes[nodeIndex].pos.y = posy[nodeIndex];
        this->nodes[nodeIndex].pos.z = posz[nodeIndex];

        this->nodes[nodeIndex].vel.x = vel.x + w.y * this->nodes[nodeIndex].pos.z - w.z * this->nodes[nodeIndex].pos.y;
        this->nodes[nodeIndex].vel.y = vel.y + w.z * this->nodes[nodeIndex].pos.x - w.x * this->nodes[nodeIndex].pos.z;
        this->nodes[nodeIndex].vel.z = vel.z + w.x * this->nodes[nodeIndex].pos.y - w.y * this->nodes[nodeIndex].pos.x;

        this->nodes[nodeIndex].vel_old.x = vel.x + w.y * this->nodes[nodeIndex].pos.z - w.z * this->nodes[nodeIndex].pos.y;
        this->nodes[nodeIndex].vel_old.y = vel.y + w.z * this->nodes[nodeIndex].pos.x - w.x * this->nodes[nodeIndex].pos.z;
        this->nodes[nodeIndex].vel_old.z = vel.z + w.x * this->nodes[nodeIndex].pos.y - w.y * this->nodes[nodeIndex].pos.x;


        // the area of sphere segment is divided by the number of node in the layer, so all nodes have the same area
        this->nodes[nodeIndex].S = this->pCenter.S / numberNodes;
    }

    //%%%%%%%% ROTATION 
    //current state rotation quartenion (STANDARD ROTATION OF 90º IN THE X - AXIS)


    dfloat4 q2 = axis_angle_to_quart(vec,angleMag);

    //rotate inertia 
    this->pCenter.I = rotate_inertia_by_quart(q2,In);

    dfloat3 new_pos;
    for (i = 0; i < numberNodes; i++) {

        new_pos = dfloat3(this->nodes[i].pos.x,this->nodes[i].pos.y,this->nodes[i].pos.z);
        new_pos = rotate_vector_by_quart_R(new_pos,q2);

        this->nodes[i].pos.x = new_pos.x + center.x;
        this->nodes[i].pos.y = new_pos.y + center.y;
        this->nodes[i].pos.z = new_pos.z + center.z;
    }

    this->pCenter.q_pos.w = q2.w;
    this->pCenter.q_pos.x = q2.x;
    this->pCenter.q_pos.y = q2.y;
    this->pCenter.q_pos.z = q2.z;

    this->pCenter.q_pos_old.w = this->pCenter.q_pos.w;
    this->pCenter.q_pos_old.x = this->pCenter.q_pos.x;
    this->pCenter.q_pos_old.y = this->pCenter.q_pos.y;
    this->pCenter.q_pos_old.z = this->pCenter.q_pos.z;

    this->pCenter.collision.semiAxis  = center + a*scaling*dfloat3(1,0,0);
    this->pCenter.collision.semiAxis2 = center + b*scaling*dfloat3(0,1,0);
    this->pCenter.collision.semiAxis3 = center + c*scaling*dfloat3(0,0,1);

    vec = vector_normalize(vec);
    if(angleMag != 0.0){
        const dfloat q0 = cos(0.5*angleMag);

        const dfloat qi = (vec.x/angleMag) * sin (0.5*angleMag);
        const dfloat qj = (vec.y/angleMag) * sin (0.5*angleMag);
        const dfloat qk = (vec.z/angleMag) * sin (0.5*angleMag);

        const dfloat tq0m1 = (q0*q0) - 0.5;


        this->pCenter.collision.semiAxis  = center + a*scaling*dfloat3(1,0,0);
        this->pCenter.collision.semiAxis2 = center + b*scaling*dfloat3(0,1,0);
        this->pCenter.collision.semiAxis3 = center + c*scaling*dfloat3(0,0,1);

        this->pCenter.collision.semiAxis  = rotate_vector_by_quart_R(this->pCenter.collision.semiAxis  - center,q2) + center;
        this->pCenter.collision.semiAxis2 = rotate_vector_by_quart_R(this->pCenter.collision.semiAxis2 - center,q2) + center;
        this->pCenter.collision.semiAxis3 = rotate_vector_by_quart_R(this->pCenter.collision.semiAxis3 - center,q2) + center;
    }



    for(int i = 0; i <MAX_ACTIVE_COLLISIONS;i++){
        this->pCenter.collision.collisionPartnerIDs[i] = -1;
        this->pCenter.collision.tangentialDisplacements[i] = dfloat3(0,0,0);
        this->pCenter.collision.lastCollisionStep[i] = -1;
    }

    this->pCenter.collision.shape = ELLIPSOID;
    for(int ii = 0;ii<numberNodes;ii++){
         ParticleNode* node_j = &(this->nodes[ii]);
         printf("%f,%f,%f \n",node_j->pos.x,node_j->pos.y,node_j->pos.z );
     }

    // Free allocated memory

    free(phi);
    free(theta);
    free(posx);
    free(posy);
    free(posz);
    free(fx);
    free(fy);
    free(fz);

     // Update old position value
    this->pCenter.pos_old = this->pCenter.pos;
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}


#endif // !IBM