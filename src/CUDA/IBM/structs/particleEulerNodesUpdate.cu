#include "particleEulerNodesUpdate.h"


ParticleEulerNodesUpdate::particleEulerNodesUpdate(){
    eulerMaskArray = nullptr;
    eulerIndexesUpdate = nullptr;
    maxEulerNodes = 0;
    currEulerNodes = 0;
    pCenterMovable = nullptr;
    particlesLastPos = nullptr;
    hasFixed = false;
    nParticlesMovable = 0;
}

ParticleEulerNodesUpdate::~particleEulerNodesUpdate(){

}

#if IBM_EULER_OPTIMIZATION
__host__
void ParticleEulerNodesUpdate::initializeEulerNodes(ParticleCenter p[NUM_PARTICLES]){
    
    int nParticlesFixed = 0;
    int idxFixed[NUM_PARTICLES], idxMoving[NUM_PARTICLES];
    // Pre process particles to calculate some necessary values
    for(int i = 0; i <  NUM_PARTICLES; i++){
        if(p[i].movable){
            idxMoving[this->nParticlesMovable] = i;
            this->nParticlesMovable += 1;
        } else {
            idxFixed[nParticlesFixed] = i;
            nParticlesFixed += 1;
            this->hasFixed = true;
        }
        // TODO: UPDATE THIS TO BE DONE BY PARTICLES FOR OTHER GEOMETRIES
        // Sum sphere shell volume considering
        // r_out=r_sphere+P_DIST+SHELL_THICKNESS+0.5
        // r_int=r_sphere-P_DIST-SHELL_THICKNESS-0.5
        // V = 4*pi*(r_out^3-r_in^3)/3
        // +0.5 in r is to make sure that all possible nodes are counted
        // SHELL_THICKNESS only for moving particles
        dfloat sum_term = P_DIST+0.5;
        if(p[i].movable){
            sum_term += IBM_EULER_SHELL_THICKNESS;
        }
        dfloat volumeShell = 4 * M_PI / 3.0 * pow(p[i].radius+sum_term, 3);
        volumeShell -= 4 * M_PI / 3.0 * pow(myMax(0, p[i].radius-sum_term), 3);
        // Volume of shell (ceil)
        this->maxEulerNodes += (int)volumeShell + 1;
    }

    // Allocate variables
    // Allocate indexes of Euler nodes to update
    checkCudaErrors(cudaMallocManaged((void**)&(this->eulerIndexesUpdate), 
        this->maxEulerNodes*sizeof(size_t)));
    // Allocate mask array
    this->eulerMaskArray = (uint32_t*) malloc((size_t)NX*NY*NZ*sizeof(uint32_t));
    // Allocate array of pointers to particleCenters, for moving particles
    this->pCenterMovable = (ParticleCenter**)malloc(this->nParticlesMovable*sizeof(ParticleCenter*));
    // Allocate particles last position array, for moving particles
    this->particlesLastPos = (dfloat3*)malloc(this->nParticlesMovable*sizeof(dfloat3));

    for(int i=0; i < nParticlesFixed; i++){
        // Mask for fixes particles is always 0b1
        this->updateEulerNodes(&p[idxFixed[i]], 0b1);
    }

    const char shift = this->hasFixed? 1 : 0;

    for(int i=0; i < this->nParticlesMovable; i++){
        ParticleCenter* mp = &(p[idxMoving[i]]);
        this->pCenterMovable[i] = mp;
        this->particlesLastPos[i] = mp->pos;
        // Mask for fixes particles is 0b1 shifted its index +shift to the left
        this->updateEulerNodes(mp, 0b1<<(shift+i));
    }
}

__host__
void ParticleEulerNodesUpdate::checkParticlesMovement(){
    
}

__host__
void ParticleEulerNodesUpdate::removeUnneededEulerNodes(){
    
}

__host__
void ParticleEulerNodesUpdate::updateEulerNodes(ParticleCenter* pc, uint32_t mask){
    dfloat3 pos = pc->pos;
    dfloat radius = pc->radius;
    // TODO: update this for other geometries than sphere
    dfloat sphereShellThick = P_DIST;
    if(pc->movable)
        sphereShellThick += IBM_EULER_SHELL_THICKNESS;
    dfloat addTerm = radius+sphereShellThick;

    int maxX= myMin(pos.x+addTerm+1, NX); // +1 for ceil
    int maxY= myMin(pos.y+addTerm+1, NY); 
    int maxZ= myMin(pos.z+addTerm+1, NZ); 

    int minX = myMax(pos.x-addTerm, 0);
    int minY = myMax(pos.y-addTerm, 0);
    int minZ = myMax(pos.z-addTerm, 0);
    
    const dfloat maxDistSq = (radius+sphereShellThick)*(radius+sphereShellThick);
    const dfloat minDistSq = (radius-sphereShellThick)*(radius-sphereShellThick);
    
    #if IBM_DEBUG
    unsigned int hit = 0;
    const unsigned int totalNodes = (maxZ-minZ+1)*(maxY-minY+1)*(maxX-minX+1);
    #endif

    for(int k=minZ; k <= maxZ; k++){
        for(int j=minY; j <= maxY; j++){
            for(int i=minX; i <= maxX; i++){
                // Distance squared
                dfloat distSq = ((k-pos.z)*(k-pos.z)
                    +(j-pos.y)*(j-pos.y)
                    +(i-pos.x)*(i-pos.x));
                if(distSq <= maxDistSq && distSq >= minDistSq){
                    size_t idx = idxScalar(i, j, k);
                    // Add Euler indexes to update
                    this->eulerIndexesUpdate[this->currEulerNodes] = idx;
                    // Update mask array
                    this->eulerMaskArray[idx] |= mask;
                    // Add one to current number of nodes
                    this->currEulerNodes += 1;
                    #if IBM_DEBUG
                    hit += 1;
                    #endif
                }
            }
        }
    }
    #if IBM_DEBUG
    printf("Hit ratio for mask %x with %d nodes: %%%.2f\n", mask, totalNodes, 100.0*(dfloat)hit/totalNodes);
    #endif

}

__global__
void ibmEulerCopyVelocities(dfloat3SoA dst, dfloat3SoA src, size_t* eulerIdxsUpdate, unsigned int currEulerNodes){
    const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i >= currEulerNodes)
        return;

    dst.x[i] = src.x[i];
    dst.y[i] = src.y[i];
    dst.z[i] = src.z[i];
}


#endif