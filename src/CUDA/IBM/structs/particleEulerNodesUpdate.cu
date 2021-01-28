#include "particleEulerNodesUpdate.h"


ParticleEulerNodesUpdate::particleEulerNodesUpdate(){
    eulerMaskArray = nullptr;
    eulerIndexesUpdate = nullptr;
    maxEulerNodes = 0;
    currEulerNodes = 0;
    eulerFixedNodes = 0;
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
        eulerFixedNodes += this->updateEulerNodes(&p[idxFixed[i]], 0b1);
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
void ParticleEulerNodesUpdate::freeEulerNodes(){
    // Free variables
    checkCudaErrors(cudaFree(this->eulerIndexesUpdate));
    free(this->eulerMaskArray);
    free(this->pCenterMovable);
    free(this->particlesLastPos);
}

__host__
void ParticleEulerNodesUpdate::checkParticlesMovement(){
    // No need to check for movement if there are no movable particles
    if(this->nParticlesMovable == 0)
        return;

    uint32_t maskRemove = 0b0;
    const unsigned int shift = this->hasFixed? 1 : 0;

    for(int i = 0; i < nParticlesMovable; i++){
        dfloat3 pos = this->pCenterMovable[i]->pos;
        dfloat3 posOld = this->particlesLastPos[i];
        dfloat distSq = (
            (pos.x-posOld.x)*(pos.x-posOld.x)+
            (pos.y-posOld.y)*(pos.y-posOld.y)+
            (pos.z-posOld.z)*(pos.z-posOld.z));
        // Check if particle moved more than IBM_EULER_SHELL_THICKNESS
        if(distSq >= (IBM_EULER_UPDATE_DIST*IBM_EULER_UPDATE_DIST)){
            // Add its particle to remove/update euler nodes
            maskRemove |= 0b1 << i+shift;
        }
    }
    // If there is any mask to remove/update (particles that moved)
    if(maskRemove > 0){
        removeUnneededEulerNodes(maskRemove);
        int count = 0;
        // Remove bit from fixed particle
        maskRemove >>= shift;
        // While there are still bits from mask (particles that moved) to process
        while(maskRemove > 0){
            if(maskRemove & 0b1){
                this->updateEulerNodes(this->pCenterMovable[count], 0b1 << (count+shift));
            }
            maskRemove >>= 1;
            count += 1;
        }
    }
}

__host__
void ParticleEulerNodesUpdate::removeUnneededEulerNodes(uint32_t maskRemove){
    // Indexes to write to the left (keep coalesced array)
    // When a node that isn't used anymore is found, this is incremented by one
    // After its first increment, all other indexes that are read are written to
    // the array back again shifted idxLeft to the left
    // Obs.: "left" means closer to start of the array

    unsigned int idxLeft = 0;
    unsigned int newCurrEulerNodes = (int)this->currEulerNodes;
    #if IBM_DEBUG
    int idxsUpdated = 0;
    #endif
    // Start after fixed nodes
    for(int i = this->eulerFixedNodes; i < this->currEulerNodes; i++){
        const size_t idx = eulerIndexesUpdate[i];
        const uint32_t val = this->eulerMaskArray[idx];
        // Val with mask (~maskRemove) passed
        const uint32_t valWithMask = val & ~maskRemove;
        // If maskVal with bits removed is 0, it means that the node 
        // should no longer be used
        if(valWithMask == 0){
            idxLeft += 1;
            newCurrEulerNodes -= 1;
        }
        // If node is used, the value must be writed back to the array, 
        // idxLeft to the left and with the bits removed
        else {
            // Check if it is required to write back
            if(idxLeft > 0 || val != valWithMask) {
                #if IBM_DEBUG
                idxsUpdated += 1;
                #endif
                this->eulerIndexesUpdate[i-idxLeft] = valWithMask;
            }
        }
    }

    #if IBM_DEBUG
    printf("For mask %x, Idx removed: %d; Updated: %d\n", maskRemove, 
        this->currEulerNodes - newCurrEulerNodes, idxsUpdated);
    #endif
    this->currEulerNodes = (unsigned int)newCurrEulerNodes;
}

__host__
unsigned int ParticleEulerNodesUpdate::updateEulerNodes(ParticleCenter* pc, uint32_t mask){
    dfloat3 pos = pc->pos;
    dfloat radius = pc->radius;
    // TODO: update this for other geometries than sphere
    dfloat sphereShellThick = P_DIST;
    if(pc->movable)
        sphereShellThick += IBM_EULER_SHELL_THICKNESS;
    dfloat addTerm = radius+sphereShellThick;

    const int maxX = myMin(pos.x+addTerm+1, NX); // +1 for ceil
    const int maxY = myMin(pos.y+addTerm+1, NY); 
    const int maxZ = myMin(pos.z+addTerm+1, NZ); 

    const int minX = myMax(pos.x-addTerm, 0);
    const int minY = myMax(pos.y-addTerm, 0);
    const int minZ = myMax(pos.z-addTerm, 0);

    const dfloat maxDistSq = (radius+sphereShellThick)*(radius+sphereShellThick);
    const dfloat minDistSq = (radius-sphereShellThick)*(radius-sphereShellThick);
    unsigned int oldCurrNodes = this->currEulerNodes;
    #if IBM_DEBUG
    unsigned int hit = 0;
    const unsigned int totalNodes = (maxZ-minZ+1)*(maxY-minY+1)*(maxX-minX+1);
    #endif
    printf("pos %f %f %f\n", pos.x, pos.y, pos.z);
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

    // Return number of added nodes
    return this->currEulerNodes - oldCurrNodes;
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