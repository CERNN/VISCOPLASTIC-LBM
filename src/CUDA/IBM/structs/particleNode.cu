#include "particleNode.h"
#include "particle.h"

__host__ __device__
ParticleNodeSoA::particleNodeSoA()
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
    this->numNodes = 0;
}

__host__ __device__
ParticleNodeSoA::~particleNodeSoA()
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
    this->numNodes = 0;
}

#ifdef IBM
void ParticleNodeSoA::allocateMemory(unsigned int numMaxNodes)
{
    this->pos.allocateMemory((size_t) numMaxNodes);
    this->vel.allocateMemory((size_t) numMaxNodes);
    this->vel_old.allocateMemory((size_t) numMaxNodes);
    this->f.allocateMemory((size_t) numMaxNodes);
    this->deltaF.allocateMemory((size_t) numMaxNodes);

    checkCudaErrors(
        cudaMallocManaged((void**)&(this->S), sizeof(dfloat) * numMaxNodes));
    checkCudaErrors(
        cudaMallocManaged((void**)&(this->particleCenterIdx), sizeof(unsigned int) * numMaxNodes));
}

void ParticleNodeSoA::freeMemory()
{
    this->numNodes = 0;

    this->pos.freeMemory();
    this->vel.freeMemory();
    this->vel_old.freeMemory();
    this->f.freeMemory();
    this->deltaF.freeMemory();

    cudaFree(this->S);
    cudaFree(this->particleCenterIdx);
}

bool is_inside_gpu(dfloat3 pos, unsigned int n_gpu){
    return pos.z >= n_gpu*NZ && pos.z < (n_gpu+1)*NZ;
}

void ParticleNodeSoA::copyNodesFromParticle(Particle p, unsigned int pCenterIdx, unsigned int n_gpu)
{
    const int baseIdx = this->numNodes;
    int nodesAdded = 0;
    for (int i = 0; i < p.numNodes; i++)
    {
        if(!is_inside_gpu(p.nodes[i].pos, n_gpu))
            continue;

        this->particleCenterIdx[nodesAdded+baseIdx] = pCenterIdx;

        this->pos.copyValuesFromFloat3(p.nodes[i].pos, nodesAdded+baseIdx);
        this->vel.copyValuesFromFloat3(p.nodes[i].vel, nodesAdded+baseIdx);
        this->vel_old.copyValuesFromFloat3(p.nodes[i].vel_old, nodesAdded+baseIdx);
        this->f.copyValuesFromFloat3(p.nodes[i].f, nodesAdded+baseIdx);
        this->deltaF.copyValuesFromFloat3(p.nodes[i].deltaF, nodesAdded+baseIdx);
        this->S[nodesAdded+baseIdx] = p.nodes[i].S;
        nodesAdded += 1;
    }

    this->numNodes += nodesAdded;
}

void ParticleNodeSoA::leftShiftNodesSoA(int idx, int left_shift){
    this->particleCenterIdx[idx-left_shift] = this->particleCenterIdx[idx];
    this->S[idx-left_shift] = this->S[idx];
    this->pos.leftShift(idx, left_shift);
    this->vel.leftShift(idx, left_shift);
    this->vel_old.leftShift(idx, left_shift);
    this->f.leftShift(idx, left_shift);
    this->deltaF.leftShift(idx, left_shift);
}

#endif // !IBM