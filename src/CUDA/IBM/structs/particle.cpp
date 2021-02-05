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
           (unsigned long)((totalIbmNodes * sizeof(particleNode) + NUM_PARTICLES * sizeof(particleCenter)) / BYTES_PER_MB));
    fflush(stdout);

    printf("Allocating particles in GPU... \t"); fflush(stdout);
    this->nodesSoA.allocateMemory(totalIbmNodes);
    // Allocate particle center array
    checkCudaErrors(
        cudaMallocManaged((void**)&(this->pCenterArray), sizeof(ParticleCenter) * NUM_PARTICLES));
    printf("Particles allocated in GPU!\n"); fflush(stdout);

    size_t baseIdx = 0;

    printf("Optimizig memory layout of particles for GPU... \t"); fflush(stdout);

    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        this->nodesSoA.copyNodesFromParticle(particles[p], p, baseIdx);
        this->pCenterArray[p] = particles[p].pCenter;

        baseIdx += particles[p].numNodes;
    }

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

void ParticlesSoA::freeNodesAndCenters(){
    this->nodesSoA.freeMemory();
    cudaFree(this->pCenterArray);
    this->pCenterArray = nullptr;
}

Particle makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move)
{
    // Particle to be returned
    Particle particleRet;
    // Maximum number of layer of sphere
    //unsigned int maxNumLayers = 5000;
    // Number of layers in sphere
    unsigned int nLayer;
    // Number of nodes per layer in sphere
    unsigned int* nNodesLayer;
    // Angles in polar coordinates and node area
    dfloat *theta, *zeta, *S;

    dfloat phase = 0.0;

    // particleRet.pCenter = ParticleCenter();

    // Define the properties of the particle
    dfloat r = diameter / 2.0;
    dfloat volume = r*r*r*4*M_PI/3;

    particleRet.pCenter.radius = r;
    particleRet.pCenter.volume = r*r*r*4*M_PI/3;
    // Particle area
    particleRet.pCenter.S = 4.0 * M_PI * r * r;

    // Particle center position
    particleRet.pCenter.pos.x = center.x;
    particleRet.pCenter.pos.y = center.y;
    particleRet.pCenter.pos.z = center.z;

    // Innertia momentum
    particleRet.pCenter.I.x = 2.0 * volume * PARTICLE_DENSITY * r * r / 5.0;
    particleRet.pCenter.I.y = 2.0 * volume * PARTICLE_DENSITY * r * r / 5.0;
    particleRet.pCenter.I.z = 2.0 * volume * PARTICLE_DENSITY * r * r / 5.0;

    particleRet.pCenter.movable = move;
    // Number of layers in the sphere
    nLayer = (unsigned int)(2.0 * sqrt(2) * r / MESH_SCALE + 1.0); 

    nNodesLayer = (unsigned int*)malloc(nLayer * sizeof(unsigned int));
    theta = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));
    zeta = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));
    S = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));

    particleRet.numNodes = 0;
    for (int i = 0; i <= nLayer; i++) {
        // Angle of each layer
        theta[i] = M_PI * ((double)i / (double)nLayer - 0.5); 
        // Determine the number of node per layer
        nNodesLayer[i] = (unsigned int)(1.5 + cos(theta[i]) * nLayer * sqrt(3)); 
        // Total number of nodes on the sphere
        particleRet.numNodes += nNodesLayer[i]; 
        zeta[i] = r * sin(theta[i]); // Height of each layer
    }

    /*
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
    */

    particleRet.nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * particleRet.numNodes);

    ParticleNode* first_node = &(particleRet.nodes[0]);

    // South node - define all properties
    first_node->pos.x = center.x;
    first_node->pos.y = center.y;
    first_node->pos.z = center.z + r * sin(theta[0]);

    // first_node->S = S[0];

    int nodeIndex = 1;
    for (int i = 1; i < nLayer; i++) {
        if (i % 2 == 1) {
            // Calculate the phase of the segmente to avoid a straight point line
            phase = phase + M_PI / nNodesLayer[i];
        }

        for (int j = 0; j < nNodesLayer[i]; j++) {
            // Determine the properties of each node in the mid layers
            particleRet.nodes[nodeIndex].pos.x = center.x + r * cos(theta[i]) * cos((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            particleRet.nodes[nodeIndex].pos.y = center.y + r * cos(theta[i]) * sin((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            particleRet.nodes[nodeIndex].pos.z = center.z + r * sin(theta[i]);

            // The area of sphere segment is divided by the number of node
            // in the layer, so all nodes have the same area
            // particleRet.nodes[nodeIndex].S = particleRet.pCenter.S/particleRet.numNodes;

            // Add one node
            nodeIndex++;
        }
    }

    // North pole -define all properties
    ParticleNode* last_node = &(particleRet.nodes[particleRet.numNodes-1]);
    
    last_node->pos.x = center.x;
    last_node->pos.y = center.y;
    last_node->pos.z = center.z + r * sin(theta[nLayer]);
    // last_node->S = S[nLayer];

    unsigned int numNodes = particleRet.numNodes;
    dfloat dA =  particleRet.pCenter.S/particleRet.numNodes;
    for(int i = 0; i < numNodes; i++){
        particleRet.nodes[i].S = dA;
    }

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
                ParticleNode* node_i = &(particleRet.nodes[i]);

                for (int j = i+1; j < numNodes; j++) {

                    ParticleNode* node_j = &(particleRet.nodes[j]);

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
                
                ParticleNode* node_i = &(particleRet.nodes[i]);

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
            ParticleNode* node_i = &(particleRet.nodes[i]);

            node_i->S = particleRet.pCenter.S / (numNodes);

            node_i->pos.x += center.x;
            node_i->pos.y += center.y;
            node_i->pos.z += center.z;
        }

        // Free coulomb force
        free(cForce);
    }

    // Free allocated variables
    free(nNodesLayer);
    free(theta);
    free(zeta);
    free(S);

    // Update old position value
    particleRet.pCenter.pos_old = particleRet.pCenter.pos;
    printf("\n" );  
    for (int i = 0; i < numNodes; i++) {
        printf("%f; %f ; %f\n",  particleRet.nodes[i].pos.x, particleRet.nodes[i].pos.y, particleRet.nodes[i].pos.z);
    }

    return particleRet;
}

#endif // !IBM