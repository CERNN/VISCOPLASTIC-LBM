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
        this->pCenterArray[p] = particles[p].bodyCenter;

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
    unsigned int maxNumLayers = 5000;
    // Number of layers in sphere
    unsigned int nLayer;
    // Number of nodes per layer in sphere
    unsigned int* nNodesLayer;
    // Angles in polar coordinates and node area
    dfloat *theta, *zeta, *S;
    // Auxiliary counters
    unsigned int i, j;

    nNodesLayer = (unsigned int*)malloc(maxNumLayers * sizeof(unsigned int));
    theta = (dfloat*)malloc(maxNumLayers * sizeof(dfloat));
    zeta = (dfloat*)malloc(maxNumLayers * sizeof(dfloat));
    S = (dfloat*)malloc(maxNumLayers * sizeof(dfloat));

    dfloat r;
    dfloat phase = 0.0;
    dfloat scale = MESH_SCALE; // min distance between each node

    // particleRet.bodyCenter = ParticleCenter();

    // Define the properties of the particle
    r = diameter / 2.0;
    particleRet.bodyCenter.radius = r;
    //translation
    particleRet.bodyCenter.pos.x = center.x;
    particleRet.bodyCenter.pos.y = center.y;
    particleRet.bodyCenter.pos.z = center.z;

    // particleRet.bodyCenter.vel.x = 0.0;
    // particleRet.bodyCenter.vel.y = 0.0;
    // particleRet.bodyCenter.vel.z = 0.0;

    // particleRet.bodyCenter.vel_old.x = 0.0;
    // particleRet.bodyCenter.vel_old.y = 0.0;
    // particleRet.bodyCenter.vel_old.z = 0.0;

    // particleRet.bodyCenter.f.x = 0.0;
    // particleRet.bodyCenter.f.y = 0.0;
    // particleRet.bodyCenter.f.z = 0.0;
    // //rotation
    // particleRet.bodyCenter.theta.x = 0.0;
    // particleRet.bodyCenter.theta.y = 0.0;
    // particleRet.bodyCenter.theta.z = 0.0;

    // particleRet.bodyCenter.w.x = 0.0;
    // particleRet.bodyCenter.w.y = 0.0;
    // particleRet.bodyCenter.w.z = 0.0;

    // particleRet.bodyCenter.w_old.x = 0.0;
    // particleRet.bodyCenter.w_old.y = 0.0;
    // particleRet.bodyCenter.w_old.z = 0.0;

    // particleRet.bodyCenter.M.x = 0.0;
    // particleRet.bodyCenter.M.y = 0.0;
    // particleRet.bodyCenter.M.z = 0.0;

    particleRet.bodyCenter.rho = PARTICLE_DENSITY;
    particleRet.bodyCenter.S = 4.0 * M_PI * r * r;
    
    particleRet.bodyCenter.mass_f = FLUID_DENSITY * 4.0 * M_PI * r * r * r / 3.0; //fluid mass
    particleRet.bodyCenter.mass_p = PARTICLE_DENSITY * 4.0 * M_PI * r * r * r / 3.0; //particle mass
    particleRet.bodyCenter.I.x = 2.0 * particleRet.bodyCenter.mass_p * r * r / 5.0;
    particleRet.bodyCenter.I.y = 2.0 * particleRet.bodyCenter.mass_p * r * r / 5.0;
    particleRet.bodyCenter.I.z = 2.0 * particleRet.bodyCenter.mass_p * r * r / 5.0;

    particleRet.bodyCenter.movable = move;

    nLayer = (unsigned int)(2.0 * sqrt(2) * r / scale + 1.0); //number of layers in the sphere

    particleRet.numNodes = 0;
    for (i = 0; i <= nLayer; i++) {
        // Angle of each layer
        theta[i] = M_PI * ((double)i / (double)nLayer - 0.5); 
        // Determine the number of node per layer
        nNodesLayer[i] = (unsigned int)(1.5 + cos(theta[i]) * nLayer * sqrt(3)); 
        // Total number of nodes on the sphere
        particleRet.numNodes += nNodesLayer[i]; 
        zeta[i] = r * sin(theta[i]); // Height of each layer
    }

    for (i = 0; i < nLayer; i++) {
        // Calculate the distance to the south pole to the mid distance of the layer and previous layer
        S[i] = (zeta[i] + zeta[i + 1]) / 2.0 - zeta[0]; 
    }
    S[nLayer] = 2 * r;
    for (i = 0; i <= nLayer; i++) {
        // Calculate the area of sphere segment since the south pole
        S[i] = 2 * M_PI * r * S[i]; 
    }
    for (i = nLayer; i > 0; i--) {
        // Calculate the area of the layer
        S[i] = S[i] - S[i - 1]; 
    }
    S[0] = S[nLayer];

    particleRet.nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * particleRet.numNodes);
    
    ParticleNode* first_node = &(particleRet.nodes[0]);

    //south node - define all properties
    first_node->pos.x = center.x;
    first_node->pos.y = center.y;
    first_node->pos.z = center.z + r * sin(theta[0]);
    // first_node->pos_ref.x = center.x;
    // first_node->pos_ref.y = center.y;
    // first_node->pos_ref.z = center.z + r * sin(theta[0]);
    // first_node->vel.x = 0.0;
    // first_node->vel.y = 0.0;
    // first_node->vel.z = 0.0;
    // first_node->vel_old.x = 0.0;
    // first_node->vel_old.y = 0.0;
    // first_node->vel_old.z = 0.0;
    // first_node->cf.x = 0.0;
    // first_node->cf.y = 0.0;
    // first_node->cf.z = 0.0;
    first_node->S = S[0];

    int nodeIndex = 0;
    for (i = 1; i < nLayer; i++) {
        if (i % 2 == 1) {
            phase = phase + M_PI / nNodesLayer[i]; // calculate the phase of the segmente to avoid a straight point line
        } else {
            phase = phase + 0;
        }
        for (j = 0; j < nNodesLayer[i]; j++) {
            // determine the properties of each node in the mid layers
            nodeIndex += 1;
            particleRet.nodes[nodeIndex].pos.x = center.x + r * cos(theta[i]) * cos((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            particleRet.nodes[nodeIndex].pos.y = center.y + r * cos(theta[i]) * sin((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            particleRet.nodes[nodeIndex].pos.z = center.z + r * sin(theta[i]);

            // particleRet.nodes[nodeIndex].pos_ref.x = particleRet.nodes[nodeIndex].pos.x;
            // particleRet.nodes[nodeIndex].pos_ref.y = particleRet.nodes[nodeIndex].pos.y;
            // particleRet.nodes[nodeIndex].pos_ref.z = particleRet.nodes[nodeIndex].pos.z;

            // particleRet.nodes[nodeIndex].vel.x = 0.0;
            // particleRet.nodes[nodeIndex].vel.y = 0.0;
            // particleRet.nodes[nodeIndex].vel.z = 0.0;
            // particleRet.nodes[nodeIndex].vel_old.x = 0.0;
            // particleRet.nodes[nodeIndex].vel_old.y = 0.0;
            // particleRet.nodes[nodeIndex].vel_old.z = 0.0;

            // particleRet.nodes[nodeIndex].cf.x = 0.0;
            // particleRet.nodes[nodeIndex].cf.y = 0.0;
            // particleRet.nodes[nodeIndex].cf.z = 0.0;
            // the area of sphere segment is divided by the number of node in the layer, so all nodes have the same area
            particleRet.nodes[nodeIndex].S = S[i] / nNodesLayer[i];
        }
    }

    ParticleNode* last_node = &(particleRet.nodes[nodeIndex+1]);
    //north pole -define all properties

    last_node->pos.x = center.x;
    last_node->pos.y = center.y;
    last_node->pos.z = center.z + r * sin(theta[nLayer]);
    // last_node->pos_ref.x = center.x;
    // last_node->pos_ref.y = center.y;
    // last_node->pos_ref.z = center.z + r * sin(theta[nLayer]);
    // last_node->vel.x = 0.0;
    // last_node->vel.y = 0.0;
    // last_node->vel.z = 0.0;
    // last_node->vel_old.x = 0.0;
    // last_node->vel_old.y = 0.0;
    // last_node->vel_old.z = 0.0;
    // last_node->cf.x = 0.0;
    // last_node->cf.y = 0.0;
    // last_node->cf.z = 0.0;
    last_node->S = S[nLayer];

    unsigned int numberNodes = nodeIndex+1;

    // Coulomb node positions distribution
    if (coulomb != 0) {

        dfloat3 dir;
        dfloat mag;
        dfloat scaleF;
        dfloat3* cForce;
        cForce = (dfloat3*)malloc(numberNodes * sizeof(dfloat3));

        for (i = 0; i < numberNodes; i++) {
            cForce[i].x = 0;
            cForce[i].y = 0;
            cForce[i].z = 0;
        }

        scaleF = 0.001;

        dfloat fx, fy, fz;

        for (unsigned int c = 0; c < coulomb; c++) {
            for (i = 1; i < numberNodes; i++) {
                ParticleNode* node_i = &(particleRet.nodes[i]);

                for (j = 0; j < i; j++) {

                    ParticleNode* node_j = &(particleRet.nodes[j]);

                    if (i != j) {

                        dir.x = node_j->pos.x - node_i->pos.x;
                        dir.y = node_j->pos.y - node_i->pos.y;
                        dir.z = node_j->pos.z - node_i->pos.z;

                        mag = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

                        cForce[i].x -= dir.x / mag;
                        cForce[i].y -= dir.y / mag;
                        cForce[i].z -= dir.z / mag;
                    }
                }
            }
            for (i = 0; i < numberNodes; i++) {
                //move particle
                fx = cForce[i].x / scaleF;
                fy = cForce[i].y / scaleF;
                fz = cForce[i].z / scaleF;
                
                ParticleNode* node_i = &(particleRet.nodes[i]);

                node_i->pos.x += fx;
                node_i->pos.y += fy;
                node_i->pos.z += fz;

                //return to sphere

                mag = sqrt(node_i->pos.x * node_i->pos.x 
                    + node_i->pos.y * node_i->pos.y 
                    + node_i->pos.z * node_i->pos.z);

                node_i->pos.x *= r / mag;
                node_i->pos.y *= r / mag;
                node_i->pos.z *= r / mag;
            }
        } 

        // Area fix
        for (i = 0; i < numberNodes; i++) {
            ParticleNode* node_i = &(particleRet.nodes[i]);

            node_i->S = particleRet.bodyCenter.S / (numberNodes);

            node_i->pos.x += center.x;
            node_i->pos.y += center.y;
            node_i->pos.z += center.z;
        }

        free(cForce);

    } //end coloumb

    free(nNodesLayer);
    free(theta);
    free(zeta);
    free(S);

    return particleRet;
}

#endif // !IBM