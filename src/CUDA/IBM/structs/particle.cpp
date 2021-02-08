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

Particle makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move,
    dfloat3 vel, dfloat3 w)
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
    particleRet.pCenter.pos = center;

    // Particle velocity
    particleRet.pCenter.vel = vel;
    particleRet.pCenter.vel_old = vel;

    // Particle rotation
    particleRet.pCenter.w = w;
    particleRet.pCenter.w_old = w;

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
    

    particleRet.nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * particleRet.numNodes);

    ParticleNode* first_node = &(particleRet.nodes[0]);

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
            particleRet.nodes[nodeIndex].pos.x = r * cos(theta[i]) * cos((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            particleRet.nodes[nodeIndex].pos.y = r * cos(theta[i]) * sin((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            particleRet.nodes[nodeIndex].pos.z = r * sin(theta[i]);

            // The area of sphere segment is divided by the number of node
            // in the layer, so all nodes have the same area in the layer
            particleRet.nodes[nodeIndex].S = S[i] / nNodesLayer[i];

            // Define node velocity
            particleRet.nodes[nodeIndex].vel.x = vel.x + w.y * particleRet.nodes[nodeIndex].pos.z - w.z * particleRet.nodes[nodeIndex].pos.y;
            particleRet.nodes[nodeIndex].vel.y = vel.y + w.z * particleRet.nodes[nodeIndex].pos.x - w.x * particleRet.nodes[nodeIndex].pos.z;
            particleRet.nodes[nodeIndex].vel.z = vel.z + w.x * particleRet.nodes[nodeIndex].pos.y - w.y * particleRet.nodes[nodeIndex].pos.x;

            particleRet.nodes[nodeIndex].vel_old.x = vel.x + w.y * particleRet.nodes[nodeIndex].pos.z - w.z * particleRet.nodes[nodeIndex].pos.y;
            particleRet.nodes[nodeIndex].vel_old.y = vel.y + w.z * particleRet.nodes[nodeIndex].pos.x - w.x * particleRet.nodes[nodeIndex].pos.z;
            particleRet.nodes[nodeIndex].vel_old.z = vel.z + w.x * particleRet.nodes[nodeIndex].pos.y - w.y * particleRet.nodes[nodeIndex].pos.x;
            

            // Add one node
            nodeIndex++;
        }
    }

    // North pole -define all properties
    ParticleNode* last_node = &(particleRet.nodes[particleRet.numNodes-1]);
    
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

    unsigned int numNodes = particleRet.numNodes;

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
        }

        // Free coulomb force
        free(cForce);

        dfloat dA =  particleRet.pCenter.S/particleRet.numNodes;
        for(int i = 0; i < numNodes; i++){
            particleRet.nodes[i].S = dA;
        }
    }
    ParticleNode* node_i;
    for (int i = 0; i < numNodes; i++) {
        node_i = &(particleRet.nodes[i]);
        node_i->pos.x += center.x;
        node_i->pos.y += center.y;
        node_i->pos.z += center.z;
    }

    // Free allocated variables
    free(nNodesLayer);
    free(theta);
    free(zeta);
    free(S);

    // Update old position value
    particleRet.pCenter.pos_old = particleRet.pCenter.pos;

    return particleRet;
}


Particle makeOpenCylinder(dfloat diameter, dfloat3 baseOneCenter, dfloat3 baseTwoCenter, bool pattern)
{
    // Particle to be returned
    Particle particleRet;

    // Define the properties of the cylinder
    dfloat r = diameter / 2.0;
    dfloat x = baseTwoCenter.x - baseOneCenter.x;
    dfloat y = baseTwoCenter.y - baseOneCenter.y;
    dfloat z = baseTwoCenter.z - baseOneCenter.z;
    dfloat length = sqrt (x*x +y*y+z*z);
    dfloat volume = r*r*M_PI*length;

    particleRet.pCenter.radius = r;
    // Particle volume
    particleRet.pCenter.volume = volume;
    // Particle area
    particleRet.pCenter.S = 2.0*M_PI*r*length;

    // Particle center position
    particleRet.pCenter.pos.x = (baseOneCenter.x + baseTwoCenter.x)/2;
    particleRet.pCenter.pos.y = (baseOneCenter.y + baseTwoCenter.y)/2;
    particleRet.pCenter.pos.z = (baseOneCenter.z + baseTwoCenter.z)/2;

    particleRet.pCenter.movable = false;


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
    particleRet.numNodes = nLayer * nNodesLayer;

    particleRet.nodes = (ParticleNode*) malloc(sizeof(ParticleNode) * particleRet.numNodes);

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
            particleRet.nodes[nodeIndex].pos.x = centerLayer[i].x + r * cos(angle) * v1.x + r * sin(angle) * v2.x; 
            particleRet.nodes[nodeIndex].pos.y = centerLayer[i].y + r * cos(angle) * v1.y + r * sin(angle) * v2.y;
            particleRet.nodes[nodeIndex].pos.z = centerLayer[i].z + r * cos(angle) * v1.z + r * sin(angle) * v2.z;

            particleRet.nodes[i].S = particleRet.pCenter.S/((dfloat)nNodesLayer * nLayer);

            nodeIndex++;
        }
    }

    return particleRet;

}



#endif // !IBM