#include "particle.h"


Particle makeSpherePolar(dfloat diameter, dfloat3 center, unsigned int coulomb, bool move)
{
    Particle result;
    unsigned int maxNumLayers = 5000;
    unsigned int nLayer;
    unsigned int* nNodesLayer;
    dfloat *theta, *zeta, *S;
    unsigned int i, j;

    nNodesLayer = (unsigned int*)malloc(maxNumLayers * sizeof(unsigned int));
    theta = (dfloat*)malloc(maxNumLayers * sizeof(dfloat));
    zeta = (dfloat*)malloc(maxNumLayers * sizeof(dfloat));
    S = (dfloat*)malloc(maxNumLayers * sizeof(dfloat));

    dfloat r;
    dfloat phase = 0.0;
    dfloat scale = MESH_SCALE; // min distance between each node

    //define the properties of the particle
    r = diameter / 2.0;
    result.radius = r;
    result.bodyCenter.radius = r;
    //translation
    result.bodyCenter.pos.x = center.x;
    result.bodyCenter.pos.y = center.y;
    result.bodyCenter.pos.z = center.z;

    result.bodyCenter.vel.x = 0.0;
    result.bodyCenter.vel.y = 0.0;
    result.bodyCenter.vel.z = 0.0;

    result.bodyCenter.vel_old.x = 0.0;
    result.bodyCenter.vel_old.y = 0.0;
    result.bodyCenter.vel_old.z = 0.0;

    result.bodyCenter.f.x = 0.0;
    result.bodyCenter.f.y = 0.0;
    result.bodyCenter.f.z = 0.0;
    //rotation
    result.bodyCenter.theta.x = 0.0;
    result.bodyCenter.theta.y = 0.0;
    result.bodyCenter.theta.z = 0.0;

    result.bodyCenter.w.x = 0.0;
    result.bodyCenter.w.y = 0.0;
    result.bodyCenter.w.z = 0.0;

    result.bodyCenter.w_old.x = 0.0;
    result.bodyCenter.w_old.y = 0.0;
    result.bodyCenter.w_old.z = 0.0;

    result.bodyCenter.M.x = 0.0;
    result.bodyCenter.M.y = 0.0;
    result.bodyCenter.M.z = 0.0;

    result.bodyCenter.rho = particle_density;
    result.bodyCenter.S = 4.0 * M_PI * r * r;

    result.bodyCenter.mass_f = fluid_density * 4.0 * M_PI * r * r * r / 3.0; //fluid mass
    result.bodyCenter.mass_p = particle_density * 4.0 * M_PI * r * r * r / 3.0; //particle mass
    result.bodyCenter.I.x = 2.0 * result.bodyCenter.mass_p * r * r / 5.0;
    result.bodyCenter.I.y = 2.0 * result.bodyCenter.mass_p * r * r / 5.0;
    result.bodyCenter.I.z = 2.0 * result.bodyCenter.mass_p * r * r / 5.0;

    result.bodyCenter.Movable = move;

    nLayer = (unsigned int)(2.0 * sqrt(2) * r / scale + 1.0); //number of layers in the sphere

    result.numNodes = 0;
    for (i = 0; i <= nLayer; i++) {
        theta[i] = M_PI * ((double)i / (double)nLayer - 0.5); // angle of each layer
        nNodesLayer[i] = (unsigned int)(1.5 + cos(theta[i]) * nLayer * sqrt(3)); // determine the number of node per layer
        result.numNodes = result.numNodes + nNodesLayer[i]; //total number of notes on the sphere
        zeta[i] = r * sin(theta[i]); //height of each layer
    }

    for (i = 0; i < nLayer; i++) {
        S[i] = (zeta[i] + zeta[i + 1]) / 2.0 - zeta[0]; // calculate the distance to the south pole to the mid distance of the layer and previous layer
    }
    S[nLayer] = 2 * r;
    for (i = 0; i <= nLayer; i++) {
        S[i] = 2 * M_PI * r * S[i]; // calculate the area of sphere segment since the south pole
    }
    for (i = nLayer; i > 0; i--) {
        S[i] = S[i] - S[i - 1]; //calculate the area of the layer
    }
    S[0] = S[nLayer];

    result.nodes = (ParticleNode*) malloc(result.numNodeParticleNode[result.numNodes];
    
    ParticleNode* first_node = &(results.node[0])

    //south node - define all properties
    first_node->pos.x = center.x;
    first_node->pos.y = center.y;
    first_node->pos.z = center.z + r * sin(theta[0]);
    first_node->pos_ref.x = center.x;
    first_node->pos_ref.y = center.y;
    first_node->pos_ref.z = center.z + r * sin(theta[0]);
    first_node->vel.x = 0.0;
    first_node->vel.y = 0.0;
    first_node->vel.z = 0.0;
    first_node->vel_old.x = 0.0;
    first_node->vel_old.y = 0.0;
    first_node->vel_old.z = 0.0;
    first_node->cf.x = 0.0;
    first_node->cf.y = 0.0;
    first_node->cf.z = 0.0;
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
            nodeIndex = nodeIndex + 1;
            result.nodes[nodeIndex].pos.x = center.x + r * cos(theta[i]) * cos((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            result.nodes[nodeIndex].pos.y = center.y + r * cos(theta[i]) * sin((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            result.nodes[nodeIndex].pos.z = center.z + r * sin(theta[i]);

            result.nodes[nodeIndex].pos_ref.x = result.nodes[nodeIndex].pos.x;
            result.nodes[nodeIndex].pos_ref.y = result.nodes[nodeIndex].pos.y;
            result.nodes[nodeIndex].pos_ref.z = result.nodes[nodeIndex].pos.z;

            result.nodes[nodeIndex].vel.x = 0.0;
            result.nodes[nodeIndex].vel.y = 0.0;
            result.nodes[nodeIndex].vel.z = 0.0;
            result.nodes[nodeIndex].vel_old.x = 0.0;
            result.nodes[nodeIndex].vel_old.y = 0.0;
            result.nodes[nodeIndex].vel_old.z = 0.0;

            result.nodes[nodeIndex].cf.x = 0.0;
            result.nodes[nodeIndex].cf.y = 0.0;
            result.nodes[nodeIndex].cf.z = 0.0;
            // the area of sphere segment is divided by the number of node in the layer, so all nodes have the same area
            result.nodes[nodeIndex].S = S[i] / nNodesLayer[i];
        }
    }

    ParticleNode* last_node = &(results.node[nodeIndex+1])
    //north pole -define all properties

    last_node->pos.x = center.x;
    last_node->pos.y = center.y;
    last_node->pos.z = center.z + r * sin(theta[nLayer]);
    last_node->pos_ref.x = center.x;
    last_node->pos_ref.y = center.y;
    last_node->pos_ref.z = center.z + r * sin(theta[nLayer]);
    last_node->vel.x = 0.0;
    last_node->vel.y = 0.0;
    last_node->vel.z = 0.0;
    last_node->vel_old.x = 0.0;
    last_node->vel_old.y = 0.0;
    last_node->vel_old.z = 0.0;
    last_node->cf.x = 0.0;
    last_node->cf.y = 0.0;
    last_node->cf.z = 0.0;
    last_node->S = S[nLayer];

    unsigned int numberNodes = nodeIndex+1;

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
            for (i = 0; i < numberNodes; i++) {
                ParticleNode* node_i = &(result.nodes[i])

                for (j = 0; j < i; j++) {

                    ParticleNode* node_j = &(result.nodes[j])

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
                
                ParticleNode* node_i = &(result.nodes[i])

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
        } //end of cycle

        //area fix
        for (i = 0; i < numberNodes; i++) {
            ParticleNode* node_i = &(result.nodes[i])

            node_i->S = result.bodyCenter.S / (numberNodes);

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

    return result;
}