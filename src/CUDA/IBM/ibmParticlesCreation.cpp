#include "ibmParticlesCreation.h"

#ifdef IBM

void createParticles(Particle particles[NUM_PARTICLES])
{
    dfloat3 bCenter[NUM_PARTICLES];
    unsigned int totalIbmNodes = 0;

    /*
    int id = 0;

    for (int i = NZ-PARTICLE_DIAMETER/2-3 ; i > PARTICLE_DIAMETER/2+3 && id < NUM_PARTICLES; i-=PARTICLE_DIAMETER-3)
    {
        for (int j = PARTICLE_DIAMETER/2+3; j < (NY-PARTICLE_DIAMETER/2-3) && id < NUM_PARTICLES; j+=PARTICLE_DIAMETER+3)
        {
            for (int k = PARTICLE_DIAMETER/2+3; k < (NX-PARTICLE_DIAMETER/2-3) && id < NUM_PARTICLES; k+=PARTICLE_DIAMETER+3)
            {
                bCenter[id].x = k; // 10.0 + (dfloat)i * 25.0 + ((dfloat)rand() / (RAND_MAX));
                bCenter[id].y = j; // 10.0 + (dfloat)j * 25.0 + ((dfloat)rand() / (RAND_MAX));
                bCenter[id].z = i; // 10.0 + (dfloat)k * 25.0 + ((dfloat)rand() / (RAND_MAX));
                particles[id] = makeSpherePolar(PARTICLE_DIAMETER, bCenter[id] , MESH_COULOMB, false);
                id++;
                if(id >= NUM_PARTICLES)
                    break;
            }
            if(id >= NUM_PARTICLES)
                break;
        }
        if(id >= NUM_PARTICLES)
            break;
    }
    */

    
    // Falling sphere
     dfloat3 center,vel, w;
    center.x = 100;
    center.y = 100;
    center.z = 14.995;
    vel.x = 0.0;
    vel.y = 0.0;
    vel.z = -0.1;
    w.x = 0.0;
    w.y = 0.0;
    w.z = 0.0;
    particles[0] = makeSpherePolar(PARTICLE_DIAMETER, center , MESH_COULOMB, true,PARTICLE_DENSITY,vel,w);
   
    /*
    // Fixed sphere
    particles[0] = makeSpherePolar(
        PARTICLE_DIAMETER, 
        dfloat3((NX)/2.0, (NY)/2.0, (NZ)/4.0), 
        MESH_COULOMB, false);
    */
    /*
    // Sphere in couette flow (Neutrally buoyant particle in a shear flow)
    particles[0] = makeSpherePolar(
        PARTICLE_DIAMETER, 
        dfloat3(NX/2, NY/2, NZ/2), 
        MESH_COULOMB, true);
    */
}

#endif // !IBM