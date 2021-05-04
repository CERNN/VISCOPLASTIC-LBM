#include "ibmParticlesCreation.h"

#ifdef IBM

void createParticles(Particle particles[NUM_PARTICLES])
{
    dfloat3 bCenter[NUM_PARTICLES];
    unsigned int totalIbmNodes = 0;

    /*
    int id = 0;
    dfloat a,b,c;
    dfloat Lx = NX/(5.0+1.0);
    dfloat Ly = NY/(5.0+1.0);
    for (int k = 0; k<20; k++){
        for (int j = 0; j<5; j++){
            for (int i = 0; i<5; i++){
                
                a = Lx * 0.75 + (dfloat)i*Lx + Lx*0.5*(k%2)         + 0.5*(((dfloat) rand() / (RAND_MAX))-0.5);
                b = Ly * 0.75 + (dfloat)j*Ly + Ly*0.5*((1+(k/2))%2) + 0.5*(((dfloat) rand() / (RAND_MAX))-0.5);
                c = NZ - 20.0 - (dfloat)k*25.0                      + 1.0*(((dfloat) rand() / (RAND_MAX))-0.5);

                bCenter[id].x = a;
                bCenter[id].y = b;
                bCenter[id].z = c;
                id++;
            }
        }
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
    for(int i = 0; i <NUM_PARTICLES ; i++){
        particles[i] = makeSpherePolar(PARTICLE_DIAMETER, center , MESH_COULOMB, true,PARTICLE_DENSITY,vel,w);
    }
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