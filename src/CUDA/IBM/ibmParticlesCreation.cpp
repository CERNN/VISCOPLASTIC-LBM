#include "ibmParticlesCreation.h"

#ifdef IBM

void createParticles(Particle *particles)
{

    dfloat3 *center;
    center = (dfloat3*) malloc(sizeof(dfloat3) * NUM_PARTICLES);

    dfloat3 vel, w;

    unsigned int totalIbmNodes = 0;

    // STAGGERED POSITION
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

   // RANDOM PLACEMENT SPHERES

    /*std::random_device rand_dev;
    std::minstd_rand generator(rand_dev());
    std::uniform_int_distribution<int>  distr(0, RAND_MAX);

    dfloat x_limit_B = IBM_BC_X_0 + PARTICLE_DIAMETER / 2.0;
    dfloat x_limit_E = IBM_BC_X_E - PARTICLE_DIAMETER / 2.0;
    dfloat y_limit_B = IBM_BC_Y_0 + PARTICLE_DIAMETER / 2.0;
    dfloat y_limit_E = IBM_BC_Y_E - PARTICLE_DIAMETER / 2.0;
    dfloat z_limit_B = IBM_BC_Z_0 + PARTICLE_DIAMETER / 2.0;
    dfloat z_limit_E = IBM_BC_Z_E - PARTICLE_DIAMETER / 2.0;

    dfloat px = x_limit_B + (x_limit_E - x_limit_B) * distr(generator) / RAND_MAX;
    dfloat py = y_limit_B + (y_limit_E - y_limit_B) * distr(generator) / RAND_MAX;
    dfloat pz = z_limit_B + (z_limit_E - z_limit_B) * distr(generator) / RAND_MAX;

    dfloat3 center[NUM_PARTICLES];
    center[0].x = px;
    center[0].y = py;
    center[0].z = pz;

    bool next_index;
    dfloat dist;

    int i, j;
    for ( i = 1; i < NUM_PARTICLES; i++) {
        px = x_limit_B + (x_limit_E - x_limit_B) * distr(generator) / RAND_MAX;
        py = y_limit_B + (y_limit_E - y_limit_B) * distr(generator) / RAND_MAX;
        pz = z_limit_B + (z_limit_E - z_limit_B) * distr(generator) / RAND_MAX;

        next_index = false;
        for (j = 0; j < i; j++) {
            dist = sqrt((px - center[j].x) * (px - center[j].x) + (py - center[j].y) * (py - center[j].y) + (pz - center[j].z) * (pz - center[j].z));
            if (dist < PARTICLE_DIAMETER) {
                j = -1;
                px = x_limit_B + (x_limit_E - x_limit_B) * distr(generator) / RAND_MAX;
                py = y_limit_B + (y_limit_E - y_limit_B) * distr(generator) / RAND_MAX;
                pz = z_limit_B + (z_limit_E - z_limit_B) * distr(generator) / RAND_MAX;
            }
        }
        center[i].x = px;
        center[i].y = py;
        center[i].z = pz;
    }*/
    
    center[0].x = (NX-1)/2.0;
    center[0].y = (NY-1)/2.0;
    center[0].z = (NZ_TOTAL-1)/2.0;//10.005 - 100.0*vel.z;   

    vel.x = 0.0;
    vel.y = 0.0; //0.01*sin(angle*M_PI/180.0);
    vel.z = 0.0; //-0.01*cos(angle*M_PI/180.0);
        
    w.x = 0.0;
    w.y = 0.0;
    w.z = 0.0;

    for(int i = 0; i <NUM_PARTICLES ; i++){
        particles[i].makeSpherePolar(PARTICLE_DIAMETER, center[i], MESH_COULOMB, true, PARTICLE_DENSITY, vel, w);
        // particles[1].makeSpherePolar(PARTICLE_DIAMETER, center1 , MESH_COULOMB, false,PARTICLE_DENSITY,vel,w);
    }
}

#endif // !IBM