#include "ibmCollision.h"

#ifdef __IBM_COLLISION_H


__global__
void gpuParticlesCollision(
    ParticleNodeSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
){

    /* Maps a 1D array to a Floyd triangle, where the last row is for checking
    collision against the wall and the other ones to check collision between 
    particles, with index given by row/column. Example for 7 particles:

    FLOYD TRIANGLE
        c0  c1  c2  c3  c4  c5  c6
    r0  0
    r1  1   2
    r2  3   4   5
    r3  6   7   8   9
    r4  10  11  12  13  14
    r5  15  16  17  18  19  20
    r6  21  22  23  24  25  26  27

    Index 7 is in r3, c1. It will compare p[1] (particle in index 1), from column,
    with p[4], from row (this is because for all rows one is added to its index)

    Index 0 will compare p[0] (column) and p[1] (row)
    Index 13 will compare p[3] (column) and p[5] (row)
    Index 20 will compare p[5] (column) and p[6] (row)

    For the last column, the particles check collision against the wall.
    Index 21 will check p[0] (column) collision against the wall
    Index 27 will check p[6] (column) collision against the wall
    Index 24 will check p[3] (column) collision against the wall

    FROM INDEX TO ROW/COLUMN
    Starting column/row from 1, the n'th row always ends (n)*(n+1)/2+1. So:

    k = (n)*(n+1)/2+1
    n^2 + n - (2k+1) = 0

    (with k=particle index)
    n_row = ceil((-1 + Sqrt(1 + 8(k+1))) / 2)
    n_column = k - n_row * (n_row - 1) / 2
    */

    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx > TOTAL_PCOLLISION_IBM_THREADS)
        return;
    
    const unsigned int row = ceil((-1.0+sqrt((float)1+8*(idx+1)))/2);
    const unsigned int column = idx - ((row-1)*row)/2;

    // Particle from column
    ParticleCenter* pc_i = &particleCenters[column];

    //Collision against walls
    if(row == NUM_PARTICLES){

        if(!pc_i->movable)
            return;

        const dfloat3 pos_i = pc_i->pos;
        const dfloat r_i = pc_i->radius;
        const dfloat min_dist = 2 * r_i;
        
        
        dfloat3 normalVector;
        dfloat displacement;
        dfloat pos_mirror,dist_abs;

        #ifdef HARD_SPHERE
        dfloat3 penetration;
        #endif

        //East x = 0
        pos_mirror = -pos_i.x;
        dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            
            displacement = (2.0 * r_i - dist_abs)/2.0;
            
            normalVector.x = 1.0;
            normalVector.y = 0.0;
            normalVector.z = 0.0;

            #ifdef SOFT_SPHERE
            gpuSoftSphereWallCollision(displacement,normalVector,pc_i);
            #endif
            #ifdef HARD_SPHERE
            penetration = dfloat3(-(min_dist - dist_abs)/2.0,0.0,0.0);
            gpuHardSphereWallCollision(column,penetration,normalVector,pc_i);
            #endif
        }
        //West x = NX-1
        pos_mirror = 2 * (NX - 1) - pos_i.x;
        dist_abs = abs(pos_i.x - pos_mirror);
        if (dist_abs <= min_dist){
            
            
            displacement = (2.0 * r_i - dist_abs)/2.0;;
            
            normalVector.x = -1.0;
            normalVector.y = 0.0;
            normalVector.z = 0.0;
            
            #ifdef SOFT_SPHERE
            gpuSoftSphereWallCollision(displacement,normalVector,pc_i);
            #endif
            #ifdef HARD_SPHERE
            penetration = dfloat3((min_dist - dist_abs)/2,0.0,0.0);
            gpuHardSphereWallCollision(column,penetration,normalVector,pc_i);
            #endif
        }
        //South y = 0
        pos_mirror = - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){            
            
            displacement = (2.0 * r_i - dist_abs)/2.0;
            
            normalVector.x = 0.0;
            normalVector.y = 1.0;
            normalVector.z = 0.0;

            #ifdef SOFT_SPHERE
            gpuSoftSphereWallCollision(displacement,normalVector,pc_i);
            #endif
            #ifdef HARD_SPHERE
            penetration = dfloat3(0.0,-(min_dist - dist_abs)/2.0,0.0);
            gpuHardSphereWallCollision(column,penetration,normalVector,pc_i);
            #endif          
        }
        //North y = NY - 1
        pos_mirror = 2 * (NY - 1) - pos_i.y;
        dist_abs = abs(pos_i.y - pos_mirror);
        if (dist_abs <= min_dist){
            

            displacement = (2.0 * r_i - dist_abs)/2.0;

            normalVector.x = 0.0;
            normalVector.y = -1.0;
            normalVector.z = 0.0;

            #ifdef SOFT_SPHERE
            gpuSoftSphereWallCollision(displacement,normalVector,pc_i);
            #endif
            #ifdef HARD_SPHERE
            penetration = dfloat3(0.0,(min_dist - dist_abs)/2.0,0.0);
            gpuHardSphereWallCollision(column,penetration,normalVector,pc_i);
            #endif   
        }
        //Back z = 0
        pos_mirror = -pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist){
            

            displacement = (2.0 * r_i - dist_abs)/2.0;
            
            normalVector.x = 0.0;
            normalVector.y = 0.0;
            normalVector.z = 1.0;

            #ifdef SOFT_SPHERE
            gpuSoftSphereWallCollision(displacement,normalVector,pc_i);
            #endif
            #ifdef HARD_SPHERE
            penetration = dfloat3(0.0,0.0,-(min_dist - dist_abs)/2.0);
            gpuHardSphereWallCollision(column,penetration,normalVector,pc_i);
            #endif              
        }
        //Front z = NZ - 1
        pos_mirror = 2 * (NZ - 1) - pos_i.z;
        dist_abs = abs(pos_i.z - pos_mirror);
        if (dist_abs <= min_dist) {
            

            displacement = (2.0 * r_i - dist_abs)/2.0;

            normalVector.x = 0.0;
            normalVector.y = 0.0;
            normalVector.z = -1.0;

            #ifdef SOFT_SPHERE
            gpuSoftSphereWallCollision(displacement,normalVector,pc_i);
            #endif
            #ifdef HARD_SPHERE
            penetration = dfloat3(0.0,0.0,(min_dist - dist_abs)/2.0);
            gpuHardSphereWallCollision(column,penetration,normalVector,pc_i);
            #endif  
        }
    }
    //Collision against particles
    else{
        ParticleCenter* pc_j = &particleCenters[row];

        if(!pc_i->movable && !pc_j->movable)
            return;

        // Particle i info (column)
        const dfloat3 pos_i = pc_i->pos;
        const dfloat r_i = pc_i->radius;

        // Particle j info (row)
        const dfloat3 pos_j = pc_j->pos;
        const dfloat r_j = pc_j->radius;

        // Particles position difference
        const dfloat3 diff_pos = dfloat3(
            pos_i.x - pos_j.x,
            pos_i.y - pos_j.y,
            pos_i.z - pos_j.z);

        const dfloat mag_dist = sqrt(
            diff_pos.x*diff_pos.x
            + diff_pos.y*diff_pos.y
            + diff_pos.z*diff_pos.z);
           
        //Check if collision will occur
        if(mag_dist < r_i+r_j){
            #ifdef SOFT_SPHERE
            gpuSoftSphereParticleCollision(r_i + r_j - mag_dist, pc_i, pc_j);
            #endif
            #ifdef HARD_SPHERE
            gpuHarSpheredParticleCollision(column,row,pc_i,pc_j,particlesNodes);
            #endif
        }
    }
}

#ifdef SOFT_SPHERE
__device__
void gpuSoftSphereWallCollision(
    dfloat displacement,
    dfloat3 n,
    ParticleCenter* pc_i
){


    //Variable declaration
    dfloat3 f_normal = dfloat3();
    dfloat3 f_tang = dfloat3();
    dfloat3 t, G, G_ct,tang_disp;
    dfloat f_n,f_kn,mag;
    dfloat3 f_dirs, m_dirs;

    // Particle position
    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat r_i = pc_i->radius;

    const dfloat3 v_i = pc_i->vel;
    const dfloat3 w_i = pc_i->w;
    const dfloat effective_radius = r_i;
    const dfloat effective_mass = m_i;

    //invert collision direction
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;

    // relative velocity vector
    G.x = v_i.x;
    G.y = v_i.y;
    G.z = v_i.z;

    const dfloat STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
    const dfloat damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);


    //normal force
    f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
    f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force       
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;

    mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    if (mag != 0){
        //tangential vector
        t.x = G_ct.x/mag;
        t.y = G_ct.y/mag;
        t.z = G_ct.z/mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }

    //TODO : this is not correct. it should take distance from impact point, not from previous time-step
    tang_disp.x = G_ct.x;
    tang_disp.y = G_ct.y;
    tang_disp.z = G_ct.z;

    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

    if(  mag > FRICTION_COEF * abs(f_n) ){
        f_tang.x = - FRICTION_COEF * f_n * t.x;
        f_tang.y = - FRICTION_COEF * f_n * t.y;
        f_tang.z = - FRICTION_COEF * f_n * t.z;
    }

    // Force in each direction
    f_dirs = dfloat3(
        f_normal.x + f_tang.x,
        f_normal.y + f_tang.y,
        f_normal.z + f_tang.z
    );
    //Torque in each direction
    m_dirs = dfloat3(
        r_i * (n.y*f_tang.z - n.z*f_tang.y),
        r_i * (n.z*f_tang.x - n.x*f_tang.z),
        r_i * (n.x*f_tang.y - n.y*f_tang.x)
    );

    atomicAdd(&(pc_i->f.x), f_dirs.x);
    atomicAdd(&(pc_i->f.y), f_dirs.y);
    atomicAdd(&(pc_i->f.z), f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs.x);
    atomicAdd(&(pc_i->M.y), m_dirs.y);
    atomicAdd(&(pc_i->M.z), m_dirs.z);

}

__device__
void gpuSoftSphereParticleCollision(
    dfloat displacement,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j
){

    dfloat3 f_normal = dfloat3();
    dfloat3 f_tang = dfloat3();
    dfloat3 t, G, G_ct,tang_disp;
    dfloat f_n,mag;

    // Force on particle
    dfloat3 f_dirs = dfloat3();
    dfloat3 m_dirs_i = dfloat3();
    dfloat3 m_dirs_j = dfloat3();

    // Particle i info (column)
    const dfloat3 pos_i = pc_i->pos;
    const dfloat r_i = pc_i->radius;
    const dfloat  m_i = pc_i ->volume * pc_i ->density;
    const dfloat3 v_i = pc_i->vel;
    const dfloat3 w_i = pc_i->w;
   
    // Particle j info (row)
    const dfloat3 pos_j = pc_j->pos;
    const dfloat r_j = pc_j->radius;
    const dfloat  m_j = pc_j ->volume * pc_j ->density;
    const dfloat3 v_j = pc_j->vel;
    const dfloat3 w_j = pc_j->w;
    
    //TODO it was already calculated, it can be passed through the function
    const dfloat3 diff_pos = dfloat3(
        pos_i.x - pos_j.x,
        pos_i.y - pos_j.y,
        pos_i.z - pos_j.z);

    const dfloat mag_dist = sqrt(
        diff_pos.x*diff_pos.x
        + diff_pos.y*diff_pos.y
        + diff_pos.z*diff_pos.z);

    //normal collision vector
    const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);

    // relative velocity vector
    G.x = v_i.x-v_j.x;
    G.y = v_i.y-v_j.y;
    G.z = v_i.z-v_j.z;

    dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (displacement);
    dfloat damping_const = (- 2.0 * log(REST_COEF)  / (sqrt(M_PI*M_PI + log(REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);
    
    
    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(displacement*displacement*displacement);
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x ;
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y ;
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z ;
    f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force       
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) + r_j*(w_j.y*n.z - w_j.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) + r_j*(w_j.z*n.x - w_j.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) + r_j*(w_j.x*n.y - w_j.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;

    mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    if (mag != 0){
        //tangential vector
        t.x = G_ct.x/mag;
        t.y = G_ct.y/mag;
        t.z = G_ct.z/mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }
    
    tang_disp.x = G_ct.x;
    tang_disp.y = G_ct.y;
    tang_disp.z = G_ct.z;

    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x;
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y;
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z;

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    if(  mag > FRICTION_COEF * abs(f_n) ){
        f_tang.x = - FRICTION_COEF * f_n * t.x;
        f_tang.y = - FRICTION_COEF * f_n * t.y;
        f_tang.z = - FRICTION_COEF * f_n * t.z;
    }

    // Force in each direction
    f_dirs = dfloat3(
        f_normal.x + f_tang.x,
        f_normal.y + f_tang.y,
        f_normal.z + f_tang.z
    );
    //Torque in each direction
    m_dirs_i = dfloat3(
        r_i * (n.y*f_tang.z - n.z*f_tang.y),
        r_i * (n.z*f_tang.x - n.x*f_tang.z),
        r_i * (n.x*f_tang.y - n.y*f_tang.x)
    );
    m_dirs_j = dfloat3(
        r_j * (n.y*f_tang.z - n.z*f_tang.y),
        r_j * (n.z*f_tang.x - n.x*f_tang.z),
        r_j * (n.x*f_tang.y - n.y*f_tang.x)
    );

    // Force positive in particle i (column)
    atomicAdd(&(pc_i->f.x), -f_dirs.x);
    atomicAdd(&(pc_i->f.y), -f_dirs.y);
    atomicAdd(&(pc_i->f.z), -f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs_i.x);
    atomicAdd(&(pc_i->M.y), m_dirs_i.y);
    atomicAdd(&(pc_i->M.z), m_dirs_i.z);

    // Force negative in particle j (row)
    atomicAdd(&(pc_j->f.x), f_dirs.x);
    atomicAdd(&(pc_j->f.y), f_dirs.y);
    atomicAdd(&(pc_j->f.z), f_dirs.z);

    atomicAdd(&(pc_j->M.x), m_dirs_j.x); //normal vector takes care of negative sign
    atomicAdd(&(pc_j->M.y), m_dirs_j.y);
    atomicAdd(&(pc_j->M.z), m_dirs_j.z); 
}
#endif //SOFT_SPHERE

#ifdef HARD_SPHERE
__device__
void gpuHardSphereWallCollision(
    dfloat column,
    dfloat3 penetration,
    dfloat3 n,
    ParticleCenter* pc_i
    //ParticleNodeSoA particlesNodes
){
    // Particle i info (column)
    const dfloat  m_i = pc_i ->volume * pc_i ->density;
    const dfloat3  I_i = pc_i ->I;
    const dfloat r_i = pc_i->radius;
    const dfloat3 pos_i = pc_i->pos;
    dfloat3 v_i = pc_i->vel;
    dfloat3 w_i = pc_i->w;
    dfloat dvx_i = 0.0;
    dfloat dvy_i = 0.0;
    dfloat dvz_i = 0.0;
    dfloat dwx_i = 0.0;
    dfloat dwy_i = 0.0;
    dfloat dwz_i = 0.0;

    dfloat ep_x, ep_y, ep_z, ep_mag;

    //velocity mag
    const dfloat vel_mag = sqrt(v_i.x*v_i.x + v_i.y*v_i.y + v_i.z*v_i.z);
    //east
    if(n.x != 0.0){
        if ( (v_i.x / vel_mag) < -n.x*2.0 / (7.0*FRICTION_COEF*(REST_COEF+1)) && FRICTION_COEF != 0){
            dvy_i -= v_i.y - (5.0/7.0)*(v_i.y - 2*r_i*w_i.z/5);
            dvz_i -= v_i.z - (5.0/7.0)*(v_i.z - 2*r_i*w_i.y/5);
    
            dvx_i -= v_i.x + REST_COEF * v_i.x;
    
            dwy_i -= w_i.y - v_i.z/r_i;
            dwx_i += 0;
            dwz_i -= w_i.z + v_i.y/r_i;
    
        } else {
            if(v_i.y == 0 || v_i.z == 0){
                ep_y = 0;
                ep_z = 0;                    
            }else if(v_i.y == 0){
                ep_z = 1;
                ep_y = 0;
            }else if(v_i.z == 0){
                ep_y = 1;
                ep_z = 0;
            }else{
                ep_mag = sqrt(v_i.y*v_i.y + v_i.z*v_i.z);
                ep_y = v_i.y/ep_mag;
                ep_z = v_i.z/ep_mag;
            }
            dvy_i += ep_y*FRICTION_COEF*(REST_COEF+1)*v_i.x;
            dvz_i += ep_z*FRICTION_COEF*(REST_COEF+1)*v_i.x;
    
            dvx_i -= v_i.x + REST_COEF * v_i.x;
    
            dwy_i += - (5.0/(2.0*r_i))*ep_z*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.x);
            dwz_i += + (5.0/(2.0*r_i))*ep_y*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.x);
            dwx_i += 0;
        }

    }
    if(n.x == 0.0 && n.z == 0){
        if ( (v_i.y / vel_mag) < -n.y* 2.0 / (7*FRICTION_COEF*(REST_COEF+1))  && FRICTION_COEF != 0){
            dvx_i -= v_i.x - (5.0/7.0)*(v_i.x - 2*r_i*w_i.z/5);
            dvz_i -= v_i.z - (5.0/7.0)*(v_i.z - 2*r_i*w_i.x/5);

            dvy_i -= v_i.y  + REST_COEF * v_i.y;


            dwx_i -= w_i.x - v_i.z/r_i;
            dwy_i += 0;
            dwz_i -= w_i.z + v_i.x/r_i;

        } else {
            if(v_i.x == 0 || v_i.z == 0){
                ep_x = 0;
                ep_z = 0;                    
            }else if(v_i.x == 0){
                ep_z = 1;
                ep_x = 0;
            }else if(v_i.z == 0){
                ep_x = 1;
                ep_z = 0;
            }else{
                ep_mag = sqrt(v_i.x*v_i.x + v_i.z*v_i.z);
                ep_x = v_i.x/ep_mag;
                ep_z = v_i.z/ep_mag;
            }

            dvx_i += ep_x*FRICTION_COEF*(REST_COEF+1)*v_i.y;
            dvz_i += ep_z*FRICTION_COEF*(REST_COEF+1)*v_i.y;

            dvy_i -= v_i.y + REST_COEF * v_i.y;

            dwx_i += - (5.0/(2.0*r_i))*ep_z*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.y);
            dwz_i += + (5.0/(2.0*r_i))*ep_x*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.y);
            dwy_i += 0;
        }
    }
    if(n.x == 0.0 && n.y == 0){
        if ( (v_i.z / vel_mag) < 2 / (7*FRICTION_COEF*(REST_COEF+1)) && FRICTION_COEF != 0){
            dvx_i -= v_i.x - (5.0/7.0)*(v_i.x - 2*r_i*w_i.y/5);
            dvy_i -= v_i.y - (5.0/7.0)*(v_i.y - 2*r_i*w_i.x/5);

            dvz_i -= v_i.z + REST_COEF * v_i.z;

            dwx_i -= w_i.x - v_i.y/r_i;
            dwz_i += 0;
            dwy_i -= w_i.y + v_i.x/r_i;
        } else {
            if(v_i.x == 0 || v_i.y == 0){
                ep_y = 0; 
                ep_x = 0;
            }else if(v_i.x == 0){
                ep_y = 1;
                ep_x = 0;
            }else if(v_i.y == 0){
                ep_x = 1;
                ep_y = 0;
            }else{
                ep_mag = sqrt(v_i.x*v_i.x + v_i.y*v_i.y);
                ep_x = v_i.x/ep_mag;
                ep_y = v_i.y/ep_mag;
            }

            dvx_i += ep_x*FRICTION_COEF*(REST_COEF+1)*v_i.z;
            dvy_i += ep_y*FRICTION_COEF*(REST_COEF+1)*v_i.z;

            dvz_i -= v_i.z + REST_COEF * v_i.z;

            dwx_i += - (5.0/(2.0*r_i))*ep_y*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.z);
            dwy_i += + (5.0/(2.0*r_i))*ep_x*FRICTION_COEF*(REST_COEF+1)*(-REST_COEF * v_i.z);
            dwz_i += 0;
        }
    }

    // Force positive in particle i (column)
    atomicAdd(&(pc_i->f.x), dvx_i * m_i);
    atomicAdd(&(pc_i->f.y), dvy_i * m_i);
    atomicAdd(&(pc_i->f.z), dvz_i * m_i);

    atomicAdd(&(pc_i->M.x), dwx_i * I_i.x);
    atomicAdd(&(pc_i->M.y), dwy_i * I_i.y);
    atomicAdd(&(pc_i->M.z), dwz_i * I_i.z);

    const dfloat add_dist = 1e-6;
    pc_i->pos.x -= penetration.x*(1.0 + add_dist);
    pc_i->pos.y -= penetration.y*(1.0 + add_dist);
    pc_i->pos.z -= penetration.z*(1.0 + add_dist);

    
    dfloat xIBM,yIBM,zIBM;
    for(int i = 0; i < particlesNodes.numNodes; i++){
        if ( particlesNodes.particleCenterIdx[i] == column){
            xIBM = particlesNodes.pos.x[i];
            yIBM = particlesNodes.pos.y[i];
            zIBM = particlesNodes.pos.z[i];
    
            particlesNodes.vel.x[i] = v_i.x + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
            particlesNodes.vel.y[i] = v_i.y + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
            particlesNodes.vel.z[i] = v_i.z + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));
    
            particlesNodes.vel_old.x[i] = particlesNodes.vel_old.x[i] + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
            particlesNodes.vel_old.y[i] = particlesNodes.vel_old.y[i] + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
            particlesNodes.vel_old.z[i] = particlesNodes.vel_old.z[i] + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));
        }
    }


}

__device__
void gpuHarSpheredParticleCollision(
    dfloat column,
    dfloat row,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j,
    ParticleNodeSoA particlesNodes
){
    dfloat3 n, t, G_0, G_c_0, G_ct_0;
    dfloat mag;

    if(!pc_i->movable && !pc_j->movable)
    return;

    // Particle i info (column)
    const dfloat  m_i = pc_i ->volume * pc_i ->density;
    const dfloat3  I_i = pc_i ->I;
    const dfloat  r_i = pc_i->radius;
    const dfloat3 pos_i = pc_i->pos;
    dfloat3 v_i = pc_i->vel;
    dfloat3 w_i = pc_i->w;

    // Particle i info (column)
    const dfloat  m_j = pc_j ->volume * pc_j ->density;
    const dfloat3  I_j = pc_j ->I;
    const dfloat  r_j = pc_j->radius;
    const dfloat3 pos_j = pc_j->pos;
    dfloat3 v_j = pc_j->vel;
    dfloat3 w_j = pc_j->w;

    // determine normal vector
    n.x = pos_i.x-pos_j.x;
    n.y = pos_i.y-pos_j.y;
    n.z = pos_i.z-pos_j.z;

    mag = n.x*n.x+n.y*n.y+n.z*n.z;
    dfloat const dist_abs = sqrt(mag);

    n.x = n.x/(dist_abs);
    n.y = n.y/(dist_abs);
    n.z = n.z/(dist_abs);

    dfloat px,py,pz;
    px = -n.x*(r_i+r_j-dist_abs);
    py = -n.y*(r_i+r_j-dist_abs);
    pz = -n.z*(r_i+r_j-dist_abs);

    // relative velocity vector
    G_0.x = v_i.x-v_j.x;
    G_0.y = v_i.y-v_j.y;
    G_0.z = v_i.z-v_j.z;

    G_c_0.x = G_0.x + r_i*(w_i.y*n.z-w_i.z*n.y)+r_j*(w_j.y*n.z-w_j.z*n.y);
    G_c_0.y = G_0.y + r_i*(w_i.z*n.x-w_i.x*n.z)+r_j*(w_j.z*n.x-w_j.x*n.z);
    G_c_0.z = G_0.z + r_i*(w_i.x*n.y-w_i.y*n.x)+r_j*(w_j.x*n.y-w_j.y*n.x);

    G_ct_0.x = G_0.x + r_i*(w_i.y*n.z-w_i.z*n.y)+r_j*(w_j.y*n.z-w_j.z*n.y) - (G_c_0.x*n.x+G_c_0.y*n.y+G_c_0.z*n.z)*n.x;
    G_ct_0.y = G_0.y + r_i*(w_i.z*n.x-w_i.x*n.z)+r_j*(w_j.z*n.x-w_j.x*n.z) - (G_c_0.x*n.x+G_c_0.y*n.y+G_c_0.z*n.z)*n.y;
    G_ct_0.z = G_0.z + r_i*(w_i.x*n.y-w_i.y*n.x)+r_j*(w_j.x*n.y-w_j.y*n.x) - (G_c_0.x*n.x+G_c_0.y*n.y+G_c_0.z*n.z)*n.z;

    mag = G_ct_0.x*G_ct_0.x+G_ct_0.y*G_ct_0.y+G_ct_0.z*G_ct_0.z;
    mag=sqrt(mag);

    if (mag != 0){
        //tangential vector
        t.x = G_ct_0.x/mag;
        t.y = G_ct_0.y/mag;
        t.z = G_ct_0.z/mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }
    dfloat nG_0;

    nG_0 = (n.x*G_0.x+n.y*G_0.y+n.z*G_0.z);

    // translational velocity change
    const dfloat dvx_i = - (n.x+FRICTION_COEF*t.x)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j));  
    const dfloat dvy_i = - (n.y+FRICTION_COEF*t.y)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j));
    const dfloat dvz_i = - (n.z+FRICTION_COEF*t.z)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j));

    const dfloat dvx_j = + (n.x+FRICTION_COEF*t.x)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j));
    const dfloat dvy_j = + (n.y+FRICTION_COEF*t.y)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j));
    const dfloat dvz_j = + (n.z+FRICTION_COEF*t.z)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j));

    //rotational velocity change
    const dfloat dwx_i = - (2.5/r_i)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j))*FRICTION_COEF*(n.y*t.z-n.z*t.y);
    const dfloat dwy_i = - (2.5/r_i)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j))*FRICTION_COEF*(n.z*t.x-n.x*t.z);
    const dfloat dwz_i = - (2.5/r_i)*nG_0*(1+REST_COEF)*(m_j/(m_i+m_j))*FRICTION_COEF*(n.x*t.y-n.y*t.x);

    const dfloat dwx_j = - (2.5/r_j)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j))*FRICTION_COEF*(n.y*t.z-n.z*t.y);
    const dfloat dwy_j = - (2.5/r_j)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j))*FRICTION_COEF*(n.z*t.x-n.x*t.z);
    const dfloat dwz_j = - (2.5/r_j)*nG_0*(1+REST_COEF)*(m_i/(m_i+m_j))*FRICTION_COEF*(n.x*t.y-n.y*t.x);

    dfloat add_dist = 1e-3;
    if(pc_i->movable && pc_j->movable){

        atomicAdd(&(pc_i->f.x), (dvx_i*m_i));
        atomicAdd(&(pc_i->f.y), (dvy_i*m_i));
        atomicAdd(&(pc_i->f.z), (dvz_i*m_i));

        atomicAdd(&(pc_i->pos.x), -px*(0.5 + add_dist));
        atomicAdd(&(pc_i->pos.y), -py*(0.5 + add_dist));
        atomicAdd(&(pc_i->pos.z), -pz*(0.5 + add_dist));

        atomicAdd(&(pc_i->M.x), (dwx_i*I_i.x));
        atomicAdd(&(pc_i->M.y), (dwy_i*I_i.y));
        atomicAdd(&(pc_i->M.z), (dwz_i*I_i.z));


        // Force negative in particle j (row)
        atomicAdd(&(pc_j->f.x), dvx_j*m_j);
        atomicAdd(&(pc_j->f.y), dvy_j*m_j);
        atomicAdd(&(pc_j->f.z), dvz_j*m_j);

        atomicAdd(&(pc_j->pos.x), px*(0.5 + add_dist));
        atomicAdd(&(pc_j->pos.y), py*(0.5 + add_dist));
        atomicAdd(&(pc_j->pos.z), pz*(0.5 + add_dist));

        atomicAdd(&(pc_i->M.x), dwx_j* I_j.x);
        atomicAdd(&(pc_i->M.y), dwy_j* I_j.y);
        atomicAdd(&(pc_i->M.z), dwz_j* I_j.z);
    }
    //update node velocities
    
    dfloat xIBM,yIBM,zIBM;
    for(int i = 0; i < particlesNodes.numNodes; i++){
        if ( particlesNodes.particleCenterIdx[i] == column){
            xIBM = particlesNodes.pos.x[i];
            yIBM = particlesNodes.pos.y[i];
            zIBM = particlesNodes.pos.z[i];
    
            particlesNodes.vel.x[i] = v_i.x + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
            particlesNodes.vel.y[i] = v_i.y + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
            particlesNodes.vel.z[i] = v_i.z + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));
    
            particlesNodes.vel_old.x[i] = particlesNodes.vel_old.x[i] + (w_i.y * (zIBM - pos_i.z) - w_i.z * (yIBM - pos_i.y));
            particlesNodes.vel_old.y[i] = particlesNodes.vel_old.y[i] + (w_i.z * (xIBM - pos_i.x) - w_i.x * (zIBM - pos_i.z));
            particlesNodes.vel_old.z[i] = particlesNodes.vel_old.z[i] + (w_i.x * (yIBM - pos_i.y) - w_i.y * (xIBM - pos_i.x));
        }
        if ( particlesNodes.particleCenterIdx[i] == row){
            xIBM = particlesNodes.pos.x[i];
            yIBM = particlesNodes.pos.y[i];
            zIBM = particlesNodes.pos.z[i];
    
            particlesNodes.vel.x[i] = v_j.x + (w_j.y * (zIBM - pos_j.z) - w_j.z * (yIBM - pos_j.y));
            particlesNodes.vel.y[i] = v_j.y + (w_j.z * (xIBM - pos_j.x) - w_j.x * (zIBM - pos_j.z));
            particlesNodes.vel.z[i] = v_j.z + (w_j.x * (yIBM - pos_j.y) - w_j.y * (xIBM - pos_j.x));
    
            particlesNodes.vel_old.x[i] = particlesNodes.vel_old.x[i] + (w_j.y * (zIBM - pos_j.z) - w_j.z * (yIBM - pos_j.y));
            particlesNodes.vel_old.y[i] = particlesNodes.vel_old.y[i] + (w_j.z * (xIBM - pos_j.x) - w_j.x * (zIBM - pos_j.z));
            particlesNodes.vel_old.z[i] = particlesNodes.vel_old.z[i] + (w_j.x * (yIBM - pos_j.y) - w_j.y * (xIBM - pos_j.x));
        }
    } 


}


#endif //HARD_SPERE

#endif //__IBM_COLLISION_H