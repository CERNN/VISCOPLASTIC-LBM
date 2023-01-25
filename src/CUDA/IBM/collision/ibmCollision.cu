#include "ibmCollision.h"

#ifdef __IBM_COLLISION_H


__global__
void gpuParticlesCollision(
    ParticleCenter particleCenters[NUM_PARTICLES],
    unsigned int step
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


        #ifdef IBM_BC_X_WALL
            //East x = 0
            pos_mirror = -pos_i.x;
            dist_abs = abs(pos_i.x - pos_mirror);
            #if defined LUBRICATION_FORCE
                // 2.0 is because is mirrored distance
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = 1.0;
                    normalVector.y = 0.0;
                    normalVector.z = 0.0;

                    if (dist_abs <= min_dist){
                    
                        displacement = (2.0 * r_i - dist_abs)/2.0;

                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist  - dist_abs,normalVector,pc_i);
                    }
                }
            #else //!LUBRICATION_FORCE
                if (dist_abs <= min_dist){
                    
                    displacement = (2.0 * r_i - dist_abs)/2.0;
                    
                    normalVector.x = 1.0;
                    normalVector.y = 0.0;
                    normalVector.z = 0.0;

                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                }
            #endif //LUBRICATION_FORCE

            //West x = NX-1
            pos_mirror = 2 * (NX - 1) - pos_i.x;
            dist_abs = abs(pos_i.x - pos_mirror);
            #if defined LUBRICATION_FORCE
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = -1.0;
                    normalVector.y = 0.0;
                    normalVector.z = 0.0;

                    if (dist_abs <= min_dist){
                    
                        displacement = (2.0 * r_i - dist_abs)/2.0;
                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);

                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist - dist_abs,normalVector,pc_i);
                    }
                }
            #else //!LUBRICATION_FORCE
                if (dist_abs <= min_dist){
                    
                    displacement = (2.0 * r_i - dist_abs)/2.0;;
                    
                    normalVector.x = -1.0;
                    normalVector.y = 0.0;
                    normalVector.z = 0.0;
                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                }
            #endif //LUBRICATION_FORCE
        #endif //IBM_BC_X_WALL

        #ifdef IBM_BC_Y_WALL
            //South y = 0
            pos_mirror = - pos_i.y;
            dist_abs = abs(pos_i.y - pos_mirror);
            #if defined LUBRICATION_FORCE
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = 0.0;
                    normalVector.y = 1.0;
                    normalVector.z = 0.0;
                    
                    if (dist_abs <= min_dist){            
                    
                        displacement = (2.0 * r_i - dist_abs)/2.0;
                
                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);   

                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist  - dist_abs,normalVector,pc_i);
                    }
                }
            #else //!LUBRICATION_FORCE
                if (dist_abs <= min_dist){            
                    
                    displacement = (2.0 * r_i - dist_abs)/2.0;
                    
                    normalVector.x = 0.0;
                    normalVector.y = 1.0;
                    normalVector.z = 0.0;

                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);        
                }
            #endif //LUBRICATION_FORCE

            //North y = NY - 1
            pos_mirror = 2 * (NY - 1) - pos_i.y;
            dist_abs = abs(pos_i.y - pos_mirror);
            #if defined LUBRICATION_FORCE
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = 0.0;
                    normalVector.y = -1.0;
                    normalVector.z = 0.0;

                    if (dist_abs <= min_dist){
                    
                        displacement = (2.0 * r_i - dist_abs)/2.0;

                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step); 
                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist - dist_abs,normalVector,pc_i);
                    }
                }
            #else //!LUBRICATION_FORCE
                if (dist_abs <= min_dist){
                    
                    displacement = (2.0 * r_i - dist_abs)/2.0;

                    normalVector.x = 0.0;
                    normalVector.y = -1.0;
                    normalVector.z = 0.0;

                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step); 
                }
            #endif //LUBRICATION_FORCE
        #endif //IBM_BC_Y_WALL

        #ifdef IBM_BC_Z_WALL
            //Back z = 0
            pos_mirror = -pos_i.z;
            dist_abs = abs(pos_i.z - pos_mirror);
            #if defined LUBRICATION_FORCE
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = 0.0;
                    normalVector.y = 0.0;
                    normalVector.z = 1.0;
                    
                    if (dist_abs <= min_dist){

                        displacement = (2.0 * r_i - dist_abs)/2.0;
                        
                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);            
                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist - dist_abs,normalVector,pc_i);
                    }
                }
            #else //!LUBRICATION_FORCE
                if (dist_abs <= min_dist){
                    
                    displacement = (2.0 * r_i - dist_abs)/2.0;
                    
                    normalVector.x = 0.0;
                    normalVector.y = 0.0;
                    normalVector.z = 1.0;

                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);           
                }
            #endif

            
            //Front z = NZ - 1
            pos_mirror = 2 * (NZ_TOTAL - 1) - pos_i.z;
            dist_abs = abs(pos_i.z - pos_mirror);


            #if defined LUBRICATION_FORCE
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = 0.0;
                    normalVector.y = 0.0;
                    normalVector.z = -1.0;

                    if (dist_abs <= min_dist) {
                    
                        displacement = (2.0 * r_i - dist_abs)/2.0;
        
                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist - dist_abs,normalVector,pc_i);
                    }
                }
            #else // !LUBRICATION_FORCE
                if (dist_abs <= min_dist) {
                    
                    displacement = (2.0 * r_i - dist_abs)/2.0;

                    normalVector.x = 0.0;
                    normalVector.y = 0.0;
                    normalVector.z = -1.0;

                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                }
            #endif // LUBRICATION_FORCE
        #endif //IBM_BC_Z_WALL
        //   --------------- DUCT BOUNDARY CONDITIONS -----------------------
        #if defined(EXTERNAL_DUCT_BC) || defined(INTERNAL_DUCT_BC)
            // "boundaryConditionsSchemes/interpolatedBounceBack.cu"
            dfloat xCenter = DUCT_CENTER_X;
            dfloat yCenter = DUCT_CENTER_Y; 




            dfloat pos_r_i = sqrt((pos_i.x-xCenter)*(pos_i.x-xCenter) + (pos_i.y-yCenter)*(pos_i.y-yCenter));

        #endif

        #ifdef EXTERNAL_DUCT_BC
            dfloat R = EXTERNAL_DUCT_BC_RADIUS;
            // pos_mirror = R + pos_r_i;
            pos_mirror = 2 * R - pos_r_i;
            dist_abs = abs(pos_r_i - pos_mirror);
            


            #if defined LUBRICATION_FORCE
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = - (pos_i.x-xCenter)/pos_r_i;
                    normalVector.y = - (pos_i.y-yCenter)/pos_r_i;
                    normalVector.z = 0.0;


                    if (dist_abs <= min_dist) {
                        displacement = (2.0 * r_i - dist_abs)/2.0;
                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist - dist_abs,normalVector,pc_i);
                    }
                }
            #else // !LUBRICATION_FORCE
                if (dist_abs <= min_dist) {
                    displacement = (2.0 * r_i - dist_abs)/2.0;

                    normalVector.x = - (pos_i.x-xCenter)/pos_r_i;
                    normalVector.y = - (pos_i.y-yCenter)/pos_r_i;
                    normalVector.z = 0.0;
                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                }
            #endif // LUBRICATION_FORCE
        #endif //EXTERNAL_DUCT_BC



        #ifdef INTERNAL_DUCT_BC
            dfloat r = INTERNAL_DUCT_BC_RADIUS;
            pos_mirror = 2 *r - pos_r_i;
            dist_abs = abs(pos_r_i - pos_mirror);
            #if defined LUBRICATION_FORCE
                if (dist_abs <= min_dist + 2.0*MAX_LUBRICATION_DISTANCE) {
                    normalVector.x = 0.0;
                    normalVector.y = 0.0;
                    normalVector.z = 0.0;


                    if (dist_abs <= min_dist) {
                        displacement = (2.0 * r_i - dist_abs)/2.0;
                        gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                    }else if(dist_abs > min_dist + MIN_LUBRICATION_DISTANCE) {
                        gpuLubricationWall(min_dist - dist_abs,normalVector,pc_i);
                    }
                }
            #else // !LUBRICATION_FORCE
                if (dist_abs <= min_dist) {
                    displacement = (2.0 * r_i - dist_abs)/2.0;

                    normalVector.x = 0.0;
                    normalVector.y = 0.0;
                    normalVector.z = 0.0;
                    gpuSoftSphereWallCollision(displacement,normalVector,pc_i,step);
                }
            #endif // LUBRICATION_FORCE
        #endif //INTERNAL_DUCT_BC

        // end of wall collision if

    }

    //Collision between particles
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
            #ifdef IBM_BC_X_WALL
                pos_i.x - pos_j.x
            #endif //IBM_BC_X_WALL
            #ifdef IBM_BC_X_PERIODIC 
            abs(pos_i.x - pos_j.x) > ((IBM_BC_X_E - IBM_BC_X_0) / 2.0) ? 
            (pos_i.x < pos_j.x ?
                (pos_i.x + (IBM_BC_X_E - IBM_BC_X_0) - pos_j.x)
                : 
                (pos_i.x - (IBM_BC_X_E - IBM_BC_X_0) - pos_j.x)
            )
            : pos_i.x - pos_j.x
            #endif //IBM_BC_X_PERIODIC
            ,
            #ifdef IBM_BC_Y_WALL
                pos_i.y - pos_j.y
            #endif //IBM_BC_Y_WALL
            #ifdef IBM_BC_Y_PERIODIC
            abs(pos_i.y - pos_j.y) > ((IBM_BC_Y_E - IBM_BC_Y_0) / 2.0) ? 
            (pos_i.y < pos_j.y ?
                (pos_i.y + (IBM_BC_Y_E - IBM_BC_Y_0) - pos_j.y)
                : 
                (pos_i.y - (IBM_BC_Y_E - IBM_BC_Y_0) - pos_j.y)
            )
            : pos_i.y - pos_j.y
            #endif //IBM_BC_Y_PERIODIC
            ,
            #ifdef IBM_BC_Z_WALL
                pos_i.z - pos_j.z
            #endif //IBM_BC_Z_WALL
            #ifdef IBM_BC_Z_PERIODIC
                abs(pos_i.z - pos_j.z) > ((IBM_BC_Z_E - IBM_BC_Z_0) / 2.0) ? 
                (pos_i.z < pos_j.z ?
                    (pos_i.z + (IBM_BC_Z_E - IBM_BC_Z_0) - pos_j.z)
                    : 
                    (pos_i.z - (IBM_BC_Z_E - IBM_BC_Z_0) - pos_j.z)
                )
                : pos_i.z - pos_j.z
            #endif //IBM_BC_Z_PERIODIC
        );



        const dfloat mag_dist = sqrt(
            diff_pos.x*diff_pos.x
            + diff_pos.y*diff_pos.y
            + diff_pos.z*diff_pos.z);

        //printf("i: %f , j: %f , dx: %f \n",pos_i.z,pos_j.z,diff_pos.z);
        #if defined LUBRICATION_FORCE
            //check if lubrication will occur
            if(mag_dist < r_i+r_j + MAX_LUBRICATION_DISTANCE){

                //Check if collision will occur
                if(mag_dist < r_i+r_j){
                    gpuSoftSphereParticleCollision(r_i + r_j - mag_dist, column,row,pc_i, pc_j,step);
                }else{ // mag_dist - r_i+r_j > MAX_LUBRICATION_DISTANCE 
                    gpuLubricationParticle(r_i + r_j - mag_dist, pc_i, pc_j);
                }
            }
        #else
            //Check if collision will occur
            if(mag_dist < r_i+r_j){
                gpuSoftSphereParticleCollision(r_i + r_j - mag_dist,column,row, pc_i, pc_j,step);
            }
        #endif
    }
}

__device__
void gpuSoftSphereWallCollision(
    dfloat displacement,
    dfloat3 n,
    ParticleCenter* pc_i,
    unsigned int step
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
    const dfloat3 pos_i = pc_i->pos;
    const dfloat effective_radius = r_i;
    const dfloat effective_mass = m_i;

    int trackerId;
    trackerId = gpuTangentialDisplacementTrackerWall(n,pc_i,step);
    dfloat3 tangDisplacement;
    tangDisplacement = pc_i->tCT[trackerId].tang_length;

    //invert collision direction
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;

    // relative velocity vector
    dfloat3 wall_speed = dfloat3(0,0,0);


    dfloat xNode = pos_i.x - DUCT_CENTER_X;
    dfloat yNode = pos_i.y - DUCT_CENTER_Y;
    
    dfloat rr =  sqrt(xNode*xNode+yNode*yNode);
    dfloat c = xNode / (rr);
    dfloat s = yNode / (rr);


    #ifdef EXTERNAL_DUCT_BC 
        wall_speed.x = - OUTER_ROTATION * OUTER_RADIUS * s;
        wall_speed.y =   OUTER_ROTATION * OUTER_RADIUS * c;
        #ifdef BC_RHEOMETER
            wall_speed.x *= (pos_i.z/NZ_TOTAL);
            wall_speed.y *= (pos_i.z/NZ_TOTAL);
        #endif
        wall_speed.z = OUTER_VELOCITY;
        #ifdef INTERNAL_DUCT_BC 
            //TODO: needs a better detection system to define if is inner cilynder or outer.
            if (rr/(OUTER_RADIUS-INNER_RADIUS)< 0.5){
                wall_speed.x = - INNER_ROTATION * INNER_RADIUS * s;
                wall_speed.y =   INNER_ROTATION * INNER_RADIUS * c;
                wall_speed.z = INNER_VELOCITY;
            }
        #endif //INTERNAL_DUCT_BC
    #endif //EXTERNAL_DUCT_BC




    G.x = v_i.x - wall_speed.x;
    G.y = v_i.y - wall_speed.y;
    G.z = v_i.z - wall_speed.z;

    const dfloat STIFFNESS_NORMAL = PW_STIFFNESS_NORMAL_CONST * sqrt(abs(effective_radius));
    const dfloat STIFFNESS_TANGENTIAL = PW_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PW_REST_COEF)  / (sqrt(M_PI*M_PI + log(PW_REST_COEF)*log(PW_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);


    //normal force
    f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25); 
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25); 
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

    //printf("\n -- G_ct.x : %f -- G_ct.y : %f -- G_ct.z : %f", G_ct.x, G_ct.y, G_ct.z);

    //TODO : Still need validation
    tang_disp.x = G_ct.x + tangDisplacement.x;
    tang_disp.y = G_ct.y + tangDisplacement.y;
    tang_disp.z = G_ct.z + tangDisplacement.z;

    pc_i->tCT[trackerId].tang_length = tang_disp;

    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    //printf("\n -- f_ct.x : %f -- f_ct.y : %f -- f_ct.z : %f", f_tang.x, f_tang.y, f_tang.z);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    if(  mag > PW_FRICTION_COEF * abs(f_n) ){
        //printf("\n entered if, mag: %f > %f", mag,FRICTION_COEF * abs(f_n));
        f_tang.x = - PW_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PW_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PW_FRICTION_COEF * f_n * t.z;
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
    unsigned int column,
    unsigned int row,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j, 
    unsigned int step
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

    //tangential collision info
    int trackerId;
    trackerId = gpuTangentialDisplacementTrackerParticle(column,row,pc_i,pc_j,step);
    dfloat3 tangDisplacement=dfloat3(0,0,0);
    //tangDisplacement = pc_i->tCT[trackerId].tang_length;
    
    //TODO it was already calculated, it can be passed through the function
    const dfloat3 diff_pos = dfloat3(
        #ifdef IBM_BC_X_WALL
            pos_i.x - pos_j.x
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC 
        abs(pos_i.x - pos_j.x) > ((IBM_BC_X_E - IBM_BC_X_0) / 2.0) ? 
        (pos_i.x < pos_j.x ?
            (pos_i.x + (IBM_BC_X_E - IBM_BC_X_0) - pos_j.x)
            : 
            (pos_i.x - (IBM_BC_X_E - IBM_BC_X_0) - pos_j.x)
        )
        : pos_i.x - pos_j.x
        #endif //IBM_BC_X_PERIODIC
        ,
        #ifdef IBM_BC_Y_WALL
            pos_i.y - pos_j.y
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
        abs(pos_i.y - pos_j.y) > ((IBM_BC_Y_E - IBM_BC_Y_0) / 2.0) ? 
        (pos_i.y < pos_j.y ?
            (pos_i.y + (IBM_BC_Y_E - IBM_BC_Y_0) - pos_j.y)
            : 
            (pos_i.y - (IBM_BC_Y_E - IBM_BC_Y_0) - pos_j.y)
        )
        : pos_i.y - pos_j.y
        #endif //IBM_BC_Y_PERIODIC
        ,
        #ifdef IBM_BC_Z_WALL
            pos_i.z - pos_j.z
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            abs(pos_i.z - pos_j.z) > ((IBM_BC_Z_E - IBM_BC_Z_0) / 2.0) ? 
            (pos_i.z < pos_j.z ?
                (pos_i.z + (IBM_BC_Z_E - IBM_BC_Z_0) - pos_j.z)
                : 
                (pos_i.z - (IBM_BC_Z_E - IBM_BC_Z_0) - pos_j.z)
            )
            : pos_i.z - pos_j.z
        #endif //IBM_BC_Z_PERIODIC
    );

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

    //HERTZ CONTACT THEORY

    dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = PP_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = PP_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);
    
    
    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25);
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25);
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
    
    //TODO : Still need validation
    tang_disp.x = G_ct.x + tangDisplacement.x;
    tang_disp.y = G_ct.y + tangDisplacement.y;
    tang_disp.z = G_ct.z + tangDisplacement.z;

    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        f_tang.x = - PP_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PP_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PP_FRICTION_COEF * f_n * t.z;
    }

    //FINAL FORCE RESULTS


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


__device__
int gpuTangentialDisplacementTrackerWall(
    dfloat3 n,
    ParticleCenter* pc_i,
    unsigned int step
){

   

    int wallIndex = (1-(int)n.x) - 2*(1-(int)n.y) - 3 *(1-(int)n.z);
    int trackerId = 0;

    tangentialCollisionTracker trackInfo_i[trackerCollisionSize];
    for(int i = 0; i < trackerCollisionSize; i++){
        trackInfo_i[i] = pc_i->tCT[i];
    }
    

    for(int i = 0; i < trackerCollisionSize; i++){
        if (trackInfo_i[i].collisionIndex == -8){
            trackerId = i;
            trackInfo_i[trackerId].collisionIndex = wallIndex;
            trackInfo_i[trackerId].tang_length = dfloat3(0,0,0);
            trackInfo_i[trackerId].lastCollisionStep = step;

            goto endloop;

        }else{
            if(step > trackInfo_i[i].lastCollisionStep + 1){

                //not a valid tracker
                // check if next is valid, if i
                for (int j = i+1; j < trackerCollisionSize; j++){
                    if(trackInfo_i[j].collisionIndex == -8 || j == trackerCollisionSize-1){
                        trackInfo_i[j].collisionIndex = -8;

                        trackerId = j;
                        trackInfo_i[trackerId].collisionIndex = wallIndex;
                        trackInfo_i[trackerId].tang_length = dfloat3(0,0,0);
                        trackInfo_i[trackerId].lastCollisionStep = step;

                        goto endloop;

                    }else{
                        trackInfo_i[j-1].collisionIndex =    trackInfo_i[j].collisionIndex;
                        trackInfo_i[j-1].tang_length =       trackInfo_i[j].tang_length;
                        trackInfo_i[j-1].lastCollisionStep = trackInfo_i[j].lastCollisionStep;
                        
                        trackInfo_i[j].collisionIndex = -8;
                    }                        
                }
            }else{ // still valid tracker
                if (trackInfo_i[i].collisionIndex == wallIndex) {
                    trackerId = i;
                    // update last collision step
                    trackInfo_i[trackerId].lastCollisionStep = step;

                    goto endloop;

                } 
            }
        }
    }
    endloop:
    // update tracker info
    for(int i = 0; i < trackerCollisionSize; i++){
        pc_i->tCT[i] = trackInfo_i[i];
    }

    return trackerId;
}


__device__
int gpuTangentialDisplacementTrackerParticle(
    unsigned int  column,
    unsigned int  row,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j,
    unsigned int step
){
    

    unsigned int pc_i_index = column;
    unsigned int pc_j_index = row;
    

    int trackerId = 0;

    

    tangentialCollisionTracker trackInfo_i[trackerCollisionSize];
    tangentialCollisionTracker trackInfo_j[trackerCollisionSize];

    //retrive tracking info
    for(int i = 0; i < trackerCollisionSize; i++){
        trackInfo_i[i] = pc_i->tCT[i];
        trackInfo_j[i] = pc_j->tCT[i];
    }
    
    
    //check tracking for particle i
    for(int i = 0; i < trackerCollisionSize; i++){
        if (trackInfo_i[i].collisionIndex == -8){
            trackerId = i;
            trackInfo_i[trackerId].collisionIndex = pc_j_index;
            trackInfo_i[trackerId].tang_length = dfloat3(0,0,0);
            trackInfo_i[trackerId].lastCollisionStep = step;

            goto endloop_i;
        }else{
            if(step > trackInfo_i[i].lastCollisionStep + 1){

                //not a valid tracker
                // check if next is valid, if i
                for (int j = i+1; j < trackerCollisionSize; j++){
                    if(trackInfo_i[j].collisionIndex == -8 || j == trackerCollisionSize-1){
                        trackInfo_i[j].collisionIndex = -8;

                        trackerId = j;
                        trackInfo_i[trackerId].collisionIndex = pc_j_index;
                        trackInfo_i[trackerId].tang_length = dfloat3(0,0,0);
                        trackInfo_i[trackerId].lastCollisionStep = step;

                        goto endloop_i;
                    }else{
                        trackInfo_i[j-1].collisionIndex =    trackInfo_i[j].collisionIndex;
                        trackInfo_i[j-1].tang_length =       trackInfo_i[j].tang_length;
                        trackInfo_i[j-1].lastCollisionStep = trackInfo_i[j].lastCollisionStep;
                        
                        trackInfo_i[j].collisionIndex = -8;
                    }    
                }
            }else{ // still valid tracker
                if (trackInfo_i[i].collisionIndex == pc_j_index) {
                    trackerId = i;
                    // update last collision step
                    trackInfo_i[trackerId].lastCollisionStep = step;
                    goto endloop_i;
                } 
            }
        }
    }
    endloop_i:
    /* Its not necessary, removing also prevents race condition
    //TODO: Evaluate if is really necessary track information in both particles.
    //check tracking for particle j
    for(int i = 0; i < trackerCollisionSize; i++){
        if (trackInfo_j[i].collisionIndex == -8){
            trackerId = i;
            trackInfo_j[trackerId].collisionIndex = pc_i_index;
            trackInfo_j[trackerId].tang_length = dfloat3(0,0,0);
            trackInfo_j[trackerId].lastCollisionStep = step;

            goto endloop_j;
        }else{
            if(step > trackInfo_i[i].lastCollisionStep + 1){

                //not a valid tracker
                // check if next is valid, if i
                for (int j = i+1; j < trackerCollisionSize; j++){
                    if(trackInfo_j[j].collisionIndex == -8 || j == trackerCollisionSize-1){
                        trackInfo_j[j].collisionIndex = -8;

                        trackerId = j;
                        trackInfo_j[trackerId].collisionIndex = pc_i_index;
                        trackInfo_j[trackerId].tang_length = dfloat3(0,0,0);
                        trackInfo_j[trackerId].lastCollisionStep = step;

                        goto endloop_j;
                    }else{
                        trackInfo_j[j-1].collisionIndex =    trackInfo_j[j].collisionIndex;
                        trackInfo_j[j-1].tang_length =       trackInfo_j[j].tang_length;
                        trackInfo_j[j-1].lastCollisionStep = trackInfo_j[j].lastCollisionStep;
                        
                        trackInfo_j[j].collisionIndex = -8;
                    }    
                }
            }else{ // still valid tracker
                if (trackInfo_j[i].collisionIndex == pc_i_index) {
                    trackerId = i;
                    // update last collision step
                    trackInfo_j[trackerId].lastCollisionStep = step;
                    goto endloop_j;
                } 
            }
        }
    }
    endloop_j:*/


    // update tracker info
    //TODO THIS NEEDS TO BE ATOMIC
    for(int i = 0; i < trackerCollisionSize; i++){
        pc_i->tCT[i] = trackInfo_i[i];
        //pc_j->tCT[i] = trackInfo_j[i];
    }

    return trackerId;
}


#if defined LUBRICATION_FORCE
__device__
void gpuLubricationWall(
    dfloat gap,
    dfloat3 wallNormalVector,
    ParticleCenter* pc_i
){
    const dfloat3 vel_i = pc_i->vel;
    const dfloat r_i = pc_i->radius;

    //const dfloat r_squared = 2*r_i + (MAX_LUBRICATION_DISTANCE-gap)*(MAX_LUBRICATION_DISTANCE-gap);
    const dfloat inv_gap = -1.0/gap ;//(1.0/gap - 1.0/(gap +r_squared/r_i ));
    const dfloat MU = (TAU-0.5)/3.0;

    atomicAdd(&(pc_i->f.x), -1.5*M_PI*r_i*r_i*inv_gap * MU * vel_i.x*wallNormalVector.x);
    atomicAdd(&(pc_i->f.y), -1.5*M_PI*r_i*r_i*inv_gap * MU * vel_i.x*wallNormalVector.y);
    atomicAdd(&(pc_i->f.z), -1.5*M_PI*r_i*r_i*inv_gap * MU * vel_i.x*wallNormalVector.z);

}

__device__ 
void gpuLubricationParticle(
    dfloat gap,
    ParticleCenter* pc_i,
    ParticleCenter* pc_j
){
    const dfloat3 pos_i = pc_i->pos;
    const dfloat3 v_i = pc_i->vel;
    const dfloat r_i = pc_i->radius;
   
    const dfloat3 pos_j = pc_j->pos;
    const dfloat3 v_j = pc_j->vel;
    const dfloat r_j = pc_j->radius;

    dfloat3 G;
    // relative velocity vector
    G.x = v_i.x-v_j.x;
    G.y = v_i.y-v_j.y;
    G.z = v_i.z-v_j.z;

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

    const dfloat r_eq = (r_i*r_j)/(r_i+r_j);

    //const dfloat r_squared = 2.0*r_eq + (MAX_LUBRICATION_DISTANCE-gap)*(MAX_LUBRICATION_DISTANCE-gap);
    const dfloat inv_gap = -1.0/gap; //(1.0/gap - 1.0/(gap +r_squared/r_eq ));

    const dfloat MU = (TAU-0.5)/3.0;
    dfloat3 FL;

    FL.x = 1.5*M_PI*r_eq*r_eq*inv_gap * MU * G.x*n.x;
    FL.y = 1.5*M_PI*r_eq*r_eq*inv_gap * MU * G.y*n.y;
    FL.z = 1.5*M_PI*r_eq*r_eq*inv_gap * MU * G.z*n.z;


    atomicAdd(&(pc_i->f.x), -FL.x);
    atomicAdd(&(pc_i->f.y), -FL.y);
    atomicAdd(&(pc_i->f.z), -FL.z);

    atomicAdd(&(pc_j->f.x), FL.x);
    atomicAdd(&(pc_j->f.y), FL.y);
    atomicAdd(&(pc_j->f.z), FL.z);
}
#endif //LUBRICATION_FORCE



#endif //__IBM_COLLISION_H