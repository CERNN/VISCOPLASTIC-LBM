#include "ibmCollision.h"

#ifdef __IBM_COLLISION_H

__device__ 
int calculateWallIndex(const dfloat3 &n) {
    // Calculate the index based on the normal vector
    return 7 + (1 - (int)n.x) - 2 * (1 - (int)n.y) - 3 * (1 - (int)n.z);
}
__device__ 
int getCollisionIndexByPartnerID(const CollisionData &collisionData, int partnerID, int currentTimeStep) {
    for (int i = 6; i < MAX_ACTIVE_COLLISIONS ; i++) {
        if (collisionData.collisionPartnerIDs[i] == partnerID &&
            currentTimeStep - collisionData.lastCollisionStep[i] <= 1) {
            return i; // Found the collision index for a particle
        }
    }

    // If no match is found, return -1
    return -1;
}
__device__ 
int startCollision(CollisionData &collisionData, int partnerID, bool isWall, const dfloat3 &wallNormal, int currentTimeStep) {
    int index = -1;
    if (isWall) {
        index = calculateWallIndex(wallNormal);
        // Initialize wall collision data
        collisionData.tangentialDisplacements[index] = {0.0, 0.0, 0.0};
        collisionData.lastCollisionStep[index] = currentTimeStep;
    } else {
        for (int i = 6; i < MAX_ACTIVE_COLLISIONS + 6; i++) {
            if (collisionData.lastCollisionStep[i] == -1) { // Check for an unused slot
                collisionData.collisionPartnerIDs[i] = partnerID;
                collisionData.tangentialDisplacements[i] = {0.0, 0.0, 0.0};
                collisionData.lastCollisionStep[i] = currentTimeStep;
                index = i;
                break;
            }
        }
    }

    return index;
}
__device__ 
dfloat3 updateTangentialDisplacement(CollisionData &collisionData, int index, const dfloat3 &displacement, int currentTimeStep) {
        collisionData.tangentialDisplacements[index] = collisionData.tangentialDisplacements[index] + displacement;
        collisionData.lastCollisionStep[index] = currentTimeStep;

    return collisionData.tangentialDisplacements[index];
}
__device__ 
void endCollision(CollisionData &collisionData, int index, int currentTimeStep) {
    // Check if index is valid for wall collisions (0 to 5)
    if (index >= 0 && index < 6) {
        collisionData.lastCollisionStep[index] = -1;
    }
    // Check if index is valid for particle collisions 
    else if (index >= 6 && index < MAX_ACTIVE_COLLISIONS ) {
        if (currentTimeStep - collisionData.lastCollisionStep[index] > 1) {
            collisionData.tangentialDisplacements[index] = {0.0, 0.0, 0.0};
            collisionData.lastCollisionStep[index] = -1; // Indicate the slot is available
            collisionData.collisionPartnerIDs[index] = -1; // Optionally reset partnerID
        }
    }
}
__device__
Wall determineCircularWall(dfloat3 pos_i, dfloat R, dfloat dir){
    Wall tempWall;

    dfloat3 center = dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,0.0);

    tempWall.normal = dfloat3( dir* (pos_i.x-DUCT_CENTER_X), dir * (pos_i.y-DUCT_CENTER_Y),0.0);
    tempWall.normal = vector_normalize(tempWall.normal);

    dfloat3 contactPoint = dfloat3(center - R * tempWall.normal);

    tempWall.distance = fabsf(dot_product(pos_i,tempWall.normal));

    return tempWall;
}
__device__
void checkCollisionWalls(ParticleCenter* pc_i, unsigned int step){
    //printf("checking particle shape for walls \n");
    switch (pc_i->collision.shape) {
        case SPHERE:
            //printf("its a sphere \n");
            checkCollisionWallsSphere(pc_i,step);
            break;
        case CAPSULE:
            //printf("its a capsule \n");
            checkCollisionWallsCapsule(pc_i,step);
            break;
        case ELLIPSOID:
            //printf("its a ellipsoid \n");
            checkCollisionWallsElipsoid(pc_i,step);
            break;
        default:
            // Handle unknown particle types
            break;
    }
    #if defined(EXTERNAL_DUCT_BC) || defined(INTERNAL_DUCT_BC) //NOT VALIDATED YET
        #ifdef EXTERNAL_DUCT_BC
            dfloat dist;
            switch (pc_i->collision.shape) {
            case SPHERE:
                dist = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x,pc_i->pos.y);
                if(dist < EXTERNAL_DUCT_BC_RADIUS + pc_i->radius)
                    sphereWallCollision(pc_i,determineCircularWall(pc_i->pos,EXTERNAL_DUCT_BC_RADIUS,-1),pc_i->radius - dist,step);
                break;
            case CAPSULE:
                dfloat distanceWall1 = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x + pc_i->collision.semiAxis.x,pc_i->pos.y + pc_i->collision.semiAxis.y);
                dfloat distanceWall2 = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x - pc_i->collision.semiAxis.x,pc_i->pos.y - pc_i->collision.semiAxis.y);
                if(distanceWall1 < EXTERNAL_DUCT_BC_RADIUS + pc_i->radius)
                    sphereWallCollision(pc_i,determineCircularWall(pc_i->pos + pc_i->collision.semiAxis,EXTERNAL_DUCT_BC_RADIUS,-1),pc_i->radius - distanceWall1,step);
                if(distanceWall2 < EXTERNAL_DUCT_BC_RADIUS + pc_i->radius)
                    sphereWallCollision(pc_i,determineCircularWall(pc_i->pos + pc_i->collision.semiAxis,EXTERNAL_DUCT_BC_RADIUS,-1),pc_i->radius - distanceWall2,step);
                break;
            case ELLIPSOID:
                //printf("its a ellipsoid \n");
                break;
            default:
                // Handle unknown particle types
                break;
        }
        #endif
        #ifdef INTERNAL_DUCT_BC
            dfloat dist;
            switch (pc_i->collision.shape) {
            case SPHERE:
                dist = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x,pc_i->pos.y);
                if(dist < INTERNAL_DUCT_BC + pc_i->radius)
                    sphereWallCollision(pc_i,determineCircularWall(pc_i->pos,INTERNAL_DUCT_BC,1),pc_i->radius - dist,step);
                break;
            case CAPSULE:
                dfloat distanceWall1 = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x + pc_i->collision.semiAxis.x,pc_i->pos.y + pc_i->collision.semiAxis.y);
                dfloat distanceWall2 = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x - pc_i->collision.semiAxis.x,pc_i->pos.y - pc_i->collision.semiAxis.y);
                if(distanceWall1 < INTERNAL_DUCT_BC + pc_i->radius)
                    sphereWallCollision(pc_i,determineCircularWall(pc_i->pos + pc_i->collision.semiAxis,INTERNAL_DUCT_BC,1),pc_i->radius - distanceWall1,step);
                if(distanceWall2 < INTERNAL_DUCT_BC + pc_i->radius)
                    sphereWallCollision(pc_i,determineCircularWall(pc_i->pos + pc_i->collision.semiAxis,INTERNAL_DUCT_BC,1),pc_i->radius - distanceWall2,step);
                break;
            case ELLIPSOID:
                //printf("its a ellipsoid \n");
                break;
            default:
                // Handle unknown particle types
                break;
            }
        #endif
    #endif
}
__device__
void checkCollisionWallsSphere(ParticleCenter* pc_i,unsigned int step){
    const dfloat3 pos_i = pc_i->pos;
    //printf("checking collision with wall as sphere \n");
    //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
    Wall wallData = wall(dfloat3( 0,0,0),0);
    dfloat distanceWall = 0;
    #ifdef IBM_BC_X_WALL
        wallData = wall(dfloat3( 1,0,0),0);
        distanceWall = dot_product(pos_i,wallData.normal) - wallData.distance;
       // printf("distance to x = 0 is %f \n",distanceWall);
        if (distanceWall < pc_i->radius){
            //printf("colliding with x = 0 with a deformation of %f \n",pc_i->radius - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->radius - distanceWall,step);

        }
        wallData = wall(dfloat3( -1,0,0),(NX - 1));
        //for this case the dot product will be always negative, while the first term will be always better, hence we have to invert and use + signal
        distanceWall = wallData.distance + dot_product(pos_i,wallData.normal);
        //printf("distance to x = 1 is %f \n",distanceWall);
        if (distanceWall < pc_i->radius){
            //printf("colliding with x = 1 with a deformation of %f \n",pc_i->radius - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->radius - distanceWall,step);
        }
    #endif
    #ifdef IBM_BC_Y_WALL
        wallData = wall(dfloat3(0,1,0),0);
        distanceWall = dot_product(pos_i,wallData.normal) - wallData.distance;
        //printf("distance to y = 0 is %f \n",distanceWall);
        if (distanceWall < pc_i->radius){
            //printf("colliding with y = 0 with a deformation of %f \n",pc_i->radius - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->radius - distanceWall,step);
        }
        wallData = wall(dfloat3( 0,-1,0),(NY - 1));
        distanceWall = wallData.distance + dot_product(pos_i,wallData.normal);
        //printf("distance to y = 1 is %f \n",distanceWall);
        if (distanceWall < pc_i->radius){
            //printf("colliding with y = 1 with a deformation of %f \n",pc_i->radius - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->radius - distanceWall,step);
        }
    #endif
    #ifdef IBM_BC_Z_WALL
        wallData = wall(dfloat3(0,0,1),0);
        distanceWall = dot_product(pos_i,wallData.normal) - wallData.distance;
        //printf("distance to z = 0 is %f \n",distanceWall);
        if (distanceWall < pc_i->radius){
            //printf("colliding with z = 0  with a deformation of %f \n",pc_i->radius -distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->radius -  distanceWall,step);
        }
        wallData = wall(dfloat3( 0,0,-1),(NZ_TOTAL - 1));
        distanceWall = wallData.distance + dot_product(pos_i,wallData.normal); 
        //printf("distance to z = 1 is %f \n",distanceWall);
        if (distanceWall < pc_i->radius){
            //printf("colliding with z = 1 with a deformation of %f \n",pc_i->radius - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->radius - distanceWall,step);
        }
    #endif

} 
__device__
void checkCollisionWallsCapsule(ParticleCenter* pc_i,unsigned int step){
    const dfloat3 pos_i = pc_i->pos;
    const dfloat halfLength = vector_length(pc_i->collision.semiAxis);
    const dfloat radius = pc_i->radius;

    // Calculate capsule endpoints using the orientation vector
    dfloat3 endpoint1 = pos_i + pc_i->collision.semiAxis;
    dfloat3 endpoint2 = pos_i - pc_i->collision.semiAxis;

    Wall wallData = wall(dfloat3(0, 0, 0), 0);
    dfloat distanceWall1 = 0;
    dfloat distanceWall2 = 0;

    #ifdef IBM_BC_X_WALL
        wallData = wall(dfloat3(1, 0, 0), 0);
        distanceWall1 = dot_product(endpoint1, wallData.normal) - wallData.distance;
        distanceWall2 = dot_product(endpoint2, wallData.normal) - wallData.distance;

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

        wallData = wall(dfloat3(-1, 0, 0), (NX - 1));
        distanceWall1 = wallData.distance +  dot_product(endpoint1, wallData.normal);
        distanceWall2 = wallData.distance +  dot_product(endpoint2, wallData.normal);


        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

    #endif
    #ifdef IBM_BC_Y_WALL
        wallData = wall(dfloat3( 0,1,0),0);
        distanceWall1 = dot_product(endpoint1, wallData.normal) - wallData.distance;
        distanceWall2 = dot_product(endpoint2, wallData.normal) - wallData.distance;

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

        
        wallData = wall(dfloat3(0, 1, 0), (NY - 1));
        distanceWall1 = wallData.distance +  dot_product(endpoint1, wallData.normal);
        distanceWall2 = wallData.distance +  dot_product(endpoint2, wallData.normal);


        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {;
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

    #endif
    #ifdef IBM_BC_Z_WALL
        wallData = wall(dfloat3( 0,0,1),0);
        distanceWall1 = dot_product(endpoint1, wallData.normal) - wallData.distance;
        distanceWall2 = dot_product(endpoint2, wallData.normal) - wallData.distance;

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }
        
        wallData = wall(dfloat3(0, 0, -1), (NZ - 1));
        distanceWall1 = wallData.distance +  dot_product(endpoint1, wallData.normal);
        distanceWall2 = wallData.distance +  dot_product(endpoint2, wallData.normal);

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }
    #endif
}
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }
    #endif
}
__device__
void checkCollisionWallsElipsoid(ParticleCenter* pc_i, unsigned int step){
    const dfloat3 pos_i = pc_i->pos; // Ellipsoid center position
    const dfloat3 radius = pc_i->collision.semiAxis; // Ellipsoid semi-axes
    const dfloat4 q = pc_i->q_pos; // Ellipsoid orientation quaternion

    Wall wallData;
    dfloat distanceWall = 0;
    dfloat3 intersectionPoint;
    dfloat3 contactPoint2[1];
    dfloat dist = 0;
    
    #ifdef IBM_BC_X_WALL
    wallData = wall(dfloat3(1, 0, 0), 0);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],step);
    }
    wallData = wall(dfloat3(-1, 0, 0), NX-1);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],step);
    }
    #endif

    #ifdef IBM_BC_Y_WALL
    wallData = wall(dfloat3(0, 1, 0), 0);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],step);
    }

    wallData = wall(dfloat3(0, -1, 0), NY-1);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],step);
    }
    #endif
    
    #ifdef IBM_BC_Z_WALL
    wallData = wall(dfloat3(0, 0, 1), 0);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],step);
    }
 
//collision mechanics with walls

__device__
void sphereWallCollision(ParticleCenter* pc_i,Wall wallData,dfloat displacement,int step){

    //particle information
    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat r_i = pc_i->radius;

    const dfloat3 v_i = pc_i->vel;
    const dfloat3 w_i = pc_i->w;

    //wall information        
    dfloat3 wall_speed = dfloat3(0,0,0); // relative velocity vector
    dfloat3 n = wallData.normal;

    //invert collision direction since is from sphere to wall
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;

    //relative velocity
    dfloat3 G = v_i - wall_speed;

    //constants 
    //effective radius and mass
    const dfloat effective_radius = r_i; //wall is r = infinity
    const dfloat effective_mass = m_i; //wall has infinite mass
    //collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(effective_radius));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PW_REST_COEF)  / (sqrt(M_PI*M_PI + log(PW_REST_COEF)*log(PW_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal;
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25); 
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25); 
    dfloat f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force
    dfloat3 G_ct; //relative tangential velocity
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
    
    dfloat mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    dfloat3 t;//tangential velocity vector
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

    //retrive stored displacedment 
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp; //total tangential displacement

    int last_step = pc_i->collision.lastCollisionStep[tang_index];
    if(step - last_step > 1){ //there is no prior collision
        //first need to erase previous collision
        endCollision(pc_i->collision,tang_index,step);
        //now we start the new collision tracking
        startCollision(pc_i->collision,tang_index,true,n,step);
        tang_disp = G_ct;
    }else{//there is already a collision in progress
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,G_ct,step);
    }
    

    //tangential force
    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    //determine if slip or not
    if(  mag > PW_FRICTION_COEF * fabsf(f_n) ){
        f_tang.x = - PW_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PW_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PW_FRICTION_COEF * f_n * t.z;
    }

    //sum the forces
    dfloat3 f_dirs = dfloat3(
        f_normal.x + f_tang.x,
        f_normal.y + f_tang.y,
        f_normal.z + f_tang.z
    );

    //calculate moments
    dfloat3 m_dirs = dfloat3(
        r_i * (n.y*f_tang.z - n.z*f_tang.y),
        r_i * (n.z*f_tang.x - n.x*f_tang.z),
        r_i * (n.x*f_tang.y - n.y*f_tang.x)
    );

    //save data in the particle information
    atomicAdd(&(pc_i->f.x), f_dirs.x);
    atomicAdd(&(pc_i->f.y), f_dirs.y);
    atomicAdd(&(pc_i->f.z), f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs.x);
    atomicAdd(&(pc_i->M.y), m_dirs.y);
    atomicAdd(&(pc_i->M.z), m_dirs.z);
}
__device__
void capsuleWallCollisionCap(ParticleCenter* pc_i,Wall wallData,dfloat displacement,dfloat3 endpoint, int step){

    //particle information
    const dfloat3 pos_i = pc_i->pos; //center position
    const dfloat3 pos_c_i = endpoint; //cap position

    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat r_i = pc_i->radius;

    const dfloat3 v_i = pc_i->vel; //VELOCITY OF THE CENTER OF MASS
    const dfloat3 w_i = pc_i->w;

    //wall information        
    dfloat3 wall_speed = dfloat3(0,0,0); // relative velocity vector
    dfloat3 n = wallData.normal;

    //invert collision direction since is from sphere to wall
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;


    //vector center-> cap
    dfloat3 rr = pos_c_i - pos_i;

    dfloat3 G = v_i - wall_speed;


    //constants 
    //effective radius and mass
    const dfloat effective_radius = r_i; //wall is r = infinity
    const dfloat effective_mass = m_i; //wall has infinite mass
    //collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(effective_radius));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PW_REST_COEF)  / (sqrt(M_PI*M_PI + log(PW_REST_COEF)*log(PW_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal = f_kn * n - DAMPING_NORMAL *  dot_product(G,n) * n * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n = vector_length(f_normal);

    //tangential force
    dfloat3 G_ct = G + r_i * cross_product(w_i,n+rr) - dot_product(G,n)*n;
    dfloat mag = vector_length(G_ct);


    dfloat3 t;//tangential velocity vector
    if (mag != 0){
        //tangential vector
        t = G_ct / mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }

    //retrive stored displacedment 
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp; //total tangential displacement

    int last_step = pc_i->collision.lastCollisionStep[tang_index];
    if(step - last_step > 1){ //there is no prior collision
        //first need to erase previous collision
        endCollision(pc_i->collision,tang_index,step);
        //now we start the new collision tracking
        startCollision(pc_i->collision,tang_index,true,n,step);
        tang_disp = G_ct;
    }else{//there is already a collision in progress
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,G_ct,step);
    }

    //tangential force
    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = vector_length(f_tang);

    //determine if slip or not,
    if(  mag > PW_FRICTION_COEF * fabsf(f_n) ){
        f_tang = - PW_FRICTION_COEF * f_n * t;
    }

    //sum the forces
    dfloat3 f_dirs = f_normal + f_tang;

    //calculate moments
    dfloat3 m_dirs = cross_product((n*r_i) + rr,f_dirs);

    //save date in the particle information
    atomicAdd(&(pc_i->f.x), f_dirs.x);
    atomicAdd(&(pc_i->f.y), f_dirs.y);
    atomicAdd(&(pc_i->f.z), f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs.x);
    atomicAdd(&(pc_i->M.y), m_dirs.y);
    atomicAdd(&(pc_i->M.z), m_dirs.z);
    

}
/* IN PROGRESS
__device__
void capsuleWallCollisionMiddle(ParticleCenter* pc_i,Wall wallData,dfloat displacement1, dfloat displacement2, dfloat3 endpoint1, dfloat3 endpoint2, int step){


    //equation 5.59 jonhson gives
    // delta = (a^2/(2R))*(2*ln(4R/a)-1) as the relation between depth, size and force pe unit of lenght
    // the solution with both R and d different than zero is
    // a = 4*R*exp(0.5*W(-delta*e/(8*R))-0.5) where W is the Lambert W function
    //since delta << 8R, it means that x in W(x) is close to zero and bigger than -1/e hence W_0, will use a Padé aproximation
    dfloat xx = displacement*EULER/(8*r_i);
    dfloat yy = 2*sqrtf(2+2.0*EULER*xx);
    dfloat aa = -1.0 +14.0*yy/45.0 + 301.0*yy*yy/1080.0;
    dfloat bb = 1.0 + 31.0*yy/45.0 + 83.0*yy*yy/1080.0;
    dfloat p2 = aa/bb;
    dfloat a = 4.0*r_i*exp(0.5*p2-0.5);
    //back to equation 5.53
    //a^2 = 4*P*R/piE
    //F = L(a^2/R)*(pi*E/4)
    dfloat f_kn = (length * a * a /effective_radius)*CYLINDER_CYLINDER_STIFFNESS_NORMAL_CONST;
    

}
*/

//sphere functions
__device__
dfloat sphereSphereGap(ParticleCenter*  pc_i, ParticleCenter*  pc_j) {
    dfloat3 p1 = pc_i->pos;
    dfloat3 p2 = pc_j->pos;

    dfloat r1 = pc_i->radius;
    dfloat r2 = pc_j->radius;

    dfloat dist = sqrtf((p1.x - p2.x) * (p1.x - p2.x) +
                    (p1.y - p2.y) * (p1.y - p2.y) +
                    (p1.z - p2.z) * (p1.z - p2.z));
    
    return dist - (r1 + r2);
}
//cylinder functions

// Distance from a point to a line segment (capsule cylinder)
__device__
dfloat point_to_segment_distance(dfloat3 p, dfloat3 segA, dfloat3 segB, dfloat3 closestOnAB[1]) {
    dfloat3 ab = segB - segA;
    dfloat3 ap = p - segA;
    dfloat t = dot_product(ap, ab) / dot_product(ab, ab);
    t = myMax(0, myMin(1, t));  // Clamp t to [0, 1]
    closestOnAB[0] = segA + ab*t;
    return vector_length(p - closestOnAB[0]);
}
// Project a point onto a segment and constrain it within the segment
__device__
dfloat3 constrain_to_segment(dfloat3 point, dfloat3 segStart, dfloat3 segEnd) {
    dfloat3 segDir = segEnd - segStart;
    dfloat segLengthSqr = dot_product(segDir,segDir);
    if (segLengthSqr == 0.0) 
        return segStart;  // The segment is a point

    dfloat t = dot_product((point - segStart), segDir) / segLengthSqr;
    t = clamp01(t);

    return (segStart + (segDir * t));
}
// Main function to compute the closest distance between two segments and return the closest points
__device__
dfloat segment_segment_closest_points(dfloat3 p1, dfloat3 q1, dfloat3 p2, dfloat3 q2, dfloat3 closestOnAB[1], dfloat3 closestOnCD[1]) {

    dfloat3 segDC = (q2 - p2);  // Vector from p2 to q2 (segment [p2, q2])
    dfloat lineDirSqrMag = dot_product(segDC,segDC);  // Square magnitude of segment [p2, q2]

    // Project p1 and q1 onto the plane defined by segment [p2, q2]
    dfloat3 inPlaneA = p1 - ((dot_product(p1-p2,segDC)/lineDirSqrMag)*segDC);
    dfloat3 inPlaneB = q1 - ((dot_product(q1-p2,segDC)/lineDirSqrMag)*segDC);
    dfloat3 inPlaneBA = (inPlaneB - inPlaneA);
    dfloat t = dot_product(p2-inPlaneA,inPlaneBA) / dot_product(inPlaneBA, inPlaneBA);


    if (dot_product(inPlaneBA, inPlaneBA) == 0.0) {
        t = 0.0;  // Handle case where inPlaneA and inPlaneB are the same (segments are parallel)
    }

    // Find the closest point on segment [p1, q1] to the line [p2, q2]
    dfloat3 segABtoLineCD = p1 + clamp01(t)*(q1-p1);

    // Constrain the result to segment [p2, q2]
    closestOnCD[0] = constrain_to_segment(segABtoLineCD, p2, q2);

    // Constrain the closest point on segment [p2, q2] back to segment [p1, q1]
    closestOnAB[0] = constrain_to_segment(closestOnCD[0], p1, q1);


    // Calculate the distance between the closest points on the two segments
    dfloat3 diff = vector_length(closestOnAB[0] - closestOnCD[0]);
    return vector_length(diff);  // Return the distance between the closest points
}
__device__
dfloat segment_segment_closest_points_periodic(dfloat3 p1, dfloat3 q1, dfloat3 p2, dfloat3 q2, dfloat3 closestOnAB[1], dfloat3 closestOnCD[1]){
    dfloat minDist = 1E+37;  // Initialize to a large value
    dfloat3 bestClosestOnAB, bestClosestOnCD;
    int dx = 0;
    int dy = 0;
    int dz = 0;
    #ifdef IBM_BC_X_PERIODIC
    for ( dx = -1; dx <= 1; dx++) {
    #endif
        #ifdef IBM_BC_Y_PERIODIC
        for ( dy = -1; dy <= 1; dy++) {
        #endif
            #ifdef IBM_BC_Z_PERIODIC
            for ( dz = -1; dz <= 1; dz++) {
            #endif
                // Translate segment [p2, q2] by periodic offsets
                dfloat3 p2_translated = p2 + dfloat3(dx * (NX-1), dy * (NY-1), dz * (NZ-1));
                dfloat3 q2_translated = q2 + dfloat3(dx * (NX-1), dy * (NY-1), dz * (NZ-1));

                // Compute closest points between segment [p1, q1] and translated segment [p2_translated, q2_translated]
                dfloat3 tempClosestOnAB, tempClosestOnCD;
                dfloat dist = segment_segment_closest_points(p1, q1, p2_translated, q2_translated, &tempClosestOnAB, &tempClosestOnCD);
                // Update minimum distance and store the best closest points
                if (dist < minDist) {
                    minDist = dist;
                    bestClosestOnAB = tempClosestOnAB;
                    bestClosestOnCD = tempClosestOnCD;
                }

            #ifdef IBM_BC_Z_PERIODIC
            }
            #endif
        #ifdef IBM_BC_Y_PERIODIC
        }
        #endif

    #ifdef IBM_BC_X_PERIODIC
    }
    #endif
    closestOnAB[0] = bestClosestOnAB;
    closestOnCD[0] = bestClosestOnCD;

    return minDist;  // Return the minimum distance between the segments
}
__device__
void capsuleCapsuleCollisionCheck(    unsigned int column,    unsigned int row,ParticleCenter* pc_i, ParticleCenter* pc_j, int step, dfloat3 cylA1, dfloat3 cylA2, dfloat radiusA, dfloat3 cylB1, dfloat3 cylB2, dfloat radiusB) {
    dfloat3 closestOnA[1];
    dfloat3 closestOnB[1];

    if(segment_segment_closest_points_periodic(cylA1, cylA2, cylB1, cylB2, closestOnA, closestOnB) < radiusA + radiusB){
        capsuleCapsuleCollision(column, row,pc_i,pc_j,closestOnA,closestOnB,step);
    }

    return;
}



__device__
void capsuleSphereCollisionCheck(
    unsigned int column,
    unsigned int row,
    ParticleCenter* pc_i, 
    ParticleCenter* pc_j, 
    int step){

    dfloat3 closestOnAB[1];

    if(pc_i->collision.shape == SPHERE){
        if(point_to_segment_distance(pc_i->pos, pc_j->pos + pc_j->collision.semiAxis, pc_j->pos - pc_j->collision.semiAxis,closestOnAB) < pc_i->radius + pc_j->radius)
            capsuleCapsuleCollision(column,row,pc_i,pc_j,&pc_i->pos,closestOnAB,step);
    }else{
        if(point_to_segment_distance(pc_j->pos, pc_i->pos + pc_i->collision.semiAxis, pc_i->pos - pc_i->collision.semiAxis,closestOnAB) < pc_i->radius + pc_j->radius)
            capsuleCapsuleCollision(column,row,pc_i,pc_j,&pc_j->pos,closestOnAB,step);
    }
    

    return;
}

// ------------------------------------------------------------------------ 
// -------------------- COLLISION BETWEEN PARTICLES -----------------------
// ------------------------------------------------------------------------ 

__device__
void checkCollisionBetweenParticles( unsigned int column,unsigned int row,ParticleCenter* pc_i,  ParticleCenter* pc_j,int step){

    int collisionType = 0;

    switch (pc_i->collision.shape) {
        case SPHERE:
            switch (pc_j->collision.shape) {
            case SPHERE:
                //printf("collision between spheres \n");
                if(sphereSphereGap( pc_i, pc_j)<0)
                    sphereSphereCollision(column,row, pc_i, pc_j,step);
                break;
            case CAPSULE:
                capsuleSphereCollisionCheck(column,row,pc_i,pc_j,step);
                break;
            case ELLIPSOID:
                //collision sphere-ellipsoid
                break;
            default:
                // Handle unknown particle types
                break;
            }
            break;
        case CAPSULE:
            switch (pc_j->collision.shape) {
            case SPHERE:
                capsuleSphereCollisionCheck(column,row,pc_i,pc_j,step);
                break;
            case CAPSULE:
                capsuleCapsuleCollisionCheck(column,row,pc_i,pc_j, step, pc_i->pos + pc_i->collision.semiAxis, pc_i->pos - pc_i->collision.semiAxis, pc_i->radius, pc_j->pos + pc_j->collision.semiAxis, pc_j->pos - pc_j->collision.semiAxis, pc_j->radius);
            case ELLIPSOID:
                //collision capsule-ellipsoid
                break;
            default:
                // Handle unknown particle types
                break;
            }
            break;
        case ELLIPSOID:
            switch (pc_j->collision.shape) {
            case SPHERE:
                //collision ellipsoid-sphere
                break;
            case CAPSULE:
                //collision ellipsoid-capsule
                break;
            case ELLIPSOID:
                //collision ellipsoid-ellipsoid
                break;
            default:
                // Handle unknown particle types
                break;
            }
            break;
        default:
            // Handle unknown particle types
            break;
    }
}

// ------------------------------------------------------------------------ 
// -------------------- SPHERE COLLISION ---------- -----------------------
// ------------------------------------------------------------------------ 

__device__
void sphereSphereCollision(unsigned int column,unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step){
    // Particle i info (column)
    const dfloat3 pos_i = pc_i->pos;
    const dfloat r_i = pc_i->radius;
    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat3 v_i = pc_i->vel;
    const dfloat3 w_i = pc_i->w;
   
    // Particle j info (row)
    const dfloat3 pos_j = pc_j->pos;
    const dfloat r_j = pc_j->radius;
    const dfloat m_j = pc_j ->volume * pc_j ->density;
    const dfloat3 v_j = pc_j->vel;
    const dfloat3 w_j = pc_j->w;



    //first check if they will collide
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

    if(mag_dist > r_i+r_j) //they dont collide
        return;

    //but if they collide, we can do some calculations

    //normal collision vector
    const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);

    //normal deformation
    dfloat displacement = r_i + r_j - mag_dist;
    // relative velocity vector
    dfloat3 G = v_i-v_j;

    //HERTZ CONTACT THEORY

    dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal;
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25);
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n;
    f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force
    dfloat3 G_ct;       
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) + r_j*(w_j.y*n.z - w_j.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) + r_j*(w_j.z*n.x - w_j.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) + r_j*(w_j.x*n.y - w_j.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;

    dfloat mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    //calculate tangential vector
    dfloat3 t;
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

    //retrive stored displacedment 
    dfloat3 tang_disp; //total tangential displacement
    int tang_index = getCollisionIndexByPartnerID(pc_i->collision,row,step);
    if(tang_index == -1){ //no previous collision was detected
        tang_index = startCollision(pc_i->collision,row,false,dfloat3(0,0,0),step);
        tang_disp = G_ct;
    }else{
        //check if the collision already exited in the past
        if(step - pc_i->collision.lastCollisionStep[tang_index] > 1){ //already existed but ended
            endCollision(pc_i->collision,tang_index,step); //end current one
            tang_index = startCollision(pc_i->collision,row,false,dfloat3(0,0,0),step);
            tang_disp = G_ct;
        }else{ //collision is still ongoing
            tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,G,step);
        }
    }

    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    //calculate if will slip
    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        f_tang.x = - PP_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PP_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PP_FRICTION_COEF * f_n * t.z;
    }
    //FINAL FORCE RESULTS


    // Force in each direction
    dfloat3 f_dirs = dfloat3(
        f_normal.x + f_tang.x,
        f_normal.y + f_tang.y,
        f_normal.z + f_tang.z
    );
    //Torque in each direction
    dfloat3 m_dirs_i = dfloat3(
        r_i * (n.y*f_tang.z - n.z*f_tang.y),
        r_i * (n.z*f_tang.x - n.x*f_tang.z),
        r_i * (n.x*f_tang.y - n.y*f_tang.x)
    );
    dfloat3 m_dirs_j = dfloat3(
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

// ------------------------------------------------------------------------ 
// -------------------- CAPSULE COLLISIONS -------- -----------------------
// ------------------------------------------------------------------------ 

__device__
void capsuleCapsuleCollision(unsigned int column, unsigned int row, ParticleCenter* pc_i,  ParticleCenter* pc_j, dfloat3 closestOnA[1], dfloat3 closestOnB[1], int step){
    // Particle i info (column)
    const dfloat3 pos_i = closestOnA[0];
    const dfloat3 pos_c_i = pc_i->pos;
    const dfloat r_i = pc_i->radius;
    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat3 v_i = pc_i->vel;
    const dfloat3 w_i = pc_i->w;
   
    // Particle j info (row)
    const dfloat3 pos_j =closestOnB[0];
    const dfloat3 pos_c_j = pc_j->pos;
    const dfloat r_j = pc_j->radius;
    const dfloat m_j = pc_j ->volume * pc_j ->density;
    const dfloat3 v_j = pc_j->vel;
    const dfloat3 w_j = pc_j->w;



    //first check if they will collide
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

    if(mag_dist > r_i+r_j) //they dont collide
        return;

    //but if they collide, we can do some calculations

    //normal collision vector
    const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);

    //normal deformation
    dfloat displacement = r_i + r_j - mag_dist;
    // relative velocity vector
    dfloat3 G = v_i-v_j;

    //HERTZ CONTACT THEORY

    dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal;
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25);
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n;
    f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force
    dfloat3 G_ct;       
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) + r_j*(w_j.y*n.z - w_j.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) + r_j*(w_j.z*n.x - w_j.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) + r_j*(w_j.x*n.y - w_j.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;

    dfloat mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    //calculate tangential vector
    dfloat3 t;
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

    //retrive stored displacedment 
    dfloat3 tang_disp; //total tangential displacement
    int tang_index = getCollisionIndexByPartnerID(pc_i->collision,row,step);
    if(tang_index == -1){ //no previous collision was detected
        tang_index = startCollision(pc_i->collision,row,false,dfloat3(0,0,0),step);
        tang_disp = G_ct;
    }else{
        //check if the collision already exited in the past
        if(step - pc_i->collision.lastCollisionStep[tang_index] > 1){ //already existed but ended
            endCollision(pc_i->collision,tang_index,step); //end current one
            tang_index = startCollision(pc_i->collision,row,false,dfloat3(0,0,0),step);
            tang_disp = G_ct;
        }else{ //collision is still ongoing
            tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,G,step);
        }
    }

    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    //calculate if will slip
    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G,step);
        f_tang.x = - PP_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PP_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PP_FRICTION_COEF * f_n * t.z;
    }

    //FINAL FORCE RESULTS

    //printf("pp  step %d fny %f fnt %f fnz %f \n",step,f_normal.x,f_tang.y, f_normal.z);

    // Force in each direction
    dfloat3 f_dirs = f_normal + f_tang;
    //Torque in each direction
    dfloat3 m_dirs_i = cross_product((pos_i-pos_c_i) + (-n*r_i) ,-f_dirs);
    dfloat3 m_dirs_j = cross_product((pos_j-pos_c_j) + ( n*r_j) , f_dirs);
    
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




// ------------------------------------------------------------------------ 
// -------------------- COLLISION HANDLER --------- -----------------------
// ------------------------------------------------------------------------ 

//collision
__global__
void gpuParticlesCollisionHandler(ParticleCenter particleCenters[NUM_PARTICLES], unsigned int step){
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
    
    const unsigned int row = ceil((-1.0+sqrt((dfloat)1+8*(idx+1)))/2);
    const unsigned int column = idx - ((row-1)*row)/2;

    ParticleCenter* pc_i = &particleCenters[column];
    #ifdef IBM_DEBUG
    printf("collision step %d x: %f \n",step,pc_i->pos.x);
    #endif
    //collision against walls
    if(row == NUM_PARTICLES){
        if(!pc_i->movable)
            return;
        //printf("checking collision with wall  \n");
        checkCollisionWalls(pc_i,step);
    }else{    //Collision between particles
        ParticleCenter* pc_j = &particleCenters[row];
        if(!pc_i->movable && !pc_j->movable)
            return;
        //printf("checking collision with other particle  \n");
        checkCollisionBetweenParticles(column,row,pc_i,pc_j,step);
    }
}

#endif //__IBM_COLLISION_H
/*
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
    

    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx > TOTAL_PCOLLISION_IBM_THREADS)
        return;
    
    const unsigned int row = ceil((-1.0+sqrt((dfloat)1+8*(idx+1)))/2);
    const unsigned int column = idx - ((row-1)*row)/2;

    // Particle from column
    ParticleCenter* pc_i = &particleCenters[column];
    
    #ifdef IBM_DEBUG
    printf("collision step %d x: %f \n",step,pc_i->pos.x);
    #endif

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

    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(effective_radius));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
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

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
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

    tangentialCollisionTracker trackInfo_i[MAX_ACTIVE_COLLISIONS];
    for(int i = 0; i < MAX_ACTIVE_COLLISIONS; i++){
        trackInfo_i[i] = pc_i->tCT[i];
    }
    

    for(int i = 0; i < MAX_ACTIVE_COLLISIONS; i++){
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
                for (int j = i+1; j < MAX_ACTIVE_COLLISIONS; j++){
                    if(trackInfo_i[j].collisionIndex == -8 || j == MAX_ACTIVE_COLLISIONS-1){
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
    for(int i = 0; i < MAX_ACTIVE_COLLISIONS; i++){
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

    

    tangentialCollisionTracker trackInfo_i[MAX_ACTIVE_COLLISIONS];
    tangentialCollisionTracker trackInfo_j[MAX_ACTIVE_COLLISIONS];

    //retrive tracking info
    for(int i = 0; i < MAX_ACTIVE_COLLISIONS; i++){
        trackInfo_i[i] = pc_i->tCT[i];
        trackInfo_j[i] = pc_j->tCT[i];
    }
    
    
    //check tracking for particle i
    for(int i = 0; i < MAX_ACTIVE_COLLISIONS; i++){
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
                for (int j = i+1; j < MAX_ACTIVE_COLLISIONS; j++){
                    if(trackInfo_i[j].collisionIndex == -8 || j == MAX_ACTIVE_COLLISIONS-1){
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
    for(int i = 0; i < MAX_ACTIVE_COLLISIONS; i++){
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
                for (int j = i+1; j < MAX_ACTIVE_COLLISIONS; j++){
                    if(trackInfo_j[j].collisionIndex == -8 || j == MAX_ACTIVE_COLLISIONS-1){
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
    endloop_j:


    // update tracker info
    //TODO THIS NEEDS TO BE ATOMIC
    for(int i = 0; i < MAX_ACTIVE_COLLISIONS; i++){
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


*/
