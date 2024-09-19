#include "ibmCollision.h"

#ifdef __IBM_COLLISION_H

__device__ 
int calculateWallIndex(const dfloat3 &n) {
    // Calculate the index based on the normal vector
    return 7 + (1 - (int)n.x) - 2 * (1 - (int)n.y) - 3 * (1 - (int)n.z);
    /*
    n.x n.y n.z index
    1   0   0   2
    -1  0   0   4
    0   1   0   5
    0   -1  0   1
    0   0   1   6
    0   0   -1  0
    0   0   0   3 -> external duct, since normal will be reduced to 0,0,0 when convert to integer
    */
}
__device__ 
int getCollisionIndexByPartnerID(const CollisionData &collisionData, int partnerID, int currentTimeStep) {
    for (int i = 7; i < MAX_ACTIVE_COLLISIONS ; i++) {
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
        for (int i = 7; i < MAX_ACTIVE_COLLISIONS; i++) {
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
    // Check if index is valid for wall collisions (0 to 6)
    if (index >= 0 && index <= 6) {
        collisionData.lastCollisionStep[index] = -1;
    }
    // Check if index is valid for particle collisions 
    else if (index >= 7 && index < MAX_ACTIVE_COLLISIONS ) {
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

    tempWall.normal = dfloat3( dir * (pos_i.x-DUCT_CENTER_X), dir * (pos_i.y-DUCT_CENTER_Y),0.0);
    tempWall.normal = vector_normalize(tempWall.normal);

    dfloat3 contactPoint = dfloat3(center - R * tempWall.normal);

    tempWall.distance = vector_length(contactPoint - pos_i);

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
        Wall wallData;
        dfloat dist;
        dfloat distanceWall1;
        dfloat distanceWall2;
        dfloat3 endpoint1;
        dfloat3 endpoint2;
        dfloat3 closestOnA[1];
        dfloat3 closestOnB[1];
        dfloat cr[1];
        
        #ifdef EXTERNAL_DUCT_BC
            switch (pc_i->collision.shape) {
            case SPHERE:
                 dist = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x,pc_i->pos.y);
                if(EXTERNAL_DUCT_BC_RADIUS - dist < pc_i->radius){
                    wallData = determineCircularWall(pc_i->pos,EXTERNAL_DUCT_BC_RADIUS,-1);
                    sphereWallCollision(pc_i,wallData,EXTERNAL_DUCT_BC_RADIUS - (pc_i->radius + dist),step);
                }
                break;
            case CAPSULE:
                endpoint1 = pc_i->collision.semiAxis;
                distanceWall1 = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,endpoint1.x,endpoint1.y);
                if(EXTERNAL_DUCT_BC_RADIUS - distanceWall1 < pc_i->radius){
                    wallData = determineCircularWall(endpoint1,EXTERNAL_DUCT_BC_RADIUS,-1);
                    capsuleWallCollisionCap(pc_i,wallData,EXTERNAL_DUCT_BC_RADIUS - (pc_i->radius + distanceWall1),endpoint1,step);
                }

                endpoint2 = pc_i->collision.semiAxis2;
                distanceWall2 = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,endpoint2.x,endpoint2.y);
                if(EXTERNAL_DUCT_BC_RADIUS - distanceWall2 < pc_i->radius){
                    wallData = determineCircularWall(endpoint2,EXTERNAL_DUCT_BC_RADIUS,-1);
                    capsuleWallCollisionCap(pc_i,wallData,EXTERNAL_DUCT_BC_RADIUS - (pc_i->radius + distanceWall2),endpoint2,step);
                }
                break;
            case ELLIPSOID:
                dist = ellipsoidSegmentCollisionDistance(pc_i,dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,-EXTERNAL_DUCT_BC_RADIUS),dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL+EXTERNAL_DUCT_BC_RADIUS),EXTERNAL_DUCT_BC_RADIUS,closestOnA,closestOnB,cr,-1,step);
                //printf("step %d dist %f \n", step, dist);
                if (dist <0){
                    ellipsoidCylinderCollision(pc_i,-dist, closestOnB[0],cr,dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,-EXTERNAL_DUCT_BC_RADIUS),dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL+EXTERNAL_DUCT_BC_RADIUS),EXTERNAL_DUCT_BC_RADIUS,-1,step);
                }
                break;
            default:
                // Handle unknown particle types
                break;
        }
        #endif
        #ifdef INTERNAL_DUCT_BC
            switch (pc_i->collision.shape) {
            case SPHERE:
                dist = distPoints2D(DUCT_CENTER_X,DUCT_CENTER_Y,pc_i->pos.x,pc_i->pos.y);
                if(dist < INTERNAL_DUCT_BC_RADIUS + pc_i->radius){
                    wallData = determineCircularWall(pc_i->pos,INTERNAL_DUCT_BC_RADIUS,1);
                    sphereWallCollision(pc_i,wallData,INTERNAL_DUCT_BC_RADIUS + pc_i->radius - dist,step);
                }                    
                break;
            case CAPSULE:
                if(segment_segment_closest_points_periodic(pc_i->collision.semiAxis,pc_i->collision.semiAxis2, dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,-INTERNAL_DUCT_BC_RADIUS), dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL + INTERNAL_DUCT_BC_RADIUS), closestOnA, closestOnB) <  pc_i->radius + INTERNAL_DUCT_BC_RADIUS){
                    capsuleInnerDuctCollision(pc_i,closestOnA,closestOnB,step);
                }           
                break;
            case ELLIPSOID:
                dist = ellipsoidSegmentCollisionDistance(pc_i,dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,-INTERNAL_DUCT_BC_RADIUS),dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL+INTERNAL_DUCT_BC_RADIUS),INTERNAL_DUCT_BC_RADIUS,closestOnA,closestOnB,cr,1,step);
                //printf("step %d dist %f \n", step, dist);
                if (dist <0){
                    ellipsoidCylinderCollision(pc_i,-dist, closestOnB[0],cr,dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,-INTERNAL_DUCT_BC_RADIUS),dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,NZ_TOTAL+INTERNAL_DUCT_BC_RADIUS),INTERNAL_DUCT_BC_RADIUS,1,step);
                }
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
    const dfloat halfLength = vector_length(pc_i->collision.semiAxis);
    const dfloat radius = pc_i->radius;

    // Calculate capsule endpoints using the orientation vector
    dfloat3 endpoint1 = pc_i->collision.semiAxis;
    dfloat3 endpoint2 = pc_i->collision.semiAxis2;

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
__device__
void checkCollisionWallsElipsoid(ParticleCenter* pc_i, unsigned int step){

    Wall wallData;
    dfloat distanceWall = 0;
    dfloat3 intersectionPoint;
    dfloat3 contactPoint2[1];
    
    dfloat cr[1];

    #ifdef IBM_BC_X_WALL
    wallData = wall(dfloat3(1, 0, 0), 0);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,cr,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],cr,step);
    }
    wallData = wall(dfloat3(-1, 0, 0), NX-1);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,cr,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],cr,step);
    }
    #endif

    #ifdef IBM_BC_Y_WALL
    wallData = wall(dfloat3(0, 1, 0), 0);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,cr,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],cr,step);
    }

    wallData = wall(dfloat3(0, -1, 0), NY-1);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,cr,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],cr,step);
    }
    #endif
    
    #ifdef IBM_BC_Z_WALL
    wallData = wall(dfloat3(0, 0, 1), 0);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,cr,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],cr,step);
    }
    
    wallData = wall(dfloat3(0, 0, -1), NZ-1);
    distanceWall = ellipsoidWallCollisionDistance(pc_i,wallData,contactPoint2,cr,step);
    if (distanceWall < 0) {
        ellipsoidWallCollision(pc_i,wallData,-distanceWall,contactPoint2[0],cr,step);
    }
    #endif
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
         tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
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
         tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
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

__device__
void capsuleInnerDuctCollision(ParticleCenter* pc_i, dfloat3 closestOnA[1], dfloat3 closestOnB[1], int step){
    // Particle i info (column)
    const dfloat3 pos_i = closestOnA[0];
    const dfloat3 pos_c_i = pc_i->pos;
    const dfloat r_i = pc_i->radius;
    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat3 v_i = pc_i->vel;
    const dfloat3 w_i = pc_i->w;
   
    // Particle j info (row)
    const dfloat3 pos_j = closestOnB[0];
    dfloat r_j = 0;
    #ifdef INTERNAL_DUCT_BC_RADIUS //some quick fix
    r_j = INTERNAL_DUCT_BC_RADIUS;
    #endif
    const dfloat3 v_j = 0.0; //WALL VELOCITY
    const dfloat3 w_j = 0.0;



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

    dfloat effective_radius = 1.0/r_i;
    dfloat effective_mass = 1.0/m_i;

    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
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
    //TODO: FIXING IT ON LAST SLOT
    dfloat3 tang_disp; //total tangential displacement
    int tang_index = getCollisionIndexByPartnerID(pc_i->collision,MAX_ACTIVE_COLLISIONS-1,step);
    if(tang_index == -1){ //no previous collision was detected
        tang_index = startCollision(pc_i->collision,MAX_ACTIVE_COLLISIONS-1,false,dfloat3(0,0,0),step);
        tang_disp = G_ct;
    }else{
        //check if the collision already exited in the past
        if(step - pc_i->collision.lastCollisionStep[tang_index] > 1){ //already existed but ended
            endCollision(pc_i->collision,tang_index,step); //end current one
            tang_index = startCollision(pc_i->collision,MAX_ACTIVE_COLLISIONS-1,false,dfloat3(0,0,0),step);
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
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
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
    
    // Force positive in particle i (column)
    atomicAdd(&(pc_i->f.x), -f_dirs.x);
    atomicAdd(&(pc_i->f.y), -f_dirs.y);
    atomicAdd(&(pc_i->f.z), -f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs_i.x);
    atomicAdd(&(pc_i->M.y), m_dirs_i.y);
    atomicAdd(&(pc_i->M.z), m_dirs_i.z);
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

__device__
void ellipsoidWallCollision(ParticleCenter* pc_i,Wall wallData,dfloat displacement,dfloat3 endpoint, dfloat cr[1], int step){

    //particle information
    const dfloat3 pos_i = pc_i->pos; //center position
    const dfloat3 pos_c_i = endpoint; //contact point + n * radius of contact

    const dfloat m_i = pc_i ->volume * pc_i ->density;
    
    const dfloat r_i = pc_i->radius; //TODO: find a way to calculate the correct radius of contact


    const dfloat3 v_i = pc_i->vel; //VELOCITY OF THE CENTER OF MASS
    const dfloat3 w_i = pc_i->w;

    //wall information        
    dfloat3 wall_speed = dfloat3(0,0,0); // relative velocity vector
    dfloat3 n = wallData.normal;

    //invert collision direction since is from sphere to wall
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;


    //vector center-> contact 
    dfloat3 rr = pos_c_i - pos_i;

    dfloat3 G = v_i - wall_speed;


    //const dfloat effective_radius = r_i; //wall is r = infinity
    const dfloat effective_radius = cr[0]; //wall is r = infinity
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
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
    }

    //sum the forces
    dfloat3 f_dirs = f_normal + f_tang;

    //calculate moments
    dfloat3 m_dirs = cross_product(rr,f_dirs);

    //save date in the particle information
    atomicAdd(&(pc_i->f.x), f_dirs.x);
    atomicAdd(&(pc_i->f.y), f_dirs.y);
    atomicAdd(&(pc_i->f.z), f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs.x);
    atomicAdd(&(pc_i->M.y), m_dirs.y);
    atomicAdd(&(pc_i->M.z), m_dirs.z);
}


__device__
void ellipsoidCylinderCollision(ParticleCenter* pc_i,dfloat displacement,dfloat3 endpoint, dfloat cr[1], dfloat3 P1, dfloat3 P2, dfloat cRadius, int cyDir, int step){
    
    const dfloat3 pos_i = pc_i->pos; //center position
    const dfloat3 pos_c_i = endpoint; //contact point + n * radius of contact
    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat r_i = cr[0];

    const dfloat3 v_i = pc_i->vel; //VELOCITY OF THE CENTER OF MASS
    const dfloat3 w_i = pc_i->w;

    //wall information        
    dfloat3 wall_speed = dfloat3(0,0,0); // relative velocity vector

    //printf("endpoint %f %f %f \n",endpoint.x,endpoint.y,endpoint.z);
    //printf("P1 %f %f %f \n",P1.x,P1.y,P1.z);
    //printf("P2 %f %f %f \n",P2.x,P2.y,P2.z);
    //printf("cRadius %f \n",cRadius);
    //printf("cyDir %d \n",cyDir);

    dfloat3 proj = segmentProjection(endpoint,P1,P2,cRadius,cyDir);
    dfloat3 dir = pc_i->pos - proj;
    dfloat3 n = vector_normalize(proj - endpoint);

    //printf("proj %f %f %f \n",proj.x,proj.y,proj.z);
    //printf("dir %f %f %f \n",dir.x,dir.y,dir.z);
    //printf("n %f %f %f \n",n.x,n.y,n.z);

    //invert collision direction since is from sphere to wall
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;

    //printf("n %f %f %f \n",n.x,n.y,n.z);
    //vector center-> contact 
    dfloat3 rri = pos_c_i - pos_i;

    //printf("rri %f %f %f \n",rri.x,rri.y,rri.z);

    dfloat3 G = v_i - wall_speed;

    //const dfloat effective_radius = r_i; //wall is r = infinity
    const dfloat effective_radius = abs(1.0/((r_i +cRadius)/(r_i*cRadius))); //TODO FIX ABS, negative values will occur when the collision contact radius is bigger than the cylinder radius when dir = -1; But that would cause a double point collision
    const dfloat effective_mass = m_i; //wall has infinite mass

    //printf("effective_radius %f \n", effective_radius);

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal = f_kn * n - DAMPING_NORMAL *  dot_product(G,n) * n * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n = vector_length(f_normal);

    //tangential force
    dfloat3 G_ct = G + r_i * cross_product(w_i,n+rri) - dot_product(G,n)*n;
    dfloat mag = vector_length(G_ct);

    //calculate tangential vector
    dfloat3 t;
    if (mag != 0){
        //tangential vector
        t = G_ct / mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }


    int row;
    if(cyDir == 1)
        row = calculateWallIndex(dfloat3(0,0,0));
    else
        row = MAX_ACTIVE_COLLISIONS-1;

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
            tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,G_ct,step);
        }
    }


    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = vector_length(f_tang);

    //calculate if will slip
    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
        f_tang = - PP_FRICTION_COEF * f_n * t;
    }

        //printf("pp  step %d fny %f fnt %f fnz %f \n",step,f_normal.x,f_tang.y, f_normal.z);

    // Force in each direction
    dfloat3 f_dirs = f_normal + f_tang;
    //Torque in each direction
    dfloat3 m_dirs_i = cross_product(rri, f_dirs);

    //printf("f_dirs   %f %f %f \n",f_dirs.x,f_dirs.y,f_dirs.z);
    //printf("m_dirs_i %f %f %f \n",m_dirs_i.x,m_dirs_i.y,m_dirs_i.z);
    //printf("rri      %f %f %f \n",rri.x,rri.y,rri.z);
    //printf("m_dirs_j %f %f %f \n",m_dirs_j.x,m_dirs_j.y,m_dirs_j.z);
    //printf("rrj      %f %f %f \n",rrj.x,rrj.y,rrj.z);
    
    // Force positive in particle i (column)
    atomicAdd(&(pc_i->f.x), f_dirs.x);
    atomicAdd(&(pc_i->f.y), f_dirs.y);
    atomicAdd(&(pc_i->f.z), f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs_i.x);
    atomicAdd(&(pc_i->M.y), m_dirs_i.y);
    atomicAdd(&(pc_i->M.z), m_dirs_i.z);


}


//sphere functions
__device__
dfloat sphereSphereGap(ParticleCenter*  pc_i, ParticleCenter*  pc_j) {
    dfloat3 p1 = pc_i->pos;
    dfloat3 p2 = pc_j->pos;

    dfloat r1 = pc_i->radius;
    dfloat r2 = pc_j->radius;

    dfloat3 delta = p1 - p2;

    #ifdef IBM_BC_X_PERIODIC
        if(delta.x > NX / 2.0) delta.x -= NX;
        if(delta.x < -NX / 2.0) delta.x += NX;
    #endif
    #ifdef IBM_BC_Y_PERIODIC
        if(delta.y > NY / 2.0) delta.y -= NY;
        if(delta.y < -NY / 2.0) delta.y += NY;
    #endif
    #ifdef IBM_BC_Z_PERIODIC
        if(delta.z > NZ / 2.0) delta.z -= NZ;
        if(delta.z < -NZ / 2.0) delta.z += NZ;
    #endif

    dfloat dist = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    
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

__device__
dfloat point_to_segment_distance_periodic(dfloat3 p, dfloat3 segA, dfloat3 segB, dfloat3 closestOnAB[1]) {
    dfloat minDist = 1E+37;  // Initialize to a large value
    dfloat3 bestClosestOnAB;
    int dx = 0, dy = 0, dz = 0;

    // Loop over periodic offsets in x, y, and z if periodic boundary conditions are enabled
    #ifdef IBM_BC_X_PERIODIC
    for (dx = -1; dx <= 1; dx++) {
    #endif
        #ifdef IBM_BC_Y_PERIODIC
        for (dy = -1; dy <= 1; dy++) {
        #endif
            #ifdef IBM_BC_Z_PERIODIC
            for (dz = -1; dz <= 1; dz++) {
            #endif
                // Translate the segment by the periodic offsets
                dfloat3 segA_translated = segA + dfloat3(dx * NX, dy * NY, dz * NZ);
                dfloat3 segB_translated = segB + dfloat3(dx * NX, dy * NY, dz * NZ);

                // Compute the closest point on the translated segment
                dfloat3 ab = segB_translated - segA_translated;
                dfloat3 ap = p - segA_translated;
                dfloat t = dot_product(ap, ab) / dot_product(ab, ab);
                t = myMax(0, myMin(1, t));  // Clamp t to [0, 1]

                dfloat3 tempClosestOnAB = segA_translated + ab * t;
                dfloat dist = vector_length(p - tempClosestOnAB);

                // Update the minimum distance and store the closest point
                if (dist < minDist) {
                    minDist = dist;
                    bestClosestOnAB = tempClosestOnAB;
                }

            #ifdef IBM_BC_Z_PERIODIC
            } // End Z loop
            #endif
        #ifdef IBM_BC_Y_PERIODIC
        } // End Y loop
        #endif
    #ifdef IBM_BC_X_PERIODIC
    } // End X loop
    #endif

    // Store the closest point on the segment
    closestOnAB[0] = bestClosestOnAB;

    // Return the minimum distance
    return minDist;
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

//ellipsoid functions
__device__
dfloat3 ellipsoid_normal(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 point, dfloat radius[1],dfloat3 translation){
    dfloat3 local_point;
    dfloat3 grad_local;
    dfloat3 normal;
    dfloat norm;


    dfloat3 center = pc_i->pos + translation;
    
    dfloat a_axis = vector_length(pc_i->collision.semiAxis-pc_i->pos);
    dfloat b_axis = vector_length(pc_i->collision.semiAxis2-pc_i->pos);
    dfloat c_axis = vector_length(pc_i->collision.semiAxis3-pc_i->pos);

    // Transform the point into the ellipsoid's local coordinates
    local_point.x = (R[0][0] * (point.x - center.x) + R[0][1] * (point.y - center.y) + R[0][2] * (point.z - center.z));
    local_point.y = (R[1][0] * (point.x - center.x) + R[1][1] * (point.y - center.y) + R[1][2] * (point.z - center.z));
    local_point.z = (R[2][0] * (point.x - center.x) + R[2][1] * (point.y - center.y) + R[2][2] * (point.z - center.z));

    dfloat a4 = a_axis*a_axis*a_axis*a_axis;
    dfloat b4 = b_axis*b_axis*b_axis*b_axis;
    dfloat c4 = c_axis*c_axis*c_axis*c_axis;


    radius[0] = (a_axis*b_axis*c_axis*(a4*b4*local_point.z*local_point.z + a4*c4*local_point.y*local_point.y+ b4*c4*local_point.x*local_point.x))/(a4*b4*c4);

    // Compute the gradient in local coordinates
    grad_local.x = 2 * local_point.x / (a_axis * a_axis);
    grad_local.y = 2 * local_point.y / (b_axis * b_axis);
    grad_local.z = 2 * local_point.z / (c_axis * c_axis);

    // Transform the gradient back to global coordinates
    normal.x = R[0][0] * grad_local.x + R[1][0] * grad_local.y + R[2][0] * grad_local.z;
    normal.y = R[0][1] * grad_local.x + R[1][1] * grad_local.y + R[2][1] * grad_local.z;
    normal.z = R[0][2] * grad_local.x + R[1][2] * grad_local.y + R[2][2] * grad_local.z;

    // Normalize the normal vector
    norm = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (norm > 0) { // Avoid division by zero
        normal.x /= norm;
        normal.y /= norm;
        normal.z /= norm;
    }


    return normal;
}
__device__
dfloat3 ellipsoid_intersection(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 line_origin, dfloat3 line_dir,dfloat3 translation){
    dfloat3 p0, p0_rotated, d_rotated;
    dfloat A, B, C;
    dfloat DELTA;
    dfloat3 t;

    dfloat3 center = pc_i->pos + translation;

    dfloat a_axis = vector_length(pc_i->collision.semiAxis-pc_i->pos);
    dfloat b_axis = vector_length(pc_i->collision.semiAxis2-pc_i->pos);
    dfloat c_axis = vector_length(pc_i->collision.semiAxis3-pc_i->pos);

    p0 = line_origin - center; // Line origin relative to the ellipsoid center


    // Apply rotation to the line origin
    p0_rotated.x = R[0][0] * p0.x + R[0][1] * p0.y + R[0][2] * p0.z;
    p0_rotated.y = R[1][0] * p0.x + R[1][1] * p0.y + R[1][2] * p0.z;
    p0_rotated.z = R[2][0] * p0.x + R[2][1] * p0.y + R[2][2] * p0.z;

    // Transform the line direction into the ellipsoid's coordinate system
    d_rotated.x = R[0][0] * line_dir.x + R[0][1] * line_dir.y + R[0][2] * line_dir.z;
    d_rotated.y = R[1][0] * line_dir.x + R[1][1] * line_dir.y + R[1][2] * line_dir.z;
    d_rotated.z = R[2][0] * line_dir.x + R[2][1] * line_dir.y + R[2][2] * line_dir.z;



    // Ellipsoid equation coefficients (in rotated space)
    A = (d_rotated.x / a_axis) * (d_rotated.x / a_axis) + (d_rotated.y / b_axis) * (d_rotated.y / b_axis) + (d_rotated.z / c_axis) * (d_rotated.z / c_axis);
    B = 2 * ((p0_rotated.x * d_rotated.x) / (a_axis * a_axis) + (p0_rotated.y * d_rotated.y) / (b_axis * b_axis) + (p0_rotated.z * d_rotated.z) / (c_axis * c_axis));
    C = (p0_rotated.x / a_axis) * (p0_rotated.x / a_axis) + (p0_rotated.y / b_axis) * (p0_rotated.y / b_axis) + (p0_rotated.z / c_axis) * (p0_rotated.z / c_axis) - 1;

    DELTA = B*B - 4*A*C;

    t = dfloat3((-B + sqrtf(DELTA)) / (2.0 * A),(-B - sqrtf(DELTA)) / (2.0 * A),0);

    return t;
}
__device__
dfloat ellipsoidWallCollisionDistance( ParticleCenter* pc_i, Wall wallData,dfloat3 contactPoint2[1], dfloat radius[1], unsigned int step){
    //contruct rotation matrix
    dfloat R[3][3];
    dfloat dist, error;
    dfloat3 new_sphere_center1, new_sphere_center2;
    dfloat3 closest_point1, closest_point2;

    dfloat a = vector_length(pc_i->collision.semiAxis-pc_i->pos);
    dfloat b = vector_length(pc_i->collision.semiAxis2-pc_i->pos);
    dfloat c = vector_length(pc_i->collision.semiAxis3-pc_i->pos);
    
    rotationMatrixFromVectors((pc_i->collision.semiAxis - pc_i->pos )/a,(pc_i->collision.semiAxis2 - pc_i->pos )/b,(pc_i->collision.semiAxis3 - pc_i->pos)/c,R);


    //projection of center into wall
    dfloat3 proj = planeProjection(pc_i->pos,wallData.normal,wallData.distance);
    dfloat3 dir = pc_i->pos - proj;
    dfloat3 t = ellipsoid_intersection(pc_i,R,proj,dir,dfloat3(0,0,0));
    dfloat3 inter1 = proj + t.x*dir;
    dfloat3 inter2 = proj + t.y*dir;


    if (dot_product(inter1,wallData.normal) < dot_product(inter2,wallData.normal)){
        closest_point2 = inter1;
    }else{
        closest_point2 = inter2;
    }    
    
    dfloat r = 3; //TODO: FIND A BETTER WAY TI DETERMINE IT

    //compute normal vector at intersection
    dfloat3 normal2 = ellipsoid_normal(pc_i,R,closest_point2,radius,dfloat3(0,0,0));

    //Compute the centers of the spheres in the opposite direction of the normals
    dfloat3 sphere_center2 = closest_point2 - r * normal2;

    //Iteration loop
    dfloat max_iters = 20;
    dfloat tolerance = 1e-3;

    for(int i = 0; i< max_iters;i++){
        proj = planeProjection(sphere_center2,wallData.normal,wallData.distance);
        dir = sphere_center2 - proj;
        t = ellipsoid_intersection(pc_i,R,proj,dir,dfloat3(0,0,0));

        inter1 = proj + t.x*dir;
        inter2 = proj + t.y*dir;

        if (dot_product(inter1,wallData.normal) < dot_product(inter2,wallData.normal)){
            closest_point2 = inter1;
        }else{
            closest_point2 = inter2;
        }    

        normal2 = ellipsoid_normal(pc_i,R,closest_point2,radius,dfloat3(0,0,0));        
        new_sphere_center2 = closest_point2 - r * normal2;

        error = vector_length(new_sphere_center2 - sphere_center2);
        if (error < tolerance ){
            break;      
        }else{
            //update values
            sphere_center2 = new_sphere_center2;
        }
    }

    contactPoint2[0] = closest_point2;
    dist = vector_length(sphere_center2 - proj) - r;
    return dist;

}

__device__
void computeContactPoints(dfloat3 pos_i, dfloat3 dir, dfloat3 t1, dfloat3 t2, dfloat3 contactPoint1[1], dfloat3 contactPoint2[1])
{


    dfloat3 inter11 = pos_i + t1.x * dir;
    dfloat3 inter12 = pos_i + t1.y * dir;
    dfloat3 inter21 = pos_i + t2.x * dir;
    dfloat3 inter22 = pos_i + t2.y * dir;

    // compute distances
    dfloat distances[2][2];
    distances[0][0] = vector_length(inter11 - inter21);
    distances[0][1] = vector_length(inter11 - inter22);
    distances[1][0] = vector_length(inter12 - inter21);
    distances[1][1] = vector_length(inter12 - inter22);

    // find minimum distances
    dfloat min_dist = 1e37;
    int i_min = 0;
    int j_min = 0;
    if (distances[0][0] < min_dist)
    {
        min_dist = distances[0][0];
        i_min = 0;
        j_min = 0;
    }
    if (distances[0][1] < min_dist)
    {
        min_dist = distances[0][1];
        i_min = 0;
        j_min = 1;
    }
    if (distances[1][0] < min_dist)
    {
        min_dist = distances[1][0];
        i_min = 1;
        j_min = 0;
    }
    if (distances[1][1] < min_dist)
    {
        min_dist = distances[1][1];
        i_min = 1;
        j_min = 1;
    }

    switch (i_min * 2 + j_min)
    {
    case 0: // i_min = 0, j_min = 0
        contactPoint1[0] = inter11;
        contactPoint2[0] = inter21;
        break;
    case 1: // i_min = 0, j_min = 1
        contactPoint1[0] = inter11;
        contactPoint2[0] = inter22;
        break;
    case 2: // i_min = 1, j_min = 0
        contactPoint1[0] = inter12;
        contactPoint2[0] = inter21;
        break;
    case 3: // i_min = 1, j_min = 1
        contactPoint1[0] = inter12;
        contactPoint2[0] = inter22;
        break;
    default:
        break;
    }
}
 
__device__
dfloat ellipsoidEllipsoidCollisionDistance( ParticleCenter* pc_i, ParticleCenter* pc_j,dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr1[1], dfloat cr2[1], dfloat3 translation, unsigned int step){
    dfloat R1[3][3];
    dfloat R2[3][3];
    dfloat dist, error;
    dfloat3 new_sphere_center1, new_sphere_center2;
    dfloat3 closest_point1, closest_point2;

    //obtain semi-axis values
    dfloat a1 = vector_length(pc_i->collision.semiAxis-pc_i->pos);
    dfloat b1 = vector_length(pc_i->collision.semiAxis2-pc_i->pos);
    dfloat c1 = vector_length(pc_i->collision.semiAxis3-pc_i->pos);
    
    dfloat a2 = vector_length(pc_j->collision.semiAxis-pc_j->pos);
    dfloat b2 = vector_length(pc_j->collision.semiAxis2-pc_j->pos);
    dfloat c2 = vector_length(pc_j->collision.semiAxis3-pc_j->pos);


    //obtain rotation matrix
    rotationMatrixFromVectors((pc_i->collision.semiAxis - pc_i->pos )/a1,(pc_i->collision.semiAxis2 - pc_i->pos )/b1,(pc_i->collision.semiAxis3 - pc_i->pos)/c1,R1);
    rotationMatrixFromVectors((pc_j->collision.semiAxis - pc_j->pos )/a2,(pc_j->collision.semiAxis2 - pc_j->pos )/b2,(pc_j->collision.semiAxis3 - pc_j->pos)/c2,R2);

    dfloat3 dir = (pc_j->pos + translation) - pc_i->pos;

    dfloat3 t1 = ellipsoid_intersection(pc_i,R1,pc_i->pos,dir,dfloat3(0,0,0));
    dfloat3 t2 = ellipsoid_intersection(pc_j,R2,pc_i->pos,dir,translation);

    computeContactPoints(pc_i->pos, dir, t1,  t2, &closest_point1, &closest_point2);

    dfloat r = 3; //TODO FIND A BETTER WAY TO GET THE BEST RADIUS FOR THIS DETECTION


    //compute normal vector at intersection
    dfloat3 normal1 = ellipsoid_normal(pc_i,R1,closest_point1,cr1,dfloat3(0,0,0));
    dfloat3 normal2 = ellipsoid_normal(pc_j,R2,closest_point2,cr2,translation);
    dfloat3 sphere_center1 = closest_point1 - r * normal1;
    dfloat3 sphere_center2 = closest_point2 - r * normal2;


    //Iteration loop
    dfloat max_iters = 20;
    dfloat tolerance = 1e-3;
    for(int i = 0; i< max_iters;i++){
         dir = sphere_center2 - sphere_center1;
         
        t1 = ellipsoid_intersection(pc_i,R1,sphere_center1,dir,dfloat3(0,0,0));
        t2 = ellipsoid_intersection(pc_j,R2,sphere_center1,dir,translation);

        computeContactPoints(sphere_center1, dir, t1, t2, &closest_point1, &closest_point2);

        normal1 = ellipsoid_normal(pc_i,R1,closest_point1,cr1,dfloat3(0,0,0));
        normal2 = ellipsoid_normal(pc_j,R2,closest_point2,cr2,translation);

        new_sphere_center1 = closest_point1 - r * normal1;
        new_sphere_center2 = closest_point2 - r * normal2;

        error = vector_length(new_sphere_center2 - sphere_center2) + vector_length(new_sphere_center1 - sphere_center1);
        //printf("error %d %f \n",error);
        if (error < tolerance ){
            break;      
        }else{
            //update values
            sphere_center1 = new_sphere_center1;
            sphere_center2 = new_sphere_center2;
        }
    }

    contactPoint1[0] = closest_point1;
    contactPoint2[0] = closest_point2;
    dist = vector_length(sphere_center2 - sphere_center1) - 2*r;
    return dist;

}

__device__
dfloat3 segmentProjection(dfloat3 P, dfloat3 P1, dfloat3 P2, dfloat cRadius, int cyDir ){
    dfloat3 closestOnAB[1];

    dfloat dist = point_to_segment_distance(P, P1, P2, closestOnAB);
    dfloat3 n =  vector_normalize(closestOnAB[0] - P);
    dfloat3 contactPoint =  closestOnAB[0] - n *  cRadius;

    return contactPoint;
}

__device__
dfloat ellipsoidSegmentCollisionDistance( ParticleCenter* pc_i, dfloat3 P1, dfloat3 P2, dfloat cRadius ,dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr[1], int cyDir, unsigned int step){
    dfloat RE[3][3];
    dfloat R2[3][3];
    dfloat dist, error;
    dfloat3 new_sphere_center1, new_sphere_center2;
    dfloat3 closest_point1, closest_point2;


    //obtain semi-axis values
    dfloat a = vector_length(pc_i->collision.semiAxis-pc_i->pos);
    dfloat b = vector_length(pc_i->collision.semiAxis2-pc_i->pos);
    dfloat c = vector_length(pc_i->collision.semiAxis3-pc_i->pos);


    rotationMatrixFromVectors((pc_i->collision.semiAxis - pc_i->pos )/a,(pc_i->collision.semiAxis2 - pc_i->pos )/b,(pc_i->collision.semiAxis3 - pc_i->pos)/c,RE);

    //projection of center into segment
    dfloat3 proj = segmentProjection(pc_i->pos,P1,P2,cRadius,cyDir);
    dfloat3 dir = pc_i->pos - proj;
    dfloat3 t = ellipsoid_intersection(pc_i,RE,proj,dir,dfloat3(0,0,0));
    dfloat3 inter1 = proj + t.x*dir;
    dfloat3 inter2 = proj + t.y*dir;
    

    if (vector_length(inter1 - proj) < vector_length(inter2 - proj)){
        closest_point2 = inter1;
    }else{
        closest_point2 = inter2;
    }    

    dfloat r = 3; //TODO: FIND A BETTER WAY TI DETERMINE IT

    //compute normal vector at intersection
    dfloat3 normal2 = ellipsoid_normal(pc_i,RE,closest_point2,cr,dfloat3(0,0,0));


    //Compute the centers of the spheres in the opposite direction of the normals
    dfloat3 sphere_center2 = closest_point2 - r * normal2;

    //Iteration loop
    dfloat max_iters = 20;
    dfloat tolerance = 1e-3;
    for(int i = 0; i< max_iters;i++){
        proj = segmentProjection(sphere_center2,P1,P2,cRadius, cyDir);
        dir = sphere_center2 - proj;
        t = ellipsoid_intersection(pc_i,RE,proj,dir,dfloat3(0,0,0));

        inter1 = proj + t.x*dir;
        inter2 = proj + t.y*dir;

        if (vector_length(inter1 - proj) < vector_length(inter2 - proj)){
            closest_point2 = inter1;
        }else{
            closest_point2 = inter2;
        }        

        normal2 = ellipsoid_normal(pc_i,RE,closest_point2,cr,dfloat3(0,0,0));
        new_sphere_center2 = closest_point2 - r * normal2;

        error = vector_length(new_sphere_center2 - sphere_center2);
        if (error < tolerance ){
            break;      
        }else{
            //update values
            sphere_center2 = new_sphere_center2;
        }
    }


    contactPoint1[0] = proj;
    contactPoint2[0] = closest_point2;
    dist = vector_length(sphere_center2 - proj) - r;
    return dist;
}


// ------------------------------------------------------------------------
// -------------------- COLLISION BETWEEN PARTICLES -----------------------
// ------------------------------------------------------------------------ 

__device__
void checkCollisionBetweenParticles( unsigned int column,unsigned int row,ParticleCenter* pc_i,  ParticleCenter* pc_j,int step){

    //printf("shape %d %d \n",pc_i->collision.shape,pc_j->collision.shape);
    switch (pc_i->collision.shape) {
        case SPHERE:
            switch (pc_j->collision.shape) {
            case SPHERE:
            //printf("sph - sph col \n");
                //printf("collision between spheres \n");
                if(sphereSphereGap( pc_i, pc_j)<0){
                    sphereSphereCollision(column,row, pc_i, pc_j,step);
                }
                break;
            case CAPSULE:
            //printf("sphe - cap col \n");
                capsuleSphereCollisionCheck(column,row,pc_i,pc_j,step);
                break;
            case ELLIPSOID:
            //printf("sph - eli col \n");
                //collision sphere-ellipsoid
                break;
            default:
            //printf("sph - def col \n");
                // Handle unknown particle types
                break;
            }
            break;
        case CAPSULE:
            switch (pc_j->collision.shape) {
            case SPHERE:
                //printf("cap - sph col \n");
                capsuleSphereCollisionCheck(column,row,pc_i,pc_j,step);
                break;
            case CAPSULE:
                //printf("cap - cap col \n");
                capsuleCapsuleCollisionCheck(column,row,pc_i,pc_j, step, pc_i->collision.semiAxis,pc_i->collision.semiAxis2, pc_i->radius,pc_j->collision.semiAxis, pc_j->collision.semiAxis2, pc_j->radius);
            case ELLIPSOID:
                //printf("cap - eli col \n");
                //collision capsule-ellipsoid
                break;
            default:
                // Handle unknown particle types
                //printf("cap - def col \n");
                break;
            }
            break;
        case ELLIPSOID:
            switch (pc_j->collision.shape) {
            case SPHERE:
                //printf("eli - sphere col \n");
                //collision ellipsoid-sphere
                break;
            case CAPSULE:
                //printf("eli - cap col \n"); 
                //collision ellipsoid-capsule
                break;
            case ELLIPSOID:
                //printf("eli - eli col \n");
                ellipsoidEllipsoidCollisionCheck(column,row,pc_i,pc_j, step);
                break;
            default:
                //printf("eli - default col \n");
                // Handle unknown particle types
                break;
            }
            break;
        default:
            //printf("default - default col \n");
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
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
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
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
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
// -------------------- ELLIPSOID COLLISIONS ------------------------------
// ------------------------------------------------------------------------ 

__device__
void ellipsoidEllipsoidCollision(unsigned int column, unsigned int row, ParticleCenter*  pc_i, ParticleCenter*  pc_j,dfloat3 closestOnA[1], dfloat3 closestOnB[1], dfloat dist, dfloat cr1[1], dfloat cr2[1], dfloat3 translation, int step){
    // Particle i info (column)
    const dfloat3 pos_i = closestOnA[0];
    const dfloat3 pos_c_i = pc_i->pos;
    const dfloat r_i = pc_i->radius;
    const dfloat m_i = pc_i ->volume * pc_i ->density;
    const dfloat3 v_i = pc_i->vel;
    const dfloat3 w_i = pc_i->w;
   
    // Particle j info (row)
    const dfloat3 pos_j = closestOnB[0] + translation;
    const dfloat3 pos_c_j = pc_j->pos + translation;
    const dfloat r_j = pc_j->radius;
    const dfloat m_j = pc_j ->volume * pc_j ->density;
    const dfloat3 v_j = pc_j->vel;
    const dfloat3 w_j = pc_j->w;

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

    const dfloat mag_dist = abs(dist);

    //normal collision vector
    const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);

    //normal deformation
    dfloat displacement = -dist;

    //vector center-> contact 
    dfloat3 rri = pos_i - pos_c_i;
    dfloat3 rrj = pos_j - pos_c_j;

    // relative velocity vector
    dfloat3 G = v_i-v_j;

    //HERTZ CONTACT THEORY

    //dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_radius = 1.0/((cr1[0] +cr2[0])/(cr1[0]*cr2[0]));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);


    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal = f_kn * n - DAMPING_NORMAL *  dot_product(G,n) * n * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n = vector_length(f_normal);


    //tangential force
    dfloat3 G_ct = G + r_i * cross_product(w_i,n+rri) - dot_product(G,n)*n;
    dfloat mag = vector_length(G_ct);


        //calculate tangential vector
    dfloat3 t;
    if (mag != 0){
        //tangential vector
        t = G_ct / mag;
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
            tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,G_ct,step);
        }
    }


    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = vector_length(f_tang);

    //calculate if will slip
    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        tang_disp = updateTangentialDisplacement(pc_i->collision,tang_index,-G_ct,step);
        f_tang = - PP_FRICTION_COEF * f_n * t;
    }

    //FINAL FORCE RESULTS

    //printf("pp  step %d fny %f fnt %f fnz %f \n",step,f_normal.x,f_tang.y, f_normal.z);

    // Force in each direction
    dfloat3 f_dirs = f_normal + f_tang;
    //Torque in each direction
    dfloat3 m_dirs_i = cross_product(rri, f_dirs);
    dfloat3 m_dirs_j = cross_product(rrj, -f_dirs);

    //printf("f_dirs   %f %f %f \n",f_dirs.x,f_dirs.y,f_dirs.z);
    //printf("m_dirs_i %f %f %f \n",m_dirs_i.x,m_dirs_i.y,m_dirs_i.z);
    //printf("rri      %f %f %f \n",rri.x,rri.y,rri.z);
    //printf("m_dirs_j %f %f %f \n",m_dirs_j.x,m_dirs_j.y,m_dirs_j.z);
    //printf("rrj      %f %f %f \n",rrj.x,rrj.y,rrj.z);
    
    // Force positive in particle i (column)
    atomicAdd(&(pc_i->f.x), f_dirs.x);
    atomicAdd(&(pc_i->f.y), f_dirs.y);
    atomicAdd(&(pc_i->f.z), f_dirs.z);

    atomicAdd(&(pc_i->M.x), m_dirs_i.x);
    atomicAdd(&(pc_i->M.y), m_dirs_i.y);
    atomicAdd(&(pc_i->M.z), m_dirs_i.z);

    // Force negative in particle j (row)
    atomicAdd(&(pc_j->f.x), -f_dirs.x);
    atomicAdd(&(pc_j->f.y), -f_dirs.y);
    atomicAdd(&(pc_j->f.z), -f_dirs.z);

    atomicAdd(&(pc_j->M.x), m_dirs_j.x); //normal vector takes care of negative sign
    atomicAdd(&(pc_j->M.y), m_dirs_j.y);
    atomicAdd(&(pc_j->M.z), m_dirs_j.z); 


}

// ------------------------------------------------------------------------ 
// -------------------- INTER PARTICLE COLLISION CHECK---------------------
// ------------------------------------------------------------------------ 

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
void capsuleSphereCollisionCheck(unsigned int column,unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step){

    dfloat3 closestOnAB[1];

    if(pc_i->collision.shape == SPHERE){
        if(point_to_segment_distance_periodic(pc_i->pos, pc_j->collision.semiAxis, pc_j->collision.semiAxis2,closestOnAB) < pc_i->radius + pc_j->radius)
            capsuleCapsuleCollision(column,row,pc_i,pc_j,&pc_i->pos,closestOnAB,step);
    }else{
        if(point_to_segment_distance_periodic(pc_j->pos, pc_i->collision.semiAxis, pc_i->collision.semiAxis2,closestOnAB) < pc_i->radius + pc_j->radius)
            capsuleCapsuleCollision(column,row,pc_i,pc_j,&pc_j->pos,closestOnAB,step);
    }
    

    return;
}

__device__
void ellipsoidEllipsoidCollisionCheck(unsigned int column, unsigned int row, ParticleCenter* pc_i,ParticleCenter* pc_j, int step){
    dfloat3 closestOnA[1];
    dfloat3 closestOnB[1];
    //printf("checking collision\n");
    dfloat cr1[1];
    dfloat cr2[1];

    dfloat minDist = 1E+37;  // Initialize to a large value
    dfloat3 bestClosestOnA;
    dfloat3 bestClosestOnB;
    dfloat bestcr1, bestcr2;
    int dx = 0, dy = 0, dz = 0;
    dfloat dist;
    dfloat3 translation, bestTranslation;

    // Loop over periodic offsets in x, y, and z if periodic boundary conditions are enabled
    #ifdef IBM_BC_X_PERIODIC
    for (dx = -1; dx <= 1; dx++) {
    #endif
        #ifdef IBM_BC_Y_PERIODIC
        for (dy = -1; dy <= 1; dy++) {
        #endif
            #ifdef IBM_BC_Z_PERIODIC
            for (dz = -1; dz <= 1; dz++) {
            #endif
                translation = dfloat3(dx * (NX-1), dy * (NY-1), dz * (NZ-1));
                dist = ellipsoidEllipsoidCollisionDistance(pc_i, pc_j,closestOnA,closestOnB,cr1, cr2,translation,step);
                // Update the minimum distance and store the closest point
                if (dist < minDist) {
                    minDist = dist;
                    bestClosestOnA = closestOnA[0];
                    bestClosestOnB = closestOnB[0];
                    bestcr1 = cr1[0];
                    bestcr2 = cr2[0];
                    bestTranslation = translation;
                }


            #ifdef IBM_BC_Z_PERIODIC
            } // End Z loop
            #endif
        #ifdef IBM_BC_Y_PERIODIC
        } // End Y loop
        #endif
    #ifdef IBM_BC_X_PERIODIC
    } // End X loop
    #endif

    // Store the closest point on the segment
    closestOnA[0] = bestClosestOnA;
    closestOnB[0] = bestClosestOnB-bestTranslation;
    cr1[0] = bestcr1;
    cr2[0] = bestcr2;
    dist = minDist;

    if(dist<0){
        ellipsoidEllipsoidCollision(column, row,pc_i,pc_j,closestOnA,closestOnB,dist,cr1, cr2,bestTranslation,step);
    }


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
