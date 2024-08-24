#include "globalFunctions.h"

__global__
void copyFromArray(dfloat3SoA dst, dfloat3SoA src){
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= (NZ+4))
        return;

    size_t i = idxScalar(x, y, z);

    dst.x[i] = src.x[i];
    dst.y[i] = src.y[i];
    dst.z[i] = src.z[i];
}

__host__ __device__
dfloat3 cross_product(dfloat3 v1, dfloat3 v2) {
    dfloat3 cross;
    cross.x = v1.y * v2.z - v1.z * v2.y;
    cross.y = v1.z * v2.x - v1.x * v2.z;
    cross.z = v1.x * v2.y - v1.y * v2.x;
    return cross;
}

__host__ __device__
dfloat dot_product(dfloat3 v1, dfloat3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}


__host__ __device__
dfloat4 quart_multiplication(dfloat4 q1, dfloat4 q2){
    dfloat4 q;
    
    q.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    q.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    q.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    q.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;

    return q;
}

__host__ __device__
dfloat3 vector_normalize(dfloat3 v) {
    dfloat inv_length = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    dfloat3 norm_v;

    if (isnan(inv_length))
        norm_v = dfloat3(0,0,0);
    else
        norm_v = dfloat3(v.x * inv_length, v.y * inv_length, v.z * inv_length);

    return norm_v;
}

__host__ __device__
dfloat4 quart_normalize(dfloat4 q) {
    dfloat norm = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    q.w /= norm;
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    return q;
}

__host__ __device__
dfloat4 quart_addition(dfloat4 q1, dfloat4 q2){
    dfloat4 q;
    q.w = q1.w + q2.w;
    q.x = q1.x + q2.x;
    q.y = q1.y + q2.y;
    q.z = q1.z + q2.z;
    return q;
}

__host__ __device__
dfloat4 vector_to_quart(dfloat3 v) {
    dfloat4 q;
    q.w = 0.0; // The scalar part is zero
    q.x = v.x;
    q.y = v.y;
    q.z = v.z;
    return q;
}

__host__ __device__
void quart_to_rotation_matrix(dfloat4 q, dfloat R[3][3]){
    dfloat qx2 = q.x * q.x;
    dfloat qy2 = q.y * q.y;
    dfloat qz2 = q.z * q.z;
    dfloat qwqx = q.w * q.x;
    dfloat qwqy = q.w * q.y;
    dfloat qwqz = q.w * q.z;
    dfloat qxqy = q.x * q.y;
    dfloat qxqz = q.x * q.z;
    dfloat qyqz = q.y * q.z;

    R[0][0] = 1 - 2 * (qy2 + qz2);
    R[0][1] = 2 * (qxqy - qwqz);
    R[0][2] = 2 * (qxqz + qwqy);

    R[1][0] = 2 * (qxqy + qwqz);
    R[1][1] = 1 - 2 * (qx2 + qz2);
    R[1][2] = 2 * (qyqz - qwqx);

    R[2][0] = 2 * (qxqz - qwqy);
    R[2][1] = 2 * (qyqz + qwqx);
    R[2][2] = 1 - 2 * (qx2 + qy2);
}


// Rotate a vector using a rotation matrix
__host__ __device__
dfloat3 rotate_vector_by_matrix(dfloat3 v, dfloat R[3][3]) {
    dfloat3 v_rot;
    v_rot.x = R[0][0] * v.x + R[0][1] * v.y + R[0][2] * v.z;
    v_rot.y = R[1][0] * v.x + R[1][1] * v.y + R[1][2] * v.z;
    v_rot.z = R[2][0] * v.x + R[2][1] * v.y + R[2][2] * v.z;
    return v_rot;
}

__host__ __device__
dfloat3 rotate_vector_by_quart_R(dfloat3 v, dfloat4 q){
    dfloat R[3][3];
    quart_to_rotation_matrix(q, R);
    return rotate_vector_by_matrix(v, R);
}

// Quaternion conjugate: q_conj = q^*
__host__ __device__
dfloat4 quart_conjugate(dfloat4 q) {
    dfloat4 q_conj;
    q_conj.w = q.w;
    q_conj.x = -q.x;
    q_conj.y = -q.y;
    q_conj.z = -q.z;
    return q_conj;
}

__host__ __device__
dfloat3 rotate_vector_by_quart(dfloat3 v, dfloat4 q) {
    // Convert vector to a quaternion with w = 0
    dfloat4 q_vec =vector_to_quart(v);

    // Calculate rotated quaternion: q_rot = q * q_vec * q^*
    dfloat4 q_conj = quart_conjugate(q);
    dfloat4 q_rot = quart_multiplication(quart_multiplication(q, q_vec), q_conj);

    // The resulting vector is the vector part of q_rot
    dfloat3 v_rot;
    v_rot.x = q_rot.x;
    v_rot.y = q_rot.y;
    v_rot.z = q_rot.z;
    
    return v_rot;
}


// Function to compute the quaternion that rotates vector v1 to vector v2
__host__ __device__
dfloat4 compute_rotation_quart(dfloat3 v1, dfloat3 v2) {
    v1 = vector_normalize(v1);
    v2 = vector_normalize(v2);

    dfloat dot = dot_product(v1, v2);

    // Calculate the angle of rotation
    dfloat angle = acos(dot);

    // Calculate the axis of rotation
    dfloat3 axis = cross_product(v1, v2);
    axis = vector_normalize(axis);

    dfloat4 q;
    q.w = cos(angle / 2.0);
    q.x = axis.x * sin(angle / 2.0);
    q.y = axis.y * sin(angle / 2.0);
    q.z = axis.z * sin(angle / 2.0);

    return q;
}

__host__ __device__
dfloat4 axis_angle_to_quart(dfloat3 axis, dfloat angle) {
    dfloat4 q;
    
    // Normalize the axis of rotation
    axis = vector_normalize(axis);
    
    // Compute the quaternion
    q.w = cos(angle / 2.0);
    q.x = axis.x * sin(angle / 2.0);
    q.y = axis.y * sin(angle / 2.0);
    q.z = axis.z * sin(angle / 2.0);
    
    return q;
}

__host__ __device__
dfloat4 euler_to_quart(dfloat roll, dfloat pitch, dfloat yaw){
    dfloat cr = cos(roll * 0.5);
    dfloat sr = sin(roll * 0.5);
    dfloat cp = cos(pitch * 0.5);
    dfloat sp = sin(pitch * 0.5);
    dfloat cy = cos(yaw * 0.5);
    dfloat sy = sin(yaw * 0.5);

    dfloat4 q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

__host__ __device__
dfloat3 quart_to_euler(dfloat4 q){
    dfloat3 angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = std::sqrt(1 + 2 * (q.w * q.y - q.x * q.z));
    double cosp = std::sqrt(1 - 2 * (q.w * q.y - q.x * q.z));
    angles.y = 2 * std::atan2(sinp, cosp) - M_PI / 2;

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.z = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}