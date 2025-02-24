#define MY_MUL(a, b, c) c.x = a.x * b.x - a.y * b.y; c.y = a.y * b.x + a.x * b.y;
#define MY_MUL_REPLACE(a, b, c, d) d.x = a.x * b.x - a.y * b.y; d.y = a.y * b.x + a.x * b.y; c = d;
#define MY_ANGLE2COMPLEX(angle, a) a.x = __cosf(angle); a.y =  __sinf(angle); 


#define turboFFT_ZADD(c, a, b) c.x = a.x + b.x; c.y = a.y + b.y;
#define turboFFT_ZSUB(c, a, b) c.x = a.x - b.x; c.y = a.y - b.y;
#define turboFFT_ZMUL(c, a, b) c.x = a.x * b.x; c.x -= a.y * b.y; c.y = a.y * b.x; c.y += a.x * b.y;

// __global__ void fft_7(float2* gPtr, float2* rPtr) {
extern __shared__ float shared_mem[];
__global__ void fft_7(float2* gPtr, float2* outputs) {
    int bid_cnt = 0;
    int j;
    int k;
    int global_j;
    int global_k;
    int data_id;
    int bs_id;
    int shared_offset_bs;
    int shared_offset_data;
    int bx;
    int tx;
    int offset;
    float2* shPtr;
    float2 rPtr[8];
    float2 rPtr_3[8];
    float2 tmp;
    float2 angle;
    float2 delta_angle;
    j = 0;
    k = -1;
    global_j = 0;
    global_k = 0;
    data_id = 0;
    bs_id = 0;
    shared_offset_bs = 0;
    shared_offset_data = 0;
    bx = blockIdx.x;
    tx = threadIdx.x;
    offset = 0;
    gPtr = gPtr;
    shPtr = (float2*)shared_mem;
    
    int bid = 0;
            
    bx = BID_X;
    tx = TID;
    
            gPtr = gPtr;
    
        rPtr[0] = *(gPtr + 0);
        rPtr_3[0].x += rPtr[0].x;
        rPtr_3[0].y += rPtr[0].y;
        
        rPtr[1] = *(gPtr + 16);
        rPtr_3[1].x += rPtr[1].x;
        rPtr_3[1].y += rPtr[1].y;
        
        rPtr[2] = *(gPtr + 32);
        rPtr_3[2].x += rPtr[2].x;
        rPtr_3[2].y += rPtr[2].y;
        
        rPtr[3] = *(gPtr + 48);
        rPtr_3[3].x += rPtr[3].x;
        rPtr_3[3].y += rPtr[3].y;
        
        rPtr[4] = *(gPtr + 64);
        rPtr_3[4].x += rPtr[4].x;
        rPtr_3[4].y += rPtr[4].y;
        
        rPtr[5] = *(gPtr + 80);
        rPtr_3[5].x += rPtr[5].x;
        rPtr_3[5].y += rPtr[5].y;
        
        rPtr[6] = *(gPtr + 96);
        rPtr_3[6].x += rPtr[6].x;
        rPtr_3[6].y += rPtr[6].y;
        
        rPtr[7] = *(gPtr + 112);
        rPtr_3[7].x += rPtr[7].x;
        rPtr_3[7].y += rPtr[7].y;
        
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[4]);
    turboFFT_ZSUB(rPtr[4], tmp, rPtr[4]);
    tmp = rPtr[4];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
        angle.x = 0.7071067811865476f;
        angle.y = -0.7071067811865475f;
        turboFFT_ZMUL(rPtr[5], tmp, angle);
        
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    rPtr[6].y = -tmp.x;
    rPtr[6].x = tmp.y;
    
    tmp = rPtr[3];
    turboFFT_ZADD(rPtr[3], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
        angle.x = -0.7071067811865475f;
        angle.y = -0.7071067811865476f;
        turboFFT_ZMUL(rPtr[7], tmp, angle);
        
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[2]);
    turboFFT_ZSUB(rPtr[2], tmp, rPtr[2]);
    tmp = rPtr[2];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    rPtr[3].y = -tmp.x;
    rPtr[3].x = tmp.y;
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    tmp = rPtr[5];
    turboFFT_ZADD(rPtr[5], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    rPtr[7].y = -tmp.x;
    rPtr[7].x = tmp.y;
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[1]);
    turboFFT_ZSUB(rPtr[1], tmp, rPtr[1]);
    tmp = rPtr[1];
    
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
    tmp = rPtr[6];
    turboFFT_ZADD(rPtr[6], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    j = 0;
    offset  = 0;
    
    offset += ((tx / 1) % 1) * 1;
    
    j = tx / 1;
    
    offset += ((tx / 1) % 2) * 8;
    
    offset += ((tx / 2) % 8) * 16;
    
    __syncthreads();
    
            rPtr_3[0] = rPtr[0];
    
            rPtr_3[1] = rPtr[4];
    
            rPtr_3[2] = rPtr[2];
    
            rPtr_3[3] = rPtr[6];
    
            rPtr_3[4] = rPtr[1];
    
            rPtr_3[5] = rPtr[5];
    
            rPtr_3[6] = rPtr[3];
    
            rPtr_3[7] = rPtr[7];
    
    delta_angle.x = __cosf(j * -0.04908738657832146f);
    delta_angle.y = __sinf(j * -0.04908738657832146f);
     
    angle.x = 1;
    angle.y = 0;
    
    shPtr[offset + 1 * (0 + (threadIdx.x / 2)) % 8] = rPtr_3[(0 + (threadIdx.x / 2)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[4];
    turboFFT_ZMUL(rPtr[4], tmp, angle);
    
    shPtr[offset + 1 * (1 + (threadIdx.x / 2)) % 8] = rPtr_3[(1 + (threadIdx.x / 2)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[2];
    turboFFT_ZMUL(rPtr[2], tmp, angle);
    
    shPtr[offset + 1 * (2 + (threadIdx.x / 2)) % 8] = rPtr_3[(2 + (threadIdx.x / 2)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[6];
    turboFFT_ZMUL(rPtr[6], tmp, angle);
    
    shPtr[offset + 1 * (3 + (threadIdx.x / 2)) % 8] = rPtr_3[(3 + (threadIdx.x / 2)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[1];
    turboFFT_ZMUL(rPtr[1], tmp, angle);
    
    shPtr[offset + 1 * (4 + (threadIdx.x / 2)) % 8] = rPtr_3[(4 + (threadIdx.x / 2)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[5];
    turboFFT_ZMUL(rPtr[5], tmp, angle);
    
    shPtr[offset + 1 * (5 + (threadIdx.x / 2)) % 8] = rPtr_3[(5 + (threadIdx.x / 2)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[3];
    turboFFT_ZMUL(rPtr[3], tmp, angle);
    
    shPtr[offset + 1 * (6 + (threadIdx.x / 2)) % 8] = rPtr_3[(6 + (threadIdx.x / 2)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[7];
    turboFFT_ZMUL(rPtr[7], tmp, angle);
    
    shPtr[offset + 1 * (7 + (threadIdx.x / 2)) % 8] = rPtr_3[(7 + (threadIdx.x / 2)) % 8];
    
    offset = 0;
    offset += tx % 16 + tx / 16 * 128;
    
    __syncthreads();
    
    rPtr[0] = shPtr[offset + 0];
    
    rPtr[1] = shPtr[offset + 16];
    
    rPtr[2] = shPtr[offset + 32];
    
    rPtr[3] = shPtr[offset + 48];
    
    rPtr[4] = shPtr[offset + 64];
    
    rPtr[5] = shPtr[offset + 80];
    
    rPtr[6] = shPtr[offset + 96];
    
    rPtr[7] = shPtr[offset + 112];
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[4]);
    turboFFT_ZSUB(rPtr[4], tmp, rPtr[4]);
    tmp = rPtr[4];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
        angle.x = 0.7071067811865476f;
        angle.y = -0.7071067811865475f;
        turboFFT_ZMUL(rPtr[5], tmp, angle);
        
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    rPtr[6].y = -tmp.x;
    rPtr[6].x = tmp.y;
    
    tmp = rPtr[3];
    turboFFT_ZADD(rPtr[3], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
        angle.x = -0.7071067811865475f;
        angle.y = -0.7071067811865476f;
        turboFFT_ZMUL(rPtr[7], tmp, angle);
        
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[2]);
    turboFFT_ZSUB(rPtr[2], tmp, rPtr[2]);
    tmp = rPtr[2];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    rPtr[3].y = -tmp.x;
    rPtr[3].x = tmp.y;
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    tmp = rPtr[5];
    turboFFT_ZADD(rPtr[5], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    rPtr[7].y = -tmp.x;
    rPtr[7].x = tmp.y;
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[1]);
    turboFFT_ZSUB(rPtr[1], tmp, rPtr[1]);
    tmp = rPtr[1];
    
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[3]);
    turboFFT_ZSUB(rPtr[3], tmp, rPtr[3]);
    tmp = rPtr[3];
    
    tmp = rPtr[4];
    turboFFT_ZADD(rPtr[4], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
    tmp = rPtr[6];
    turboFFT_ZADD(rPtr[6], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
    
    j = 0;
    offset  = 0;
    
    offset += ((tx / 1) % 1) * 1;
    
    offset += ((tx / 1) % 8) * 1;
    
    j = tx / 8;
    
    offset += ((tx / 8) % 2) * 64;
    
    __syncthreads();
    
            rPtr_3[0] = rPtr[0];
    
            rPtr_3[1] = rPtr[4];
    
            rPtr_3[2] = rPtr[2];
    
            rPtr_3[3] = rPtr[6];
    
            rPtr_3[4] = rPtr[1];
    
            rPtr_3[5] = rPtr[5];
    
            rPtr_3[6] = rPtr[3];
    
            rPtr_3[7] = rPtr[7];
    
    delta_angle.x = __cosf(j * -0.39269909262657166f);
    delta_angle.y = __sinf(j * -0.39269909262657166f);
     
    angle.x = 1;
    angle.y = 0;
    
    shPtr[offset + 8 * (0 + (threadIdx.x / 8)) % 8] = rPtr_3[(0 + (threadIdx.x / 8)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[4];
    turboFFT_ZMUL(rPtr[4], tmp, angle);
    
    shPtr[offset + 8 * (1 + (threadIdx.x / 8)) % 8] = rPtr_3[(1 + (threadIdx.x / 8)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[2];
    turboFFT_ZMUL(rPtr[2], tmp, angle);
    
    shPtr[offset + 8 * (2 + (threadIdx.x / 8)) % 8] = rPtr_3[(2 + (threadIdx.x / 8)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[6];
    turboFFT_ZMUL(rPtr[6], tmp, angle);
    
    shPtr[offset + 8 * (3 + (threadIdx.x / 8)) % 8] = rPtr_3[(3 + (threadIdx.x / 8)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[1];
    turboFFT_ZMUL(rPtr[1], tmp, angle);
    
    shPtr[offset + 8 * (4 + (threadIdx.x / 8)) % 8] = rPtr_3[(4 + (threadIdx.x / 8)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[5];
    turboFFT_ZMUL(rPtr[5], tmp, angle);
    
    shPtr[offset + 8 * (5 + (threadIdx.x / 8)) % 8] = rPtr_3[(5 + (threadIdx.x / 8)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[3];
    turboFFT_ZMUL(rPtr[3], tmp, angle);
    
    shPtr[offset + 8 * (6 + (threadIdx.x / 8)) % 8] = rPtr_3[(6 + (threadIdx.x / 8)) % 8];
    
    tmp = angle;
    turboFFT_ZMUL(angle, tmp, delta_angle);
    tmp = rPtr[7];
    turboFFT_ZMUL(rPtr[7], tmp, angle);
    
    shPtr[offset + 8 * (7 + (threadIdx.x / 8)) % 8] = rPtr_3[(7 + (threadIdx.x / 8)) % 8];
    
    offset = 0;
    offset += tx % 16 + tx / 16 * 128;
    
    __syncthreads();
    
    rPtr[0] = shPtr[offset + 0];
    
    rPtr[1] = shPtr[offset + 16];
    
    rPtr[2] = shPtr[offset + 32];
    
    rPtr[3] = shPtr[offset + 48];
    
    rPtr[4] = shPtr[offset + 64];
    
    rPtr[5] = shPtr[offset + 80];
    
    rPtr[6] = shPtr[offset + 96];
    
    rPtr[7] = shPtr[offset + 112];
    
    tmp = rPtr[0];
    turboFFT_ZADD(rPtr[0], tmp, rPtr[4]);
    turboFFT_ZSUB(rPtr[4], tmp, rPtr[4]);
    tmp = rPtr[4];
    
    tmp = rPtr[1];
    turboFFT_ZADD(rPtr[1], tmp, rPtr[5]);
    turboFFT_ZSUB(rPtr[5], tmp, rPtr[5]);
    tmp = rPtr[5];
    
    tmp = rPtr[2];
    turboFFT_ZADD(rPtr[2], tmp, rPtr[6]);
    turboFFT_ZSUB(rPtr[6], tmp, rPtr[6]);
    tmp = rPtr[6];
    
    tmp = rPtr[3];
    turboFFT_ZADD(rPtr[3], tmp, rPtr[7]);
    turboFFT_ZSUB(rPtr[7], tmp, rPtr[7]);
    tmp = rPtr[7];
            
    bx = BID_X;
    tx = TID;
    gPtr = outputs;
    
            *(gPtr + 0) = rPtr[0];
            
            *(gPtr + 16) = rPtr[1];
            
            *(gPtr + 32) = rPtr[2];
            
            *(gPtr + 48) = rPtr[3];
            
            *(gPtr + 64) = rPtr[4];
            
            *(gPtr + 80) = rPtr[5];
            
            *(gPtr + 96) = rPtr[6];
            
            *(gPtr + 112) = rPtr[7];
}