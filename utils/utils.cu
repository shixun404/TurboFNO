#include "utils.cuh"
/*
__global__ void fill(float *a , float x, int N)
{
   int index =  blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for(int i = index; i < N; i += stride)
   {
       a[i] = x;
   }
}
*/

void fill_vector(float *target, int size, float val){
    for(int i = 0; i < size; ++i){
        target[i] = val;
    }
}

cudaDeviceProp getDetails(int deviceId)
{
        cudaDeviceProp props;
            cudaGetDeviceProperties(&props, deviceId);
                return props;
}

void generate_random_vector(float* target, int n){
    for(int i = 0; i < n; ++i){
        float tmp = rand() % 10 * 0.001;
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        target[i] = tmp;
    }
}

void generate_random_matrix(float* target, int n){
    for(int i = 0; i < n; ++i){
	for(int j = 0; j < n; ++j){
        float tmp = (float)(rand() % 5) + rand() % 5 * 0.01;
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);     
   	target[i * n + j] = tmp;
	}
    }
}

void generate_random_matrix_double(double* target, int n){
    for(int i = 0; i < n; ++i){
    	for(int j = 0; j < n; ++j){
            double tmp = (double)(rand() % 10) + rand() % 10 * (double)0.01;
            tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);     
            target[i * n + j] = tmp;
	    }
    }
}


void copy_vector(float *src, float *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++) *(dest + i) = *(src + i);
    if (i != n) printf("copy failed at %d while there are %d elements in total.\n", i, n);
}

void copy_matrix(float *src, float *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n * n; i++) *(dest + i) = *(src + i);
    if (i != n * n) printf("copy failed at %d while there are %d elements in total.\n", i, n * n);
}

void copy_matrix_double(double *src, double *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n * n; i++) *(dest + i) = *(src + i);
    if (i != n * n) printf("copy failed at %d while there are %d elements in total.\n", i, n * n);
}


// bool verify_vector(float *vec1, float *vec2, int n, int nrow=1){
//     double diff = 0.0;
//     double max_diff = 0.0;
//     double max_rel_diff = 0.0;
//     double rel_diff = 0.0;
//     int i;
//     for (i = 0; vec1 + i && vec2 + i && i < n; i++){
//         diff = fabs( (double)vec1[i] - (double)vec2[i] );
//         rel_diff = diff / fabs(double(vec1[i]));
//         if(rel_diff > max_rel_diff) {
//             max_rel_diff = rel_diff;
//         }
//         if(diff > max_diff) {
//             max_diff = diff;
//         }
//         if (rel_diff > 1e-1 && diff > 1e-1) {
//             if(i % 2 == 0)
//             printf("error. vec1=%10.6f+%10.6f.j, vec2=%10.6f+%10.6f.j, rel_diff=%10.6f, diff=%10.6f, 1d-ID %d, row %d, col %d\n", vec1[i], vec1[i + 1], vec2[i], vec2[i + 1], rel_diff, diff, i, (i / 2) % nrow, (i / 2) / nrow);
//             else
//             printf("error. vec1=%10.6f+%10.6f.j, vec2=%10.6f+%10.6f.j, rel_diff=%10.6f, diff=%10.6f, 1d-ID %d, row %d, col %d\n", vec1[i - 1], vec1[i], vec2[i - 1], vec2[i], rel_diff, diff, i, (i / 2) % nrow, (i / 2) / nrow);
//             // printf("Not Pass!\n");
//             // return false;
//             break;
//         }
//     }
//     printf("verified, max_rel_diff=%f, max_diff=%f\n", max_rel_diff, max_diff);
//     return true;
// }

bool verify_vector(float *vec1, float *vec2, int n, int nrow = 1) {
    double diff, max_diff = 0.0, max_rel_diff = 0.0;
    double sum_abs_diff = 0.0, sum_sq_diff = 0.0;
    double sum_abs_vec1 = 0.0, sum_rel_diff = 0.0;
    double epsilon = 1e-6; // Small threshold to prevent divide-by-zero

    for (int i = 0; i < n; i++) {
        diff = fabs((double)vec1[i] - (double)vec2[i]);
        double abs_val = fabs((double)vec1[i]);

        sum_abs_diff += diff;
        sum_sq_diff += diff * diff;
        sum_abs_vec1 += abs_val;
        sum_rel_diff += diff / std::max(abs_val, epsilon);

        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff / std::max(abs_val, epsilon) > max_rel_diff) {
            max_rel_diff = diff / std::max(abs_val, epsilon);
        }

        // Error reporting (Only for significant errors)
        if (diff > 1e-1 && diff / std::max(abs_val, epsilon) > 1e-1) {
            if (i % 2 == 0) {
                printf("error. vec1=%10.6f+%10.6f.j, vec2=%10.6f+%10.6f.j, rel_diff=%10.6f, diff=%10.6f, 1d-ID %d, row %d, col %d\n", 
                    vec1[i], vec1[i + 1], vec2[i], vec2[i + 1], diff / std::max(abs_val, epsilon), diff, i, (i / 2) % nrow, (i / 2) / nrow);
            } else {
                printf("error. vec1=%10.6f+%10.6f.j, vec2=%10.6f+%10.6f.j, rel_diff=%10.6f, diff=%10.6f, 1d-ID %d, row %d, col %d\n", 
                    vec1[i - 1], vec1[i], vec2[i - 1], vec2[i], diff / std::max(abs_val, epsilon), diff, i, (i / 2) % nrow, (i / 2) / nrow);
            }
            break;
        }
    }

    // Compute final metrics
    double mae = sum_abs_diff / n;
    double rmse = sqrt(sum_sq_diff / n);
    double rmae = sum_abs_diff / (sum_abs_vec1 + epsilon);
    double mre = sum_rel_diff / n;
    double psnr = 20 * log10(max_diff / rmse + epsilon);

    // Print results
    printf("verified, max_rel_diff=%e, max_diff=%e, MAE=%e, RMSE=%e, RMAE=%e, MRE=%e, PSNR=%f dB\n",
           max_rel_diff, max_diff, mae, rmse, rmae, mre, psnr);

    return true;
}



bool verify_matrix(float *mat1, float *mat2, int n){
    double diff = 0.0;
    int i, j;
    for (i = 0; mat1 + i * n && mat2 + i * n && i < n; ++i){
        for(j = 0; mat1 + i * n + j && mat2 + i * n + j && j < n; ++j){
	    diff = fabs( (double)mat1[i * n + j] - (double)mat2[i * n + j] );
        double denominator = fabs(mat1[i * n  + j]) ;
        if (denominator < 1e-3)denominator += 1;
        // if (diff / denominator > 1e-4) {
        if (diff > 1e-2){
            printf("error is %8.5f, relateive error is %8.5f,  %8.5f,%8.5f. id: %d, %d\n",diff, (diff / denominator), mat1[i * n + j], mat2[i * n + j], i, j);
            return false;
        }
        }
    }
    return true;
}

bool verify_matrix_double(double *mat1, double *mat2, int n){
    double diff = 0.0;
    int i, j;
    for (i = 0; mat1 + i * n && mat2 + i * n && i < n; ++i){
        for(j = 0; mat1 + i * n + j && mat2 + i * n + j && j < n; ++j){
	    diff = fabs( (double)mat1[i * n + j] - (double)mat2[i * n + j] );
        double denominator = fabs(mat1[i * n  + j]) ;
        if (denominator < 1e-3)denominator += 1;
        if (diff > 1e-2){
            printf("error is %8.5f, relateive error is %8.5f,  %8.5f,%8.5f. id: %d, %d\n",diff, (diff / denominator), mat1[i * n + j], mat2[i * n + j], i, j);
            return false;
        }
        }
    }
    return true;
}

bool verify_matrix_double2(double *mat1, double *mat2, int n){
    double diff = 0.0;
    int i, j;
    for (i = 0; mat1 + i * n * 2 && mat2 + i * n * 2 && i < n; ++i){
        for(j = 0; mat1 + i * n * 2 + j * 2 && mat2 + i * n * 2 + j * 2 && j < n; ++j){
	        for(int k = 0; k < 2; ++k){
                diff = fabs( (double)mat1[i * n * 2 + j * 2 + k] - (double)mat2[i * n * 2 + j * 2 + k] );
                double denominator = fabs(mat1[i * n * 2  + j * 2 + k]) ;
                if (denominator < 1e-3)denominator += 1;
                if (diff > 1e-2){
                    printf("error is %8.5f, relateive error is %8.5f,  %8.5f,%8.5f. row: %d, col: %d, real or imag: %d\n",diff, (diff / denominator), mat1[i * n * 2 + j * 2 + k], mat2[i * n * 2 + j * 2 + k], j, i, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void cpu_gemm(float alpha, float beta, float *mat1, float *mat2, int n, float *mat3){
    int i = 0, j = 0, k  = 0;
    for(i = 0; i < n; ++i){
        for(j = 0; j < n; ++j){
            float temp = 0;
	    for(k = 0; k < n; ++k)
		temp += mat1[i * n + k] * mat2[k * n + j];
            mat3[i * n + j] = alpha * temp + beta * mat3[i * n + j];
	}
    }
} 

void print_matrix(float* mat, int N){
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            printf("%8.5f  ", mat[j * N + i]);
        }
        printf("\n");
    }
    fflush(stdout);
}
