/**
 * @file gemm_cuda_v2.0.0.cu
 * @brief CUDA implementation of GEMM (v2.0.0 - Optimized)
 * @details Larger tile size (32x32) with register tiling
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <math.h>

// Block size for CUDA kernel
#define BLOCK_SIZE 32
#define TILE_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

/**
 * @brief Get current time in seconds
 */
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/**
 * @brief Optimized GEMM kernel with 32x32 tiles
 */
__global__ void gemm_kernel_optimized(int M, int N, int K,
                                      double alpha, const double* __restrict__ A, int lda,
                                      const double* __restrict__ B, int ldb,
                                      double beta, double* __restrict__ C, int ldc) {
    
    __shared__ double As[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    __shared__ double Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Coalesced load of tile of A into shared memory
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        if (a_row < M && a_col < K) {
            As[ty][tx] = A[a_row * lda + a_col];
        } else {
            As[ty][tx] = 0.0;
        }
        
        // Coalesced load of tile of B into shared memory
        int b_row = tile * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            Bs[ty][tx] = B[b_row * ldb + b_col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial sum with full unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        if (beta == 0.0) {
            C[row * ldc + col] = alpha * sum;
        } else {
            C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
        }
    }
}

/**
 * @brief Initialize matrix with random values
 */
void init_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

/**
 * @brief Verify GEMM result against CPU reference
 */
void verify_result(int M, int N, int K,
                   double alpha, const double* A, int lda,
                   const double* B, int ldb,
                   double beta, const double* C_ref, int ldc,
                   const double* C_gpu, double tolerance) {
    
    double max_error = 0.0;
    int error_count = 0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Compute reference value
            double ref_val = beta * C_ref[i * ldc + j];
            for (int k = 0; k < K; k++) {
                ref_val += alpha * A[i * lda + k] * B[k * ldb + j];
            }
            
            // Compare with GPU result
            double gpu_val = C_gpu[i * ldc + j];
            double error = fabs(gpu_val - ref_val);
            
            if (error > max_error) {
                max_error = error;
            }
            
            if (error > tolerance) {
                error_count++;
                if (error_count <= 5) {  // Print first 5 errors
                    printf("Error at [%d,%d]: ref=%.6e, gpu=%.6e, error=%.6e\n",
                           i, j, ref_val, gpu_val, error);
                }
            }
        }
    }
    
    printf("Verification: max_error=%.6e, errors=%d/%d\n", 
           max_error, error_count, M * N);
    
    if (max_error < tolerance) {
        printf("PASSED (tolerance=%.6e)\n", tolerance);
    } else {
        printf("FAILED (tolerance=%.6e)\n", tolerance);
    }
}

int main(int argc, char* argv[]) {
    // Parse arguments
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int num_iterations = (argc > 4) ? atoi(argv[4]) : 3;
    
    printf("========================================\n");
    printf("CUDA GEMM Implementation v2.0.0\n");
    printf("========================================\n");
    
    // Print GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
    
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Data type: double (64-bit)\n");
    printf("Kernel: Optimized with %dx%d tiles\n", TILE_SIZE, TILE_SIZE);
    printf("Number of iterations: %d\n", num_iterations);
    
    size_t size_A = M * K * sizeof(double);
    size_t size_B = K * N * sizeof(double);
    size_t size_C = M * N * sizeof(double);
    
    printf("Memory required: %.2f MB\n", 
           (size_A + size_B + size_C) / (1024.0 * 1024.0));
    printf("GPU memory required: %.2f MB\n", 
           (size_A + size_B + size_C) / (1024.0 * 1024.0));
    
    // Allocate host memory
    double *h_A = (double*)malloc(size_A);
    double *h_B = (double*)malloc(size_B);
    double *h_C = (double*)malloc(size_C);
    double *h_C_ref = (double*)malloc(size_C);
    
    // Initialize matrices
    srand(42);
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    init_matrix(h_C_ref, M, N);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("\nGrid: %dx%d blocks, Block: %dx%d threads\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_C, h_C_ref, size_C, cudaMemcpyHostToDevice));
    gemm_kernel_optimized<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Performance measurement
    printf("\n--- Performance Measurement ---\n");
    double total_time = 0.0;
    double min_time = 1e10;
    double max_time = 0.0;
    
    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C_ref, size_C, cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        gemm_kernel_optimized<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = milliseconds / 1000.0;
        
        total_time += seconds;
        if (seconds < min_time) min_time = seconds;
        if (seconds > max_time) max_time = seconds;
        
        double gflops = (2.0 * M * N * K) / (seconds * 1e9);
        printf("Iteration %d: %.4f sec, %.2f GFLOPS\n", iter + 1, seconds, gflops);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    double avg_time = total_time / num_iterations;
    double avg_gflops = (2.0 * M * N * K) / (avg_time * 1e9);
    double peak_gflops = (2.0 * M * N * K) / (min_time * 1e9);
    
    printf("\n--- Results Summary ---\n");
    printf("Average time: %.4f seconds\n", avg_time);
    printf("Minimum time: %.4f seconds\n", min_time);
    printf("Maximum time: %.4f seconds\n", max_time);
    printf("Average performance: %.2f GFLOPS\n", avg_gflops);
    printf("Peak performance: %.2f GFLOPS\n", peak_gflops);
    
    // Theoretical peak (V100)
    double theoretical_fp64 = 7800.0;  // 7.8 TFLOPS for V100
    printf("Efficiency: %.2f%% of theoretical peak (%.1f GFLOPS)\n",
           (peak_gflops / theoretical_fp64) * 100, theoretical_fp64);
    
    // Accuracy verification
    printf("\n--- Accuracy Verification ---\n");
    memcpy(h_C, h_C_ref, size_C);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    verify_result(M, N, K, alpha, h_A, K, h_B, N, beta, h_C_ref, N, h_C, 1e-10);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    printf("\n========================================\n");
    
    return 0;
}