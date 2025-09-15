/**
 * @file gemm_base.c
 * @brief Basic implementation of General Matrix Multiplication (GEMM)
 * @details Double precision matrix multiplication: C = alpha * A * B + beta * C
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

// Matrix size definitions
#define DEFAULT_M 1024
#define DEFAULT_N 1024
#define DEFAULT_K 1024

// Performance calculation macros
#define GFLOPS(m, n, k, time) ((2.0 * (double)(m) * (double)(n) * (double)(k)) / ((time) * 1e9))

/**
 * @brief Get current time in seconds
 */
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/**
 * @brief Initialize matrix with random values
 */
void init_matrix(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;  // Random values between -1 and 1
    }
}

/**
 * @brief Initialize matrix with specific pattern for validation
 */
void init_matrix_pattern(double* matrix, int rows, int cols, double value) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = value;
    }
}

/**
 * @brief Basic GEMM implementation (naive triple loop)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param alpha Scalar multiplier for A*B
 * @param A Matrix A (M x K)
 * @param lda Leading dimension of A
 * @param B Matrix B (K x N)
 * @param ldb Leading dimension of B
 * @param beta Scalar multiplier for C
 * @param C Matrix C (M x N)
 * @param ldc Leading dimension of C
 */
void gemm_baseline(int M, int N, int K,
                   double alpha, const double* A, int lda,
                   const double* B, int ldb,
                   double beta, double* C, int ldc) {
    
    // Scale C by beta
    if (beta != 1.0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    // Compute C += alpha * A * B
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

/**
 * @brief Verify GEMM result by computing error
 * @return Maximum absolute error
 */
double verify_gemm(int M, int N, int K,
                   double alpha, const double* A, int lda,
                   const double* B, int ldb,
                   double beta, const double* C_ref, int ldc,
                   const double* C_test) {
    
    double max_error = 0.0;
    double* C_verify = (double*)malloc(M * N * sizeof(double));
    memcpy(C_verify, C_ref, M * N * sizeof(double));
    
    // Compute reference result
    gemm_baseline(M, N, K, alpha, A, lda, B, ldb, beta, C_verify, ldc);
    
    // Compare results
    for (int i = 0; i < M * N; i++) {
        double error = fabs(C_verify[i] - C_test[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    
    free(C_verify);
    return max_error;
}

/**
 * @brief Compute Frobenius norm of error
 */
double compute_frobenius_error(const double* C_ref, const double* C_test, int M, int N) {
    double sum = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = C_ref[i] - C_test[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    // Parse command line arguments
    int M = (argc > 1) ? atoi(argv[1]) : DEFAULT_M;
    int N = (argc > 2) ? atoi(argv[2]) : DEFAULT_N;
    int K = (argc > 3) ? atoi(argv[3]) : DEFAULT_K;
    int num_iterations = (argc > 4) ? atoi(argv[4]) : 5;
    
    printf("========================================\n");
    printf("GEMM Baseline Implementation\n");
    printf("========================================\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Data type: double (64-bit)\n");
    printf("Number of iterations: %d\n", num_iterations);
    printf("Memory required: %.2f MB\n", 
           (M*K + K*N + M*N) * sizeof(double) / (1024.0 * 1024.0));
    
    // Allocate matrices
    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C = (double*)malloc(M * N * sizeof(double));
    double* C_ref = (double*)malloc(M * N * sizeof(double));
    
    if (!A || !B || !C || !C_ref) {
        printf("Error: Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices
    srand(42);  // Fixed seed for reproducibility
    init_matrix(A, M, K);
    init_matrix(B, K, N);
    init_matrix(C_ref, M, N);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // Warmup run
    memcpy(C, C_ref, M * N * sizeof(double));
    gemm_baseline(M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // Performance measurement
    double total_time = 0.0;
    double min_time = 1e10;
    double max_time = 0.0;
    
    printf("\n--- Performance Measurement ---\n");
    for (int iter = 0; iter < num_iterations; iter++) {
        memcpy(C, C_ref, M * N * sizeof(double));
        
        double start = get_time();
        gemm_baseline(M, N, K, alpha, A, K, B, N, beta, C, N);
        double end = get_time();
        
        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
        
        printf("Iteration %d: %.4f sec, %.2f GFLOPS\n", 
               iter + 1, elapsed, GFLOPS(M, N, K, elapsed));
    }
    
    double avg_time = total_time / num_iterations;
    
    // Results
    printf("\n--- Results Summary ---\n");
    printf("Average time: %.4f seconds\n", avg_time);
    printf("Minimum time: %.4f seconds\n", min_time);
    printf("Maximum time: %.4f seconds\n", max_time);
    printf("Average performance: %.2f GFLOPS\n", GFLOPS(M, N, K, avg_time));
    printf("Peak performance: %.2f GFLOPS\n", GFLOPS(M, N, K, min_time));
    
    // Accuracy verification
    printf("\n--- Accuracy Verification ---\n");
    memcpy(C, C_ref, M * N * sizeof(double));
    gemm_baseline(M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // Simple validation with known pattern
    double* A_test = (double*)malloc(M * K * sizeof(double));
    double* B_test = (double*)malloc(K * N * sizeof(double));
    double* C_test = (double*)malloc(M * N * sizeof(double));
    
    init_matrix_pattern(A_test, M, K, 1.0);
    init_matrix_pattern(B_test, K, N, 1.0);
    init_matrix_pattern(C_test, M, N, 0.0);
    
    gemm_baseline(M, N, K, 1.0, A_test, K, B_test, N, 0.0, C_test, N);
    
    // Expected value: each element should be K
    double expected = (double)K;
    double max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        double error = fabs(C_test[i] - expected);
        if (error > max_error) max_error = error;
    }
    
    printf("Validation test (all 1s): ");
    if (max_error < 1e-10) {
        printf("PASSED (max error: %.2e)\n", max_error);
    } else {
        printf("FAILED (max error: %.2e)\n", max_error);
    }
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(A_test);
    free(B_test);
    free(C_test);
    
    printf("\n========================================\n");
    
    return 0;
}