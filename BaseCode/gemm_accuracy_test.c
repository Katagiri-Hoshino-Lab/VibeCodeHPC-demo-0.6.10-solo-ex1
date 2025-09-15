/**
 * @file gemm_accuracy_test.c
 * @brief Accuracy testing routines for GEMM implementations
 * @details Provides comprehensive accuracy verification for GEMM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Tolerance levels for double precision
#define TOLERANCE_STRICT   1e-14  // Machine epsilon level
#define TOLERANCE_NORMAL   1e-12  // Standard numerical computation
#define TOLERANCE_RELAXED  1e-10  // For accumulated errors

// Matrix element access macro
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/**
 * @brief Reference GEMM implementation for verification
 */
void gemm_reference(int M, int N, int K,
                    double alpha, const double* A, int lda,
                    const double* B, int ldb,
                    double beta, double* C, int ldc) {
    
    // Scale C by beta
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] *= beta;
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
 * @brief Initialize matrix with deterministic pattern
 */
void init_pattern_matrix(double* matrix, int rows, int cols, int pattern) {
    switch(pattern) {
        case 0:  // Identity matrix
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i * cols + j] = (i == j) ? 1.0 : 0.0;
                }
            }
            break;
        case 1:  // Sequential values
            for (int i = 0; i < rows * cols; i++) {
                matrix[i] = (double)(i + 1);
            }
            break;
        case 2:  // Checkerboard pattern
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i * cols + j] = ((i + j) % 2) ? 1.0 : -1.0;
                }
            }
            break;
        case 3:  // Random with fixed seed
            srand(12345);
            for (int i = 0; i < rows * cols; i++) {
                matrix[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            }
            break;
        case 4:  // Small integers for exact arithmetic
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    matrix[i * cols + j] = (double)((i + j) % 10 - 5);
                }
            }
            break;
    }
}

/**
 * @brief Compute relative error
 */
double compute_relative_error(double ref, double test) {
    if (fabs(ref) < DBL_EPSILON) {
        return fabs(test);
    }
    return fabs((test - ref) / ref);
}

/**
 * @brief Compute Frobenius norm
 */
double frobenius_norm(const double* matrix, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        sum += matrix[i] * matrix[i];
    }
    return sqrt(sum);
}

/**
 * @brief Compute relative Frobenius norm error
 */
double relative_frobenius_error(const double* C_ref, const double* C_test, 
                                int M, int N) {
    double error_norm = 0.0;
    double ref_norm = 0.0;
    
    for (int i = 0; i < M * N; i++) {
        double diff = C_test[i] - C_ref[i];
        error_norm += diff * diff;
        ref_norm += C_ref[i] * C_ref[i];
    }
    
    if (ref_norm < DBL_EPSILON) {
        return sqrt(error_norm);
    }
    return sqrt(error_norm / ref_norm);
}

/**
 * @brief Detailed error analysis
 */
typedef struct {
    double max_abs_error;
    double max_rel_error;
    double avg_abs_error;
    double avg_rel_error;
    double frobenius_error;
    double rel_frobenius_error;
    int max_error_i;
    int max_error_j;
    int num_large_errors;  // Count of errors > tolerance
} ErrorStats;

ErrorStats analyze_error(const double* C_ref, const double* C_test,
                         int M, int N, double tolerance) {
    ErrorStats stats = {0};
    double sum_abs = 0.0;
    double sum_rel = 0.0;
    int count = 0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            double abs_err = fabs(C_test[idx] - C_ref[idx]);
            double rel_err = compute_relative_error(C_ref[idx], C_test[idx]);
            
            sum_abs += abs_err;
            sum_rel += rel_err;
            count++;
            
            if (abs_err > stats.max_abs_error) {
                stats.max_abs_error = abs_err;
                stats.max_error_i = i;
                stats.max_error_j = j;
            }
            
            if (rel_err > stats.max_rel_error) {
                stats.max_rel_error = rel_err;
            }
            
            if (abs_err > tolerance) {
                stats.num_large_errors++;
            }
        }
    }
    
    stats.avg_abs_error = sum_abs / count;
    stats.avg_rel_error = sum_rel / count;
    stats.frobenius_error = frobenius_norm(C_test, M, N) - frobenius_norm(C_ref, M, N);
    stats.rel_frobenius_error = relative_frobenius_error(C_ref, C_test, M, N);
    
    return stats;
}

/**
 * @brief Test GEMM with specific test case
 */
int test_gemm_case(const char* test_name, int M, int N, int K,
                   double alpha, double beta, int pattern,
                   void (*gemm_func)(int, int, int, double, const double*, int,
                                    const double*, int, double, double*, int),
                   double tolerance) {
    
    printf("\n=== Test Case: %s ===\n", test_name);
    printf("Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Parameters: alpha=%.2f, beta=%.2f\n", alpha, beta);
    printf("Pattern: %d, Tolerance: %.2e\n", pattern, tolerance);
    
    // Allocate matrices
    double* A = (double*)calloc(M * K, sizeof(double));
    double* B = (double*)calloc(K * N, sizeof(double));
    double* C_ref = (double*)calloc(M * N, sizeof(double));
    double* C_test = (double*)calloc(M * N, sizeof(double));
    double* C_init = (double*)calloc(M * N, sizeof(double));
    
    // Initialize matrices
    init_pattern_matrix(A, M, K, pattern);
    init_pattern_matrix(B, K, N, (pattern + 1) % 5);
    init_pattern_matrix(C_init, M, N, (pattern + 2) % 5);
    
    // Copy initial C
    memcpy(C_ref, C_init, M * N * sizeof(double));
    memcpy(C_test, C_init, M * N * sizeof(double));
    
    // Compute reference
    gemm_reference(M, N, K, alpha, A, K, B, N, beta, C_ref, N);
    
    // Compute test
    gemm_func(M, N, K, alpha, A, K, B, N, beta, C_test, N);
    
    // Analyze error
    ErrorStats stats = analyze_error(C_ref, C_test, M, N, tolerance);
    
    // Print results
    printf("Results:\n");
    printf("  Max absolute error: %.2e at [%d,%d]\n", 
           stats.max_abs_error, stats.max_error_i, stats.max_error_j);
    printf("  Max relative error: %.2e\n", stats.max_rel_error);
    printf("  Avg absolute error: %.2e\n", stats.avg_abs_error);
    printf("  Avg relative error: %.2e\n", stats.avg_rel_error);
    printf("  Relative Frobenius error: %.2e\n", stats.rel_frobenius_error);
    printf("  Elements exceeding tolerance: %d / %d\n", 
           stats.num_large_errors, M * N);
    
    // Determine pass/fail
    int passed = (stats.max_abs_error <= tolerance) && 
                 (stats.num_large_errors == 0);
    
    if (passed) {
        printf("Status: PASSED ✓\n");
    } else {
        printf("Status: FAILED ✗\n");
        
        // Print some failing elements for debugging
        if (stats.num_large_errors > 0 && stats.num_large_errors <= 10) {
            printf("Failed elements:\n");
            for (int i = 0; i < M && i < 5; i++) {
                for (int j = 0; j < N && j < 5; j++) {
                    int idx = i * N + j;
                    double err = fabs(C_test[idx] - C_ref[idx]);
                    if (err > tolerance) {
                        printf("  [%d,%d]: ref=%.6e, test=%.6e, err=%.2e\n",
                               i, j, C_ref[idx], C_test[idx], err);
                    }
                }
            }
        }
    }
    
    // Cleanup
    free(A);
    free(B);
    free(C_ref);
    free(C_test);
    free(C_init);
    
    return passed ? 0 : 1;
}

/**
 * @brief Run comprehensive accuracy test suite
 */
int run_accuracy_test_suite(void (*gemm_func)(int, int, int, double, const double*, int,
                                              const double*, int, double, double*, int)) {
    int total_tests = 0;
    int failed_tests = 0;
    
    printf("========================================\n");
    printf("GEMM Accuracy Test Suite\n");
    printf("========================================\n");
    printf("Tolerance levels:\n");
    printf("  Strict:  %.2e\n", TOLERANCE_STRICT);
    printf("  Normal:  %.2e\n", TOLERANCE_NORMAL);
    printf("  Relaxed: %.2e\n", TOLERANCE_RELAXED);
    
    // Test 1: Small matrices with identity
    total_tests++;
    failed_tests += test_gemm_case("Identity Test", 
                                   8, 8, 8, 1.0, 0.0, 0,
                                   gemm_func, TOLERANCE_STRICT);
    
    // Test 2: Square matrices
    total_tests++;
    failed_tests += test_gemm_case("Square Matrices", 
                                   64, 64, 64, 2.0, 1.0, 3,
                                   gemm_func, TOLERANCE_NORMAL);
    
    // Test 3: Rectangular matrices (M > N)
    total_tests++;
    failed_tests += test_gemm_case("Tall Matrix", 
                                   128, 64, 32, 1.0, 0.5, 1,
                                   gemm_func, TOLERANCE_NORMAL);
    
    // Test 4: Rectangular matrices (M < N)
    total_tests++;
    failed_tests += test_gemm_case("Wide Matrix", 
                                   64, 128, 32, -1.0, 2.0, 2,
                                   gemm_func, TOLERANCE_NORMAL);
    
    // Test 5: Large matrices
    total_tests++;
    failed_tests += test_gemm_case("Large Matrices", 
                                   512, 512, 512, 1.0, 0.0, 3,
                                   gemm_func, TOLERANCE_RELAXED);
    
    // Test 6: Alpha = 0 case
    total_tests++;
    failed_tests += test_gemm_case("Alpha Zero", 
                                   32, 32, 32, 0.0, 1.0, 4,
                                   gemm_func, TOLERANCE_STRICT);
    
    // Test 7: Beta = 0 case
    total_tests++;
    failed_tests += test_gemm_case("Beta Zero", 
                                   32, 32, 32, 1.0, 0.0, 4,
                                   gemm_func, TOLERANCE_NORMAL);
    
    // Test 8: Small integer test (exact arithmetic)
    total_tests++;
    failed_tests += test_gemm_case("Integer Arithmetic", 
                                   16, 16, 16, 1.0, 1.0, 4,
                                   gemm_func, TOLERANCE_STRICT);
    
    // Summary
    printf("\n========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed: %d\n", total_tests - failed_tests);
    printf("Failed: %d\n", failed_tests);
    
    if (failed_tests == 0) {
        printf("\n✓ All accuracy tests PASSED\n");
        printf("The implementation meets BLAS-level precision requirements.\n");
    } else {
        printf("\n✗ Some tests FAILED\n");
        printf("The implementation needs accuracy improvements.\n");
    }
    
    return failed_tests;
}

// External declaration for the function to test
extern void gemm_baseline(int M, int N, int K,
                          double alpha, const double* A, int lda,
                          const double* B, int ldb,
                          double beta, double* C, int ldc);

/**
 * @brief Main function for standalone testing
 */
int main(int argc, char* argv[]) {
    // Run the test suite on baseline implementation
    int result = run_accuracy_test_suite(gemm_baseline);
    
    return result > 0 ? 1 : 0;
}