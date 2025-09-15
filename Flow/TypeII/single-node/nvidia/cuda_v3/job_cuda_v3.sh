#!/bin/bash

#------ pjsub option --------#
#PJM -L rscgrp=cx-small       # Small GPU job
#PJM -L node=1                # 1 node (4 GPUs)
#PJM -L elapse=0:10:00        # 10 minutes
#PJM -j                       # Merge stderr to stdout
#PJM -o cuda_v3.out          # Output file

#------- Module setup -------#
module load gcc/11.3.0
module load cuda/12.1.1

#------- Program execution -------#
echo "========================================"
echo "CUDA GEMM v3.0.0 Performance Test (cuBLAS)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# GPU information
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Compile CUDA code
echo ""
echo "Compiling CUDA code with cuBLAS..."
nvcc -O3 -arch=sm_70 -lcublas -o gemm_cuda_v3.0.0 gemm_cuda_v3.0.0.cu

# Run tests with different matrix sizes
echo ""
echo "Running performance tests..."

# Small test
echo "--- Small matrix test (512x512x512) ---"
./gemm_cuda_v3.0.0 512 512 512 5

# Medium test
echo ""
echo "--- Medium matrix test (1024x1024x1024) ---"
./gemm_cuda_v3.0.0 1024 1024 1024 5

# Large test
echo ""
echo "--- Large matrix test (2048x2048x2048) ---"
./gemm_cuda_v3.0.0 2048 2048 2048 3

# Very large test
echo ""
echo "--- Very large matrix test (4096x4096x4096) ---"
./gemm_cuda_v3.0.0 4096 4096 4096 2

# Extra large test (if memory permits)
echo ""
echo "--- Extra large matrix test (8192x8192x8192) ---"
./gemm_cuda_v3.0.0 8192 8192 8192 1

echo ""
echo "========================================"
echo "Tests completed"
echo "========================================"