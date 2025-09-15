#!/bin/bash

#------ pjsub option --------#
#PJM -L rscgrp=cx-single      # Single GPU node for baseline test
#PJM -L node=1                # 1 node
#PJM -L elapse=0:10:00        # 10 minutes for baseline
#PJM -j                       # Merge stderr to stdout
#PJM -o gemm_baseline.out     # Output file name

#------- Module setup -------#
# Load appropriate modules (may need adjustment based on actual environment)
module load gcc/11.3.0
module load cuda/12.1

#------- Program execution -------#
echo "========================================" 
echo "GEMM Baseline Performance Test"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Compile the baseline code
echo ""
echo "Compiling baseline code..."
make clean
make gemm_base

# Run tests with different matrix sizes
echo ""
echo "Running baseline tests..."

# Small test for validation
echo "--- Small matrix test (512x512x512) ---"
./gemm_base 512 512 512 5

# Medium test
echo ""
echo "--- Medium matrix test (1024x1024x1024) ---"
./gemm_base 1024 1024 1024 5

# Large test
echo ""
echo "--- Large matrix test (2048x2048x2048) ---"
./gemm_base 2048 2048 2048 3

# Very large test (if memory permits)
echo ""
echo "--- Very large matrix test (4096x4096x4096) ---"
./gemm_base 4096 4096 4096 2

echo ""
echo "========================================"
echo "Baseline tests completed"
echo "========================================"