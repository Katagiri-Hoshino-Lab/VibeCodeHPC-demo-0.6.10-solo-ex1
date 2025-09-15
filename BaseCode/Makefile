# Makefile for GEMM implementations
# Default compiler
CC = gcc
NVCC = nvcc

# Compiler flags
CFLAGS = -O3 -march=native -fopenmp -Wall
NVCCFLAGS = -O3 -arch=sm_70 -Xcompiler -fopenmp

# Libraries
LIBS = -lm

# Targets
TARGETS = gemm_base

# Default target
all: $(TARGETS)

# Build baseline GEMM
gemm_base: gemm_base.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# Build with debug symbols
debug: CFLAGS = -g -O0 -Wall
debug: $(TARGETS)

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o

# Run baseline test
test: gemm_base
	./gemm_base 512 512 512 3

# Run performance test
perf: gemm_base
	./gemm_base 2048 2048 2048 5

.PHONY: all clean test perf debug