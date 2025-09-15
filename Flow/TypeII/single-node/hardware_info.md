# ハードウェア情報 - 不老 TypeII (GPU)ノード

## 収集日時
- 2025-09-15
- ノード: cx113 (インタラクティブジョブ)

## CPU情報

### 基本仕様
- **モデル**: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
- **ソケット数**: 2
- **コア数**: 40コア (20コア × 2ソケット)
- **スレッド数**: 40 (HTT無効)
- **ベース周波数**: 2.10 GHz
- **最大周波数**: 3.90 GHz
- **L1 cache**: 32KB (d-cache), 32KB (i-cache)
- **L2 cache**: 1MB per core
- **L3 cache**: 28MB per socket

### SIMD命令セット
- AVX-512 対応 (avx512f, avx512cd, avx512bw, avx512dq, avx512vl, avx512_vnni)
- AVX2, AVX
- FMA (Fused Multiply-Add)
- SSE4.2, SSE4.1, SSE3, SSE2, SSE

### 理論演算性能 (CPU)
```
FP64 (double precision):
= 40 cores × 2.1 GHz × 2 (FMA) × 8 (AVX-512) 
= 1,344 GFLOPS

FP32 (single precision):
= 40 cores × 2.1 GHz × 2 (FMA) × 16 (AVX-512)
= 2,688 GFLOPS

最大ターボ時 (3.9 GHz):
FP64: 2,496 GFLOPS
FP32: 4,992 GFLOPS
```

## メモリ情報

### 容量
- **総メモリ**: 376 GB (384 GB実装)
- **NUMA構成**: 2ノード
  - Node 0: 191 GB (CPUs 0-19)
  - Node 1: 192 GB (CPUs 20-39)

### メモリバンド幅（理論値）
- DDR4-2933
- 6チャネル × 2ソケット = 12チャネル
- 理論バンド幅: 約 281.6 GB/s (2933 MT/s × 8 bytes × 12 channels)

## GPU情報

### 基本仕様
- **モデル**: Tesla V100-SXM2-32GB × 4
- **アーキテクチャ**: Volta (Compute Capability 7.0)
- **メモリ**: 32 GB HBM2 per GPU (総計 128 GB)
- **メモリバンド幅**: 900 GB/s per GPU
- **SMクロック**: 1530 MHz (Boost)
- **メモリクロック**: 877 MHz

### GPU間接続トポロジー
- **NVLink2**: 各GPU間は2本のNVLinkで接続
  - GPU0-GPU1: NV2 (同一NUMAノード)
  - GPU2-GPU3: NV2 (同一NUMAノード)
  - GPU0/1 - GPU2/3: NV2 (NUMA間)
- **バンド幅**: 50 GB/s (片方向) × 2本 = 100 GB/s per GPU pair

### CPU-GPU NUMA配置
- GPU0, GPU1: NUMA Node 0 (CPUs 0-19)
- GPU2, GPU3: NUMA Node 1 (CPUs 20-39)

### 理論演算性能 (GPU)

#### 単一GPU (V100)
```
FP64 (double precision):
= 7.8 TFLOPS

FP32 (single precision):
= 15.7 TFLOPS

Tensor Core FP16:
= 125 TFLOPS
```

#### 4GPU合計
```
FP64: 31.2 TFLOPS
FP32: 62.8 TFLOPS
Tensor Core FP16: 500 TFLOPS
```

## 総合理論演算性能

### 1GPU使用時
- **FP64**: 7.8 TFLOPS (GPU) + 1.34 TFLOPS (CPU) = 9.14 TFLOPS
- **FP32**: 15.7 TFLOPS (GPU) + 2.69 TFLOPS (CPU) = 18.39 TFLOPS

### 4GPU使用時
- **FP64**: 31.2 TFLOPS (GPU) + 1.34 TFLOPS (CPU) = 32.54 TFLOPS
- **FP32**: 62.8 TFLOPS (GPU) + 2.69 TFLOPS (CPU) = 65.49 TFLOPS

## 最適化のための推奨事項

### NUMA最適化
- GPU0/1使用時: `numactl --cpunodebind=0 --membind=0`
- GPU2/3使用時: `numactl --cpunodebind=1 --membind=1`
- 4GPU使用時: NVLinkを活用した通信最適化が重要

### OpenMP設定
```bash
export OMP_NUM_THREADS=10  # 1GPU当たり10コア
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

### コンパイラフラグ推奨
- Intel: `-xCORE-AVX512 -qopt-zmm-usage=high`
- GCC: `-march=skylake-avx512 -mprefer-vector-width=512`
- NVCC: `-arch=sm_70 -use_fast_math`

## 備考
- ノード共有なし（占有利用）
- 最大実行時間制限あり（リソースグループによる）
- 消費電力制限による動的周波数調整あり