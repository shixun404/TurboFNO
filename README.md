# TurboFNO

TurboFNO is a high-performance, fault-tolerant implementation of the Fourier Neural Operator (FNO), optimized for GPUs through FFT-GEMM-iFFT kernel fusion.

## 1. Compilation

```bash
make
```

> 🔧 Please ensure you have CUDA 12.3+ and a compatible GPU (e.g., A100, H100) installed.

## 2. Run

### a. Input Format

TurboFNO currently supports **2D problems** with the following input format:

```txt
BS DimX DimY dimX dimY K N
```

* `BS`: Batch size
* `DimX`, `DimY`: Resolution of the input grid
* `dimX`, `dimY`: Number of modes to keep in Fourier domain
* `K`: Number of Fourier layers
* `N`: Channel dimension per layer

Example:

```bash
./bin/turbo_fno 8 64 64 20 20 4 32
```

### b. Data Type

* Only **FP32** is supported.
* FFT type: **Complex-to-Complex (C2C)**

## 3. Notes on Kernel Customization

### FFT Kernel Code Generation

FFT kernel templates are generated via a lightweight code generation engine. You may customize or extend the template behavior by modifying:

```
src/kernels/fft_generator.cu
```

Recompile after changes to regenerate the kernels.

### GEMM Tiling Configuration

You can configure the tiling and performance tuning parameters for the fused GEMM kernel in:

```
include/utils/TurboFNO.h
```

Key parameters include:

* Tile sizes for M/N/K dimensions
* Number of threads per block
* Warp-level memory layout options