# TurboFNO

TurboFNO is a high-performance, fault-tolerant implementation of the Fourier Neural Operator (FNO), optimized for GPUs through FFT–GEMM–iFFT kernel fusion. It supports progressive kernel fusion strategies and is benchmark-ready on both 1D and 2D FNO workloads.

---

## 📦 Repository Structure

```
TurboFNO/
├── fusion_variants/           # All kernel fusion variants (stepwise E→A→B→C→D for 1D/2D)
├── benchmark_config/          # Input problem sizes for 1D and 2D
├── TurboFFT/                  # Git submodule (TurboFNO_dev branch)
├── utils/, Common/            # Shared code and support modules
├── install.sh                 # Batch compile and PATH setup script
└── README.md
```

---

## 1. Compilation

Before building, make sure to set the project root path:

```bash
export PROJECT_ROOT=$(pwd)
```

To compile all kernel fusion variants (1D and 2D):

```bash
cd fusion_variants
bash install.sh
```

> 🧹 To clean up all builds:
>
> ```bash
> bash install.sh uninstall
> ```

---

## 2. Run

Each variant builds a `TurboFNO` binary that accepts problem size configurations via a **runtime `.txt` config file** (no recompilation needed).

### a. Input Format (via `benchmark_config/problem_size_1d.txt` or `problem_size_2d.txt`)

```txt
bs_list = 1 2 4 8 16 32 64
dimX_list = 1
DY_list = 128 256
N_list = 64 128
K_list = 8 16 24 32
```

### b. Launching a Variant

```bash
cd fusion_variants/1D_B_exp_fused_fft_cgemm+ifft/build
./TurboFNO ../../../benchmark_config/problem_size_1d.txt
```

> ⚠️ If no config path is provided, a default path is compiled in via CMake.

---

## 3. Data Type and FFT Settings

* Only **FP32** is supported
* FFT type: **Complex-to-Complex (C2C)**

---

## 4. Kernel Customization

### 🧠 FFT Kernel Code Generation

FFT kernels are auto-generated. You can customize templates in:

```
TurboFFT/TurboFFT/include/code_gen/generated/...
```

Changes require a rebuild of the corresponding variant.

### ⚙️ GEMM Tiling Configuration

Tuning parameters (e.g., tile sizes, threads per block) are set in:

```
utils/TurboFNO.h
```

These control shared memory layout, tiling, and warp fusion strategies.

---

## 5. Submodule Setup

This project depends on the [`TurboFFT`](https://github.com/shixun404/TurboFFT) repo (branch `TurboFNO_dev`). Make sure to initialize submodules:

```bash
git submodule update --init --recursive
```

---

## 6. Variant Summary (Progressive Kernel Fusion)

| Variant | Fusion Strategy                      | Description                    |
| ------- | ------------------------------------ | ------------------------------ |
| E       | No fusion                            | Baseline                       |
| A       | FFT + GEMM + iFFT (separate kernels) | Initial kernel sequence        |
| B       | Fused FFT + GEMM                     | First-stage fusion             |
| C       | FFT + Fused GEMM + iFFT              | Mid-stage fusion               |
| D       | Fully fused FFT + GEMM + iFFT        | Final optimized implementation |

---

## 📖 Citation

If you use TurboFNO in your work, please cite:

```bibtex
@article{wu2025turbofno,
  title={TurboFNO: High-Performance Fourier Neural Operator with Fused FFT-GEMM-iFFT on GPU},
  author={Wu, Shixun and Zhai, Yujia and Dai, Huangliang and Zhao, Hairui and Zhu, Yue and Hu, Haiyang and Chen, Zizhong},
  journal={arXiv preprint arXiv:2504.11681},
  year={2025}
}
```
