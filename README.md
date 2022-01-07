# course project: parallel graph layout algorithm on GPU

## directories and usage

scripts

- `test_cpu.fish`: run once verification with `data/test`.
- `test_gpu.fish`: run once verification with `data/test`.
- `all.fish`: run a full-benchmark.
- `build.fish`: compile the project.

environment variables

- `OMP_NUM_THREADS`: number of CPU threads used by OpenMP version.
- `CUDA_THREADS_PER_BLOCK`: threads per block used by CUDA version.

## backgrounds

## algorithm design

## testcase

## performance

## reference

- [Multi-Level Graph Layout on the GPU](https://ieeexplore.ieee.org/abstract/document/4376155)
