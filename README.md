# course project: parallel graph layout algorithm on GPU

## directories and usage

scripts

- `test_cpu.fish`: run once verification with `data/test`.
- `test_gpu.fish`: run once verification with `data/test`.
- the above two testing scripts will run graph layout algorithm with `data/test` as input and give a `data/image.png` as output
- `all.fish`: run a full-benchmark.
- `build.fish`: compile the project.

environment variables

- `OMP_NUM_THREADS`: number of CPU threads used by OpenMP version.
- `CUDA_THREADS_PER_BLOCK`: threads per block used by CUDA version.

## backgrounds

## algorithm design



## testcase

On GPU, we launch bunch of blocks of `128 x 8` threads.  
We tested for the best block configuration.

```plaintext
BLOCK TIME
128,8 545.86
64,16 778.63
32,32 1204.94
256,4 563.73
512,2 571.94
```

## performance

### GPU

```plaintext
data/clique100 41.07
data/clique200 97.23
data/clique300 182.64
data/clique400 340.9
data/clique500 546.08
data/tree100 31.8
data/tree200 47.39
data/tree300 60.33
data/tree400 89.59
data/tree500 106.75
data/soc-wiki-Vote 259.75
data/fb-pages-tvshow 4228.8
```

### CPU

## reference

1. [Multi-Level Graph Layout on the GPU](https://ieeexplore.ieee.org/abstract/document/4376155)