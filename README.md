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

## reference

1. [Multi-Level Graph Layout on the GPU](https://ieeexplore.ieee.org/abstract/document/4376155)

## appendix

### performance testing result

#### GPU

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

#### CPU

```plaintext
data/clique100 1 243.12
data/clique100 2 144.17
data/clique100 4 93.52
data/clique100 6 78.69
data/clique100 8 70.06
data/clique100 10 74.17
data/clique100 12 73.46
data/clique100 14 71.44
data/clique100 16 72.44
data/clique200 1 824.61
data/clique200 2 483.37
data/clique200 4 294.95
data/clique200 6 223.3
data/clique200 8 188.67
data/clique200 10 215.27
data/clique200 12 205.24
data/clique200 14 208.05
data/clique200 16 191.4
data/clique300 1 1725.44
data/clique300 2 987.06
data/clique300 4 578.49
data/clique300 6 451.03
data/clique300 8 371.39
data/clique300 10 432.27
data/clique300 12 412.53
data/clique300 14 390.89
data/clique300 16 373.38
data/clique400 1 2992.93
data/clique400 2 1676.04
data/clique400 4 959.73
data/clique400 6 723.91
data/clique400 8 607.95
data/clique400 10 640.96
data/clique400 12 668.36
data/clique400 14 633.19
data/clique400 16 603.78
data/clique500 1 4614.78
data/clique500 2 2544.45
data/clique500 4 1437.81
data/clique500 6 1117.46
data/clique500 8 865.21
data/clique500 10 1024.5
data/clique500 12 954.76
data/clique500 14 898.43
data/clique500 16 907.67
data/tree100 1 130.19
data/tree100 2 78.09
data/tree100 4 61.99
data/tree100 6 64.45
data/tree100 8 47.61
data/tree100 10 54.3
data/tree100 12 50.08
data/tree100 14 47.89
data/tree100 16 116.41
data/tree200 1 468.39
data/tree200 2 287.99
data/tree200 4 172.21
data/tree200 6 141.8
data/tree200 8 112.35
data/tree200 10 138.43
data/tree200 12 126.18
data/tree200 14 116.45
data/tree200 16 120.62
data/tree300 1 955.09
data/tree300 2 587.68
data/tree300 4 339.89
data/tree300 6 254.18
data/tree300 8 211.16
data/tree300 10 274.68
data/tree300 12 247.42
data/tree300 14 274.3
data/tree300 16 261.04
data/tree400 1 1619.27
data/tree400 2 941.25
data/tree400 4 555.14
data/tree400 6 418.7
data/tree400 8 335.99
data/tree400 10 438.14
data/tree400 12 406.7
data/tree400 14 367.31
data/tree400 16 330.03
data/tree500 1 2412.16
data/tree500 2 1377.01
data/tree500 4 833.57
data/tree500 6 595.69
data/tree500 8 725.26
data/tree500 10 648.93
data/tree500 12 573.43
data/tree500 14 589.91
data/tree500 16 565.24
data/soc-wiki-Vote 1 7448.42
data/soc-wiki-Vote 2 3979.09
data/soc-wiki-Vote 4 2114.5
data/soc-wiki-Vote 6 1507.25
data/soc-wiki-Vote 8 1213.01
data/soc-wiki-Vote 10 1660.53
data/soc-wiki-Vote 12 1441.69
data/soc-wiki-Vote 14 1283.57
data/soc-wiki-Vote 16 1447.32
data/fb-pages-tvshow 1 141656
data/fb-pages-tvshow 2 71117.6
data/fb-pages-tvshow 4 36015.5
data/fb-pages-tvshow 6 24470
data/fb-pages-tvshow 8 20045.5
data/fb-pages-tvshow 10 28451.7
data/fb-pages-tvshow 12 23728.2
data/fb-pages-tvshow 14 20393.1
data/fb-pages-tvshow 16 18588.9
```

