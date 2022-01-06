#!/usr/bin/fish

set GPU_SM_ARCH 75 # on Turing RTX2080Ti. CUDA 11.1.105
set GPU_SM_ARCH 35 # on Kepler K80/K40. CUDA 10.0.130
set BUILD_TYPE Debug
set BUILD_TYPE Release

cmake -D CMAKE_CUDA_ARCHITECTURES=$GPU_SM_ARCH -D CMAKE_BUILD_TYPE=$BUILD_TYPE .
make -j10

# benchmarking

set threads 1 4 8 12 16 20
set files \
    data/clique100 data/clique200 data/clique300 data/clique400 data/clique500 \
    data/wikivote \
    data/Gowalla

echo -n '' > data/bench.cpu
echo -n '' > data/bench.gpu

for fin in $files
    cp data/test.in $fin
end

for fin in $files
    # ./once_gpu $fin data/bench.gpu
    for ths in $threads
        set -x OMP_NUM_THREADS
        ./once_cpu $fin data/bench.cpu
    end
end
