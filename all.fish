#!/usr/bin/fish

set GPU_SM_ARCH 75 # on Turing RTX2080Ti. CUDA 11.1.105
set GPU_SM_ARCH 35 # on Kepler K80/K40. CUDA 10.0.130
set BUILD_TYPE Debug
set BUILD_TYPE Release

cmake -D CMAKE_CUDA_ARCHITECTURES=$GPU_SM_ARCH -D CMAKE_BUILD_TYPE=$BUILD_TYPE .
make -j10

# benchmarking

set cpu_threads 1 4 8 12 16 20
set gpu_threads 16 64 256 1024
set files \
    data/clique100 data/clique200 data/clique300 data/clique400 data/clique500 \
    data/wikivote data/Gowalla

mkdir data
echo -n '' > data/bench.cpu
echo -n '' > data/bench.gpu
python src/generate_testcases.py

for fin in $files
    echo "##### benchmark on $fin"

    for ths in $gpu_threads
        set -x CUDA_THREADS_PER_BLOCK $ths
        echo "## GPU program started, with block=$ths"
        ./once_gpu.fish $fin data/bench.gpu
        echo "## GPU program ended"
    end

    for ths in $cpu_threads
        set -x OMP_NUM_THREADS $ths
        echo "## CPU program started, with threads=$OMP_NUM_THREADS"
        ./once_cpu.fish $fin data/bench.cpu
        echo "## CPU program ended"
    end
end
