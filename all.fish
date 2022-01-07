#!/usr/bin/fish

set GPU_SM_ARCH 75 # on Turing RTX2080Ti. CUDA 11.1.105
set GPU_SM_ARCH 35 # on Kepler K80/K40. CUDA 10.0.130
set BUILD_TYPE Debug
set BUILD_TYPE Release

cmake -D CMAKE_CUDA_ARCHITECTURES=$GPU_SM_ARCH -D CMAKE_BUILD_TYPE=$BUILD_TYPE .
make -j10

# benchmarking
set files \
    data/clique100 data/clique200 data/clique300 data/clique400 data/clique500 \
    data/tree100 data/tree200 data/tree300 data/tree400 data/tree500 \
    data/soc-wiki-Vote data/fb-pages-tvshow
# https://nrvis.com/download/data/soc/soc-wiki-Vote.zip
# https://nrvis.com/./download/data/soc/fb-pages-tvshow.zip

mkdir data
python src/generate_testcases.py
echo -n '' > data/bench.cpu
echo -n '' > data/bench.gpu

set cpu_threads 1 2 4 6 8 10 12 14 16
set -x CUDA_BLOCK_X 128
set -x CUDA_BLOCK_Y 8

for fin in $files
    echo "##### benchmark on $fin"


    echo "## GPU program started"
    ./once_gpu.fish $fin data/bench.gpu
    echo "## GPU program ended"

    for ths in $cpu_threads
        set -x OMP_NUM_THREADS $ths
        echo "## CPU program started, with threads=$OMP_NUM_THREADS"
        ./once_cpu.fish $fin data/bench.cpu
        echo "## CPU program ended"
    end
end
