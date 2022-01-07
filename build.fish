#!/usr/bin/fish

set GPU_SM_ARCH 75 # on Turing RTX2080Ti. CUDA 11.1.105
set GPU_SM_ARCH 35 # on Kepler K80/K40. CUDA 10.0.130
set BUILD_TYPE Debug
set BUILD_TYPE Release

cmake -D CMAKE_CUDA_ARCHITECTURES=$GPU_SM_ARCH -D CMAKE_BUILD_TYPE=$BUILD_TYPE .
make -j10