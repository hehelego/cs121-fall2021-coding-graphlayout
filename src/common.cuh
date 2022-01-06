#pragma once

// C++ standard library
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>

// Nvidia CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math_functions.h>

// my headers
#include "cuda_error.cuh"
#include "debug_log"