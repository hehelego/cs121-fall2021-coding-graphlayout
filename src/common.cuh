#ifndef CS121_PROJ_COMM_H
#define CS121_PROJ_COMM_H

// C++ standard library
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cassert>
#include <cstdint>

// Nvidia CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

// my headers
#include "cuda_error.cuh"
#include "debug_log.cuh"
#include "timer.cuh"
// checkCudaErrors(val)

// types
using i8 = int8_t;
using u8 = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using f32 = float;
using f64 = long double;
using FP = f32;
using String = std::string;

// constants
const FP EPS = 1e-10;
const FP MIN_DIS = 1e-3;
const FP COOLING_FACTOR = 0.98;

// OMP CPU threads
const u32 CPU_THS = 8;

// common functions
template <typename T> __host__ __device__ inline T max2(T a, T b) {
  return a < b ? b : a;
}
template <typename T> __host__ __device__ inline T min2(T a, T b) {
  return a < b ? a : b;
}
template <typename T> __host__ __device__ inline void swap2(T &a, T &b) {
  T c = a;
  b = a;
  a = c;
}
template <typename T> __host__ __device__ inline T force_push(T d, T k) {
  return k * k / d;
}
template <typename T> __host__ __device__ inline T force_pull(T d, T k) {
  return d * d / k;
}

namespace CODA {
template <typename T> inline T *cuda_new(u32 n = 1) {
  T *p;
  checkCudaErrors(cudaMalloc(&p, sizeof(T) * n));
  return p;
}
template <typename T> inline void cuda_delete(T *p) {
  checkCudaErrors(cudaFree(p));
}

template <typename T> inline void D2D(T *dst, T *src, u32 n = 1) {
  cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToDevice);
}
template <typename T> inline void D2H(T *dst, T *src, u32 n = 1) {
  cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost);
}
template <typename T> inline void H2D(T *dst, T *src, u32 n = 1) {
  cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice);
}
template <typename T> inline void H2H(T *dst, T *src, u32 n = 1) {
  cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToHost);
}
template <typename T> inline void setSymbol(T *dst, T *src, u32 n = 1) {
  cudaMemcpyToSymbol(dst, src, sizeof(T) * n);
}
} // namespace CODA

struct Vec2D {
  FP x, y;
  __host__ __device__ Vec2D(FP x = 0, FP y = 0) : x(x), y(y) {}
  __host__ __device__ inline Vec2D operator+(const Vec2D &p) const {
    return Vec2D(x + p.x, y + p.y);
  }
  __host__ __device__ inline void operator+=(const Vec2D &p) {
    x += p.x, y += p.y;
  }
  __host__ __device__ inline Vec2D operator-(const Vec2D &p) const {
    return Vec2D(x - p.x, y - p.y);
  }
  __host__ __device__ inline void operator-=(const Vec2D &p) {
    x -= p.x, y -= p.y;
  }
  __host__ __device__ inline Vec2D operator*(const FP k) const {
    return Vec2D(x * k, y * k);
  }
  __host__ __device__ inline void operator*=(const FP k) { x *= k, y *= k; }
  __host__ __device__ inline Vec2D operator/(const FP k) const {
    return Vec2D(x / k, y / k);
  }
  __host__ __device__ inline void operator/=(const FP k) { x /= k, y /= k; }

  __host__ __device__ inline FP dot(const Vec2D &p) const {
    return x * p.x + y * p.y;
  }
  __host__ __device__ inline FP norm() const { return sqrt(x * x + y * y); }
};
std::ostream &operator<<(std::ostream &os, const Vec2D &v) {
  os << "Vec2D(" << v.x << "," << v.y << ")";
  return os;
}

#endif
