#include "common.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>

struct Vertex {
  Vec2D pos, disp;
  __host__ __device__ Vertex() : pos(), disp() {}
  __host__ __device__ Vertex(const Vec2D &p) : pos(p), disp() {}
};

struct Edge {
  u32 u, v;
  __host__ __device__ Edge(u32 u = 0, u32 v = 0) { this->u = u, this->v = v; }
  __host__ __device__ inline bool operator<(const Edge &e) const {
    return u == e.u ? v < e.v : u < e.u;
  }
};

// INPUT:
//   - vertices: array of vertex coordinates
//   - edges: array of edges in the graph
//   - edge_head,edge_tail: range(head[i], tail[i]) are the edges for i
//   - k: push/pull force coefficient
//   - N: number of vertices
//   - M: number of edges
//   - ITER: number of iterations to run
// OUTPUT:
//   - return: FP number, cost of time, in ms (1e-6 second)
static inline __device__ void bound(FP &x, FP lo, FP hi) {
  x = min2(hi, max2(lo, x));
}

static __global__ void pushKernel(Vertex *vertices, FP K, u32 N, FP MIN_DIS) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    Vec2D posi = vertices[i].pos;
    Vec2D disp(0, 0);
    for (u32 j = 0; j < N; j++) {
      auto diff = posi - vertices[j].pos;
      auto dis = max2(MIN_DIS, diff.norm());

      auto push = force_push(dis, K);
      disp += diff / dis * push;
    }
    vertices[i].disp += disp;
  }
}
static __global__ void pullKernel(Vertex *vertices, Edge *edges, FP K, u32 N,
                                  u32 M, FP MIN_DIS) {
  u32 k = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;
  while (k < M) {
    auto e = edges[k];
    u32 i = e.u, j = e.v;
    auto diff = vertices[i].pos - vertices[j].pos;
    auto dis = max2(MIN_DIS, diff.norm());

    auto pull = force_pull(dis, K);
    auto disp = diff / (-dis) * pull;
    atomicAdd(&vertices[i].disp.x, disp.x),
        atomicAdd(&vertices[i].disp.y, disp.y);
    k += stride;
  }
}
static __global__ void updateKernel(Vertex *vertices, u32 N, FP temperature,
                                    FP SIZE) {
  u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    auto disp = vertices[i].disp;
    auto dis = disp.norm();
    vertices[i].pos += (disp / dis) * min2(temperature, dis);
    bound(vertices[i].pos.x, 0, SIZE), bound(vertices[i].pos.y, 0, SIZE);
    vertices[i].disp = Vec2D(0, 0);
  }
}

static inline u32 div_ceil(u32 a, u32 b) { return (a + b - 1) / b; }
static FP layout(Vertex *vertices, Edge *edges, FP K, u32 N, u32 M, u32 ITER,
                 FP START_TEMP, FP SIZE, u32 THREADS_PER_BLOCK) {
  Timer timer;
  checkCudaErrors(cudaDeviceSynchronize());
  timer.start();

  u32 BLOCKS_N = div_ceil(N, THREADS_PER_BLOCK);
  u32 BLOCKS_M = div_ceil(N, THREADS_PER_BLOCK);
  FP temperature = START_TEMP;
  for (u32 run = 0; run < ITER; run++) {
    // compuate the forces in parallel
    pushKernel<<<BLOCKS_N, THREADS_PER_BLOCK>>>(vertices, K, N, MIN_DIS);
    pullKernel<<<BLOCKS_M, THREADS_PER_BLOCK>>>(vertices, edges, K, N, M,
                                                MIN_DIS);
    // move to new coordinates
    updateKernel<<<BLOCKS_N, THREADS_PER_BLOCK>>>(vertices, N, temperature,
                                                  SIZE);
    temperature *= COOLING_FACTOR;
  }

  checkCudaErrors(cudaDeviceSynchronize());
  timer.end();
  return timer.delta();
}

i32 main(int argc, char *argv[]) {
  auto raw_threads = getenv("CUDA_THREADS_PER_BLOCK");
  const u32 THREADS_PER_BLOCK = std::stoul(raw_threads ? raw_threads : "1024");
  std::cout << THREADS_PER_BLOCK << " ";

  if (argc < 6) {
    std::cerr << "Usage: bin/gpu N M ITER in_file out_file\n";
    std::exit(1);
  }

  const u32 N = std::stoul(argv[1]), M = std::stoul(argv[2]),
            ITER = std::stoul(argv[3]);
  const String in_file(argv[4]), out_file(argv[5]);
  std::ifstream in_stream(in_file);
  std::ofstream out_stream(out_file);

  const FP SIZE = 200 * sqrt(1.0 * N);
  const FP START_TEMP = SIZE / 10;

  auto vertices = new Vertex[N];
  auto edges = new Edge[M];

  for (u32 i = 0, u, v; i < M; i++) {
    in_stream >> u >> v;
    edges[i] = Edge(u, v);
  }
  std::sort(edges, edges + M);

  std::random_device rdev;
  std::mt19937 rng(rdev());
  std::uniform_real_distribution<f32> unif(0.0, 1.0);
  for (u32 i = 0; i < N; i++) {
    FP x = unif(rng) * SIZE, y = unif(rng) * SIZE;
    vertices[i] = Vertex(Vec2D(x, y));
  }

  f32 K = sqrt(SIZE * SIZE / N) / (1.0 * N * N / M);

  auto vertices_device = CODA::cuda_new<Vertex>(N);
  auto edges_device = CODA::cuda_new<Edge>(M);
  CODA::H2D(vertices_device, vertices, N);
  CODA::H2D(edges_device, edges, M);

  auto runtime = layout(vertices_device, edges_device, K, N, M, ITER,
                        START_TEMP, SIZE, THREADS_PER_BLOCK);

  CODA::D2H(vertices, vertices_device, N);
  for (u32 i = 0; i < N; i++) {
    FP x = vertices[i].pos.x, y = vertices[i].pos.y;
    out_stream << x << ' ' << y << '\n';
  }

  CODA::cuda_delete(vertices_device);
  CODA::cuda_delete(edges_device);
  delete[] vertices;
  delete[] edges;

  std::cout << runtime / ITER << "\n";
  return 0;
}
