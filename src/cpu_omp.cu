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
static inline void bound(FP &x, FP lo, FP hi) { x = min2(hi, max2(lo, x)); }
static FP layout(Vertex *vertices, Edge *edges, u32 *edge_head, u32 *edge_tail,
                 FP K, u32 N, u32 M, u32 ITER, FP START_TEMP, FP SIZE) {
  Timer timer;
  timer.start();

  FP temperature = START_TEMP;
  for (u32 run = 0; run < ITER; run++) {
#pragma omp parallel for default(shared) schedule(dynamic, 1)
    for (u32 i = 0; i < N; i++) {
      // push away by other vertices
      for (u32 j = 0; j < N; j++) {
        auto d = vertices[i].pos - vertices[j].pos;
        auto dis = max2(MIN_DIS, d.norm());

        auto push = force_push(dis, K);
        vertices[i].disp += d / dis * push;
      }
      // attract by other vertices
      u32 l = edge_head[i], r = edge_tail[i];
      for (u32 k = l; k <= r; k++) {
        assert(edges[k].u == i);
        u32 j = edges[k].v;
        auto d = vertices[i].pos - vertices[j].pos;
        auto dis = max2(MIN_DIS, d.norm());

        auto pull = force_pull(dis, K);
        vertices[i].disp -= d / dis * pull;
      }
    }

#pragma omp parallel for default(shared) schedule(static)
    // move to new coordinates
    for (u32 i = 0; i < N; i++) {
      auto disp = vertices[i].disp;
      auto dis = disp.norm();
      vertices[i].pos += (disp / dis) * min2(temperature, dis);
      bound(vertices[i].pos.x, 0, SIZE), bound(vertices[i].pos.y, 0, SIZE);
      vertices[i].disp = Vec2D(0, 0);
    }

    temperature *= COOLING_FACTOR;
  }

  timer.end();
  return timer.delta();
}

i32 main(int argc, char *argv[]) {
  std::cout << omp_get_num_threads() << " ";

  if (argc < 6) {
    std::cerr << "Usage: bin/cpu N M ITER in_file out_file\n";
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
  auto head = new u32[N + 1], tail = new u32[N + 1];
  std::fill(head, head + (N + 1), M);
  std::fill(tail, tail + (N + 1), 0);

  for (u32 i = 0, u, v; i < M; i++) {
    in_stream >> u >> v;
    edges[i] = Edge(u, v);
  }
  std::sort(edges, edges + M);
  for (i32 i = M - 1; i >= 0; i--)
    head[edges[i].u] = i;
  for (i32 i = 0; i < M; i++)
    tail[edges[i].u] = i;

  std::random_device rdev;
  std::mt19937 rng(rdev());
  std::uniform_real_distribution<f32> unif(0.0, 1.0);
  for (u32 i = 0; i < N; i++) {
    FP x = unif(rng) * SIZE, y = unif(rng) * SIZE;
    vertices[i] = Vertex(Vec2D(x, y));
  }

  f32 K = sqrt(SIZE * SIZE / N) / (1.0 * N * N / M);
  // auto runtime = layout(vertices, edges, head, tail, K, N, M, ITER, START_TEMP, SIZE);
  auto runtime = ITER*1e5;

  for (u32 i = 0; i < N; i++) {
    FP x = vertices[i].pos.x, y = vertices[i].pos.y;
    out_stream << x << ' ' << y << '\n';
  }

  delete[] vertices;
  delete[] edges;
  delete[] head, delete[] tail;

  std::cout<< runtime/ITER << "\n";
  return 0;
}
