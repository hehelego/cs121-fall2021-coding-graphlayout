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
  Vertex(FP x = 0, FP y = 0) {
    pos.x = x, pos.y = y;
    disp.x = disp.y = 0;
  }
};

struct Edge {
  u32 u, v;
  Edge(u32 u = 0, u32 v = 0) { this->u = u, this->v = v; }
  inline bool operator<(const Edge &e) const { return u == e.u ? v < e.v : u < e.u; }
};

// INPUT:
//   - vertices: array of vertex coordinates
//   - edges: array of edges in the graph
//   - edge_head,edge_tail: range(head[i], tail[i]) are the edges for i
//   - cr: repulsive force coefficient
//   - ca: attractive force coefficient
//   - N: number of vertices
//   - M: number of edges
//   - ITER: number of iterations to run
// OUTPUT:
//   - return: FP number, cost of time, in ms (1e-6 second)
static inline void bound(FP &x, FP lo, FP hi) { x = min2(hi, max2(lo, x)); }
static FP layout(Vertex *vertices, Edge *edges, u32 *edge_head, u32 *edge_tail, FP cr, FP ca, u32 N, u32 M, u32 ITER) {
  Timer timer;
  timer.start();

  cr /= N, ca /= N;
  FP temperature = START_TEMPERATURE;

  for (u32 run = 0; run < ITER; run++) {
#pragma omp parallel for default(shared) schedule(dynamic, 16)
    for (u32 i = 0; i < N; i++) {
      // push away other vertices
      for (u32 j = 0; j < N; j++) {
        auto dx = vertices[j].pos.x - vertices[i].pos.x;
        auto dy = vertices[j].pos.y - vertices[i].pos.y;
        auto dis = max2(MIN_DIS, sqrt(dx * dx + dy * dy));

        auto push = force_hookelastic(dis, cr);
        vertices[j].disp.x += dx / dis * push;
        vertices[j].disp.y += dy / dis * push;
      }
      // attract other vertices
      u32 l = edge_head[i], r = edge_tail[i];
      for (u32 k = l; k <= r; k++) {
        u32 j = edges[k].v;
        auto dx = vertices[j].pos.x - vertices[i].pos.x;
        auto dy = vertices[j].pos.y - vertices[i].pos.y;
        auto dis = max2(MIN_DIS, sqrt(dx * dx + dy * dy));

        auto pull = force_gravitation(dis, ca);
        vertices[j].disp.x -= dx / dis * pull;
        vertices[j].disp.y -= dy / dis * pull;
      }
    }

    // move to new coordinates
#pragma omp parallel for default(shared) schedule(dynamic, 16)
    for (u32 i = 0; i < N; i++) {
      auto dx = vertices[i].disp.x;
      auto dy = vertices[i].disp.y;
      FP dis = sqrt(dx * dx + dy * dy);
      if (dis > temperature) {
        vertices[i].pos.x += dx / dis * temperature;
        vertices[i].pos.y += dy / dis * temperature;
      }
      bound(vertices[i].pos.x, -Width / 2.0, Width / 2.0);
      bound(vertices[i].pos.y, -Height / 2.0, Height / 2.0);
      vertices[i].disp.x = vertices[i].disp.y = 0;
    }

    temperature *= COOLING_FACTOR;
  }

  timer.end();
  return timer.delta();
}

i32 main(int argc, char *argv[]) {
  omp_set_num_threads(CPU_THS);

  if (argc < 6) {
    Debug() << "Usage: bin/cpu N M ITER in_file out_file\n";
    std::exit(1);
  }

  const u32 N = std::stoul(argv[1]), M = std::stoul(argv[2]), ITER = std::stoul(argv[3]);
  const String in_file(argv[4]), out_file(argv[5]);
  std::ifstream in_stream(in_file);
  std::ofstream out_stream(out_file);

  auto vertices = new Vertex[N];
  auto edges = new Edge[M];
  auto head = new u32[N + 1], tail = new u32[N + 1];
  std::fill(head, head + (N + 1), M);
  std::fill(tail, tail + (N + 1), 0);

  for (u32 i = 0, u, v; i < M; i++) {
    in_stream >> u >> v;
    if (u > v) swap2(u, v);
    edges[i] = Edge(u, v);
  }
  std::sort(edges, edges + M);
  for (i32 i = M - 1; i >= 0; i--) head[edges[i].u] = i;
  for (i32 i = 0; i < M; i++) tail[edges[i].u] = i;

  std::random_device rdev;
  std::mt19937 rng(rdev());
  std::uniform_real_distribution<f32> unif(-0.5, 0.5);
  for (u32 i = 0; i < N; i++) {
    FP x = unif(rng) * Width, y = unif(rng) * Height;
    vertices[i] = Vertex(x, y);
  }
  f32 K = sqrt((1.0 / N) * (Height * Width));
  layout(vertices, edges, head, tail, K, K, N, M, ITER);

  for (u32 i = 0; i < N; i++) {
    FP x = vertices[i].pos.x, y = vertices[i].pos.y;
    out_stream << x << ' ' << y << '\n';
  }
  delete[] vertices;
  delete[] edges;
  delete[] head, delete[] tail;
  return 0;
}
