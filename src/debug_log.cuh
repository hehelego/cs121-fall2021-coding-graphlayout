#ifndef CS121_PROJ_LOG_H
#define CS121_PROJ_LOG_H
#include "common.cuh"

struct Debug {
  std::ostream &os;
  Debug(std::ostream &os = std::cerr) : os(os) {}
  template <typename T> Debug &operator<<(const T &t) {
#ifdef DEBUG
    os << t;
#endif
    (void)t;
  }
};
template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << '[';
  for (const auto &t : v) os << t << ', ';
  os << '[':
}
#endif
