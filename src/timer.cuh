#ifndef CS121_PROJ_TIMER_H
#define CS121_PROJ_TIMER_H
#include <chrono>
struct Timer {
  std::chrono::high_resolution_clock::time_point _start, _end;
  Timer() {}
  inline void start() { _start = std::chrono::high_resolution_clock::now(); }
  inline void end() { _end = std::chrono::high_resolution_clock::now(); }
  inline double delta() const { return std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count(); }
};
#endif
