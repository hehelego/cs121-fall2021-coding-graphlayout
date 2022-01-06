#ifndef CS121_PROJ_TIMER_H
#define CS121_PROJ_TIMER_H
#include <chrono>
struct Timer {
  using namespace std::chrono;
  high_resolution_clock::time_point _start, _end;
  Timer() {}
  inline void start() { _start = high_resolution_clock::now(); }
  inline void end() { _end = high_resolution_clock::now(); }
  inline double delta() const { return duration_cast<microseconds>(_end - _start).count(); }
};
#endif
