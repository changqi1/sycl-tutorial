#include <CL/sycl.hpp>

/*
Keys:
  1) queue: Use DPC++ sycl::queue to schedule and execute command queues on devices.
  2) malloc_shared(): Use sycl::malloc_shared Unified Shared Memory (USM) for data management.
  3) parallel_for(): Execute lambda expressions in queue with parallel on accelerator devices.
  4) free(): Use sycl::free to deallocate allocated Unified Shared Memory (USM).
*/

constexpr int N = 16;
using namespace sycl;

int main() {
    queue q;                               //     ──┐
    int *data = malloc_shared<int>(N, q);  //       ├─  Host code
                                           //       │
    q.parallel_for(N, [=](auto i) {        // ──┐ ──┘
        data[i] = i;                       //   ├─ Device code
    }).wait();                             // ──┘ ──┐
                                           //       │
    for (int i = 0; i < N; i++)            //       ├─  Host code
        std::cout << data[i] << "\n";      //       │
                                           //       │
    free(data, q);                         //       │
    return 0;                              //     ──┘
}
