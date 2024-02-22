#include <CL/sycl.hpp>

/*
Keys:
  1) queue: Use DPC++ sycl::queue to schedule and execute command queues on devices.
  2) malloc_shared(): Use sycl::malloc_shared Unified Shared Memory (USM) for data management.
  3) parallel_for(): Execute lambda expressions in queue with parallel on accelerator devices.
  4) submit(): Submits the command group functor to the queue for scheduling on the device.
  5) handler: Command group handler object, providing a series of scheduling functions.
  5) free(): Use sycl::free to deallocate allocated Unified Shared Memory (USM).

Queue:
  • A queue is used to submit command groups to the SYCL runtime for execution.
    - Member function submit: Submits the command group functor to the queue for scheduling on the device.
    - The parallel_for member function is a simplified form of submit.
  • A queue is a mechanism for submitting work to a device.
  • One queue maps to one device, and multiple queues can map to the same device.

Handler:
  • parallel_for: Execute lambda expressions or function objects in parallel on the device.
  • single_task: Execute a single task (single kernel) on the device.
  • copy: Copy data between devices.
  • update_host: Update data from the device to the host.
  • memset: Set the value of a memory block on the device.
  • fill: Fill a buffer or array on the device.
  • prefetch: Prefetch data to the device.
*/

constexpr int N = 16;
using namespace sycl;

int main() {
    queue q;                               //     ──┐
    int *data = malloc_shared<int>(N, q);  //       ├─ Host code
                                           //       │
    q.parallel_for(N, [=](auto i) {        // ──┐ ──┘
        data[i] = i;                       //   ├─ Device code
    }).wait();                             // ──┘ ──┐
                                           //       ├─ Host code
    for (int i = 0; i < N; i++)            //       │
        std::cout << data[i] << "\n";      //     ──┘

    q.submit([&] (handler &h) {
        h.parallel_for(N, [=](auto i) {
            data[i] = i * 10;
        });
    }).wait();

    for (int i = 0; i < N; i++)
        std::cout << data[i] << "\n";

    free(data, q);                         // Host code
    return 0;
}
