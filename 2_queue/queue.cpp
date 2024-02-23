#include <CL/sycl.hpp>
#include <iostream>
#include <memory>

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
  • queue could be initialize by device, like: sycl::queue q(device)

Handler:
  • parallel_for: Execute lambda expressions or function objects in parallel on the device.
  • single_task: Execute a single task (single kernel) on the device.
  • copy: Copy data between devices.
  • update_host: Update data from the device to the host.
  • memset: Set the value of a memory block on the device.
  • fill: Fill a buffer or array on the device.
  • prefetch: Prefetch data to the device.

USM:
  Unified Shared Memory (USM) is a pointer-based memory management in SYCL, like malloc or new of C or C++ to allocate data.
  ---------------------------------------
      CPU           GPU     |  CPU   GPU
       │             │      |    \   /
   HostMemory --- GPUMemory |     USM
  ---------------------------------------
  | Type   | Function call  | Description                                               | Accessible on Host  | Accessible on Device |
  | ------ | -------------- | --------------------------------------------------------- | ------------------- | -------------------- |
  | Device | malloc_device  | Allocation on device (explicit)                           | NO                  | YES                  |
  | Host   | malloc_host    | Allocation on host (implicit)                             | YES                 | YES                  |
  | Shared | malloc_shared  | Allocation can migrate between host and device (implicit) | YES                 | YES                  |
*/

constexpr int N = 16;

int main() {
    // USM Implicit Data Movement
    sycl::queue q;                               //     ──┐
    int *data = sycl::malloc_shared<int>(N, q);  //       ├─ Host code
                                                 //       │
    q.parallel_for(N, [=](auto i) {              // ──┐ ──┘
        data[i] = i;                             //   ├─ Device code
    }).wait();                                   // ──┘ ──┐
                                                 //       ├─ Host code
    for (int i = 0; i < N; i++)                  //       │
        std::cout << data[i] << "\n";            //     ──┘

    q.submit([&](sycl::handler &h) {
        h.parallel_for(N, [=](auto i) {
            data[i] = i * 10;
        });
    }).wait();

    for (int i = 0; i < N; i++)
        std::cout << data[i] << "\n";

    sycl::free(data, q);                         // Host code

    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::string IntelDev = q.get_device().get_info<sycl::info::device::name>().find("Intel") != std::string::npos ? "Yes" : "No";
    std::cout << "Device is from Intel: " << IntelDev << std::endl;
    std::string isGPU = q.get_device().is_gpu() == true ? "Yes" : "No";
    std::cout << "Device is GPU: " << isGPU << std::endl;

    // USM Explicit Data Movement
    {
        sycl::queue q;
        int *data_cpu = static_cast<int *>(std::malloc(N * sizeof(int)));
        int *data_gpu = sycl::malloc_device<int>(N, q);

        q.memcpy(data_gpu, data_cpu, sizeof(int) * N).wait();

        // Add data dependency
        auto e = q.parallel_for(N, [=](auto i) { data_gpu[i] = i + 1; });
        q.parallel_for(N, e, [=](auto i) { data_gpu[i] = i + 2; });
        q.wait();

        q.memcpy(data_cpu, data_gpu, sizeof(int) * N).wait();

        for (int i = 0; i < N; i++)
            std::cout << data_cpu[i] << "\n";

        free(data_cpu);
        sycl::free(data_gpu, q);
    }

    return 0;
}
