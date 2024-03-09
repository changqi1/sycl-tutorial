#include <CL/sycl.hpp>
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>

/*
Keys:
  1) queue: Use DPC++ sycl::queue to schedule and execute command queues on devices.
*/

constexpr int N = 4096;

template <typename WeiT>
void memcpy_MT_device(sycl::queue &q, WeiT *dst, const WeiT *src, size_t size) {
#define THREADS 2
#pragma omp parallel for num_threads(THREADS)
    for (uint64_t i = 0; i < THREADS; i++) {
        size_t length = size / THREADS;
        q.memcpy(dst + length * i, src + length * i, length * sizeof(WeiT));
    }
    q.wait();
}

int main() {
    sycl::queue q;
    int16_t *data_cpu = static_cast<int16_t *>(std::malloc(N * sizeof(int16_t)));
    int16_t *data_cpu_pinned = sycl::malloc_host<int16_t>(N, q);
    int16_t *data_gpu = sycl::malloc_device<int16_t>(N, q);

    memset(data_cpu, 88, N * sizeof(int16_t));
    memcpy(data_cpu_pinned, data_cpu, N * sizeof(int16_t));
    q.fill(data_gpu, 88, N);

    {
        std::cout << "parallel_for: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        q.parallel_for(N, [=](auto i) { data_gpu[i] = data_cpu_pinned[i]; }).wait();
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "c2g bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    {
        std::cout << "parallel_for: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        q.parallel_for(N, [=](auto i) { data_cpu_pinned[i] = data_gpu[i]; }).wait();
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "g2c bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    {
        std::cout << "Paged memory: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        q.memcpy(data_gpu, data_cpu, sizeof(int16_t) * N).wait();
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "c2g bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    {
        std::cout << "Paged memory: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        q.memcpy(data_gpu, data_cpu, sizeof(int16_t) * N).wait();
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "g2x bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    {
        std::cout << "Pinned memory: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        q.memcpy(data_gpu, data_cpu_pinned, sizeof(int16_t) * N).wait();
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "c2g bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    {
        std::cout << "Pinned memory: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        q.memcpy(data_cpu_pinned, data_gpu, sizeof(int16_t) * N).wait();
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "g2c bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    {
        std::cout << "Pinned memory MT: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        memcpy_MT_device(q, data_gpu, data_cpu_pinned, N);
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "c2g bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    {
        std::cout << "Pinned memory MT: " << std::endl;
        auto tag_0 = std::chrono::high_resolution_clock::now();
        memcpy_MT_device(q, data_cpu_pinned, data_gpu, N);
        auto tag_1 = std::chrono::high_resolution_clock::now();
        auto diff_0_1 = std::chrono::duration_cast<std::chrono::microseconds>(tag_1 - tag_0);
        std::cout << "diff_0_1: " << diff_0_1.count() << " usec" << std::endl;
        std::cout << "g2c bandwidth: " << (double) N * sizeof(int16_t) / diff_0_1.count() / 1024 << " GB/s" << std::endl;
    }

    free(data_cpu);
    sycl::free(data_cpu_pinned, q);
    sycl::free(data_gpu, q);

    return 0;
}
