#include <CL/sycl.hpp>
#include <iostream>
#include <memory>

// constexpr int N = 128*1024;
constexpr int N = 8;

int main() {
    sycl::queue q;
    float *data = sycl::malloc_shared<float>(N, q);

    q.parallel_for(N, [=](auto i) {
        data[i] = 0.01;
    }).wait();

    for (int i = 0; i < N; i++)
        std::cout << data[i] << "\n";

    q.submit([&](sycl::handler &h) {
        h.parallel_for(N, [=](auto i) {
            data[i] = data[i] * float(1.0) / (float(1.0) + sycl::native::exp(-data[i]));
        });
    }).wait();

    for (int i = 0; i < N; i++)
        std::cout << data[i] << "\n";

    sycl::free(data, q);

    return 0;
}