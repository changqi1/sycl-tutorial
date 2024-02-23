#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

/*
Keys:
  1) buffer: Use sycl::buffer to create buffer on devices. Buffers encapsulate data in a SYCL application across both devices and host.
  2) accessor: the accessor is the mechanism to access buffer data.

Advantages: Expresses clear data dependencies.
Disadvantages: Using buffers is not as convenient as directly using pointers and arrays.
*/

constexpr int N = 16;

int main() {
    sycl::queue q;
    std::vector<int> v(N, 2);

    {
        sycl::buffer buf(v);
        q.submit([&](sycl::handler &h) {
            sycl::accessor a1(buf, h, sycl::read_only);
            sycl::accessor a2(buf, h, sycl::write_only);
            h.parallel_for(N, [=](auto i) {
                a2[i] = a1[i] + i;
            });
        }).wait();

        for (int i = 0; i < N; i++)
            std::cout << v[i] << " ";
        std::cout << std::endl; // 0 0 0 0 ...

        // Creating host accessor is a blocking call and will only return after all
        // enqueued SYCL kernels that modify the same buffer in any queue completes
        // execution and the data is available to the host via this host accessor.
        sycl::host_accessor b(buf, sycl::read_only);
        for (int i = 0; i < N; i++)
            std::cout << b[i] << " ";
        std::cout << std::endl; // 0 1 2 3 ...
    }

    // When execution advances beyond this function scope, buffer destructor is
    // invoked which relinquishes the ownership of data and copies back the data to
    // the host memory.
    for (int i = 0; i < N; i++)
        std::cout << v[i] << " ";
    std::cout << std::endl; // 0 1 2 3 ...

    return 0;
}
