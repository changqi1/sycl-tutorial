#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

/*
Keys:
  1) buffer: Use sycl::buffer to create buffer on devices.
  2) accessor: the accessor of buffer.

Advantages: Expresses clear data dependencies.
Disadvantages: Using buffers is not as convenient as directly using pointers and arrays.
*/

constexpr int N = 16;

int main() {
    sycl::queue q;
    std::vector<int> v(N);

    {
        sycl::buffer buf(v);
        q.submit([&](sycl::handler &h) {
            sycl::accessor a(buf, h, sycl::write_only);
            h.parallel_for(N, [=](auto i) {
                a[i] = i;
            });
        }).wait();
    }

    for (int i = 0; i < N; i++)
        std::cout << v[i] << "\n";

    return 0;
}
