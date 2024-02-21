#include <CL/sycl.hpp>
#include <iostream>

#define FLT_MIN 1e9

constexpr size_t length = 10;
constexpr size_t xval = 1;
constexpr size_t yval = 2;
constexpr size_t zval = 3;

const float aval(3);

const float correct = (zval + aval * xval + yval);

int main() {
  // Create a queue to execute the kernels
  cl::sycl::queue q(cl::sycl::default_selector{});
  // host data
  std::vector<float> h_X(length, xval);
  std::vector<float> h_Y(length, yval);
  std::vector<float> h_Z(length, zval);
  try {
    const float A(aval);
    // create buffer to handle host data
    sycl::buffer<float, 1> d_X{h_X.data(), sycl::range<1>(h_X.size())};
    sycl::buffer<float, 1> d_Y{h_Y.data(), sycl::range<1>(h_Y.size())};
    sycl::buffer<float, 1> d_Z{h_Z.data(), sycl::range<1>(h_Z.size())};

    q.submit([&](sycl::handler &h) {
      auto X = d_X.get_access<sycl::access::mode::read>(h);
      auto Y = d_Y.get_access<sycl::access::mode::read>(h);
      auto Z = d_Z.get_access<sycl::access::mode::read_write>(h);

      h.parallel_for<class axpy>(sycl::range<1>{length}, [=](sycl::id<1> it) {
        const size_t i = it[0];
        Z[i] += A * X[i] + Y[i];
      });
    });
    q.wait();
  } catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  // check for correctness
  for (size_t i = 0; i < length; ++i) {
    if (std::abs(h_Z[i] - correct) > FLT_MIN) {
      std::cout << "error Index:" << i << ","
                << "value: " << h_Z[i] << std::endl;
    }
  }

  return 0;
}
