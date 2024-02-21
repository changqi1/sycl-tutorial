#include <CL/sycl.hpp>
#include <iostream>

#define FLT_MIN 1e9

inline int OneDimArrayFMA(const sycl::device &device)
{
    // Create a queue to execute the kernels
    cl::sycl::queue q(device);
    constexpr size_t length = 10;
    constexpr size_t xval = 1;
    constexpr size_t yval = 2;
    constexpr size_t zval = 3;

    const float aval(3);

    const float correct = (zval + aval * xval + yval);
    // host data
    std::vector<float> h_X(length, xval);
    std::vector<float> h_Y(length, yval);
    std::vector<float> h_Z(length, zval);
    try
    {
        const float A(aval);
        // create buffer to handle host data
        sycl::buffer<float, 1> d_X{h_X.data(), sycl::range<1>(h_X.size())};
        sycl::buffer<float, 1> d_Y{h_Y.data(), sycl::range<1>(h_Y.size())};
        sycl::buffer<float, 1> d_Z{h_Z.data(), sycl::range<1>(h_Z.size())};

        q.submit([&](sycl::handler &h)
                 {
      auto X = d_X.get_access<sycl::access::mode::read>(h);
      auto Y = d_Y.get_access<sycl::access::mode::read>(h);
      auto Z = d_Z.get_access<sycl::access::mode::read_write>(h);

      h.parallel_for<class axpy>(sycl::range<1>{length}, [=](sycl::id<1> it) {
        const size_t i = it[0];
        Z[i] += A * X[i] + Y[i];
      }); });
        q.wait();
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    // check for correctness
    for (size_t i = 0; i < length; ++i)
    {
        if (std::abs(h_Z[i] - correct) > FLT_MIN)
        {
            std::cout << "error Index:" << i << ","
                      << "h_Z[i] value: " << h_Z[i] << std::endl;
        }
    }
}

inline int TwoDimArrayMatmul(const sycl::device &device)
{
    // Create a queue to execute the kernels
    cl::sycl::queue q(device);
    // host data
    size_t M = 2;
    size_t K = 2;
    size_t N = 3;

    int *host_matrix1 = (int *)aligned_alloc(64, M * N * sizeof(int));
    int *host_matrix2 = (int *)aligned_alloc(64, K * N * sizeof(int));
    int *host_result = (int *)aligned_alloc(64, M * N * sizeof(int));
    int *res_buffer = (int *)aligned_alloc(64, M * N * sizeof(int));

    for (size_t i = 0; i < M * K; i++)
    {
        host_matrix1[i] = i + 1;
    }

    for (size_t i = 0; i < K * N; i++)
    {
        host_matrix2[i] = i + 5;
    }

    for (size_t i = 0; i < M * N; i++)
    {
        host_result[i] = 0;
        res_buffer[i] = 0;
    }

    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < K; ++k)
            {
                host_result[m * N + n] += host_matrix1[m * K + k] * host_matrix2[k * N + n];
            }
        }
    }
    try
    {
        // create buffer to handle host data
        sycl::buffer<int, 1> device_mat1{host_matrix1, sycl::range<1>(M * K)};
        sycl::buffer<int, 1> device_mat2{host_matrix2, sycl::range<1>(K * N)};
        sycl::buffer<int, 1> device_res{res_buffer, sycl::range<1>(M * N)};

        q.submit([&](sycl::handler &h)
                 {
            auto X = device_mat1.get_access<sycl::access::mode::read>(h);
            auto Y = device_mat2.get_access<sycl::access::mode::read>(h);
            auto Z = device_res.get_access<sycl::access::mode::read_write>(h);


            h.parallel_for<class matrix_multiply>(cl::sycl::range<2>(M, N), [=](cl::sycl::item<2> idx)
                                                  {
                                                      int m = idx.get_id(0);
                                                      int n = idx.get_id(1);
                                                        for (int k = 0; k < K; ++k)
                                                        {
                                                            Z[m * N + n] += X[m * K + k] * Y[k * N + n];
                                                        }
                                                    });                                      
            q.wait(); });
    }
    catch (sycl::exception &e)
    {
        std::cout << e.what() << std::endl;
        free(host_matrix1);
        free(host_matrix2);
        free(host_result);
        free(res_buffer);
        return 1;
    }

    // check for correctness
    for (size_t i = 0; i < M * N; ++i)
    {
        if (std::abs(res_buffer[i] - host_result[i]) > FLT_MIN)
        {
            std::cout << "error Index:" << i << ","
                      << "res_buffer[i] value: " << res_buffer[i] << std::endl;
        }
    }

    free(host_matrix1);
    free(host_matrix2);
    free(host_result);
    free(res_buffer);
}

int main()
{
    auto devices = sycl::default_selector{}.select_device();
    OneDimArrayFMA(devices);
    TwoDimArrayMatmul(devices);
    return 0;
}
