cmake_minimum_required(VERSION 3.15.1)
project(sycl_tutorial LANGUAGES C CXX)

set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-device-code-split=per_kernel -lOpenCL")

add_subdirectory(1_gpu_info)
add_subdirectory(2_queue)
add_subdirectory(2_array_operation)
add_subdirectory(N_MyTest)