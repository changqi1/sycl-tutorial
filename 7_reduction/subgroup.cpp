#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

/*
Keys:
一、基本并行内核的功能通过 range、id 和 item 类提供

              ◄── range[1] ───────────►
          ▲   ┌───┬───┬───┬───┬───┬───┐
          |   │0,0│   │   │   │   │   │
    range[0]  ├───┼───┼───┼───┼───┼───┤
          |   │   │   │   │   │ ◄─┼───┼── item with id(1, 4)
          |   ├───┼───┼───┼───┼───┼───┤
          |   │   │   │   │   │   │   │
          |   ├───┼───┼───┼───┼───┼───┤
          |   │   │   │   │   │   │   │
          |   ├───┼───┼───┼───┼───┼───┤
          |   │   │   │   │   │   │   │
          ▼   └───┴───┴───┴───┴───┴───┘

        h.parallel_for(range<1>(1024), [=](item<1> item){
            auto idx = item.get_id();
            auto R = item.get_range();
        })

    1) range 类用于描述并行执行维度和大小
        • 可以表示1、2、3维
        • 维度需要在编译时确定
        • 每个维度的大小可以是运行时指定
    2) id 类用于表示range空间中的索引
        • 同样可以表示1、2、3维
        • 维度需要在编译时确定
        • 索引一个并行运行的实例
        • buffer的偏移
    3) item 类代表内核函数的单个实例
        • 封装内核的执行范围和该范围内的实例索引（分别使用 get_id 和 get_range）
        • 与 range 和 id 一样，它的维度必须在编译时确定

二、ND-Range 对应于硬件资源，影响着程序性能！
    • 基础并行内核虽然使用方便，但是无法根据硬件架构进行优化
    • ND-Range 内核可以将实例分为不同类型的分组，并且将它们精确的映射到硬件平台上
    • 正确的使用 ND-Range 内核可以充分的发挥出硬件性能潜力，包括内存访问和计算单元分配等

    | ---------- | ---------------------------------------------------------------------------------- | --------------------------- |
    |     SYCL code                                                                                   |         GPU Hardware        |
    | ---------- | ---------------------------------------------------------------------------------- | --------------------------- |
    | ND-Range   |                                                                                    | L2 Cache + GDDR             |
    | ---------- | ---------------------------------------------------------------------------------- | --------------------------- |
    | Work-group | The entire iteration space is divided into smaller groups called work-groups,      | Mapped into single Xe-core  |
    |            | work-items within a work-group are scheduled on a single compute unit on hardware. | in Arc GPU + L1 Cache       |
    | ---------- | ---------------------------------------------------------------------------------- | --------------------------- |
    | Work-item  | Represents the individual instances of a kernel function.                          | Mapped into one of a SIMD   |
    | ---------- | ---------------------------------------------------------------------------------- | --------------------------- |
    | Sub-group  | A subset of work-items within a work-group that are executed simultaneously,       | Mapped into a single engine |
    |            | may be mapped to vector hardware.                                                  | (Vector or Matrix Engine)   |
    | ---------- | ---------------------------------------------------------------------------------- | --------------------------- |


三、内核范围支持的 C++ 有部分限制
    • 更广泛的设备支持和大规模并行性
    • 不支持C++特性包括：动态多态性、动态内存分配（因此不使用 new 或 delete 运算符进行对象管理）、静态变量、函数指针、运行时类型信息 (RTTI) 和异常处理。
    • 不允许从内核代码调用任何虚拟成员函数和可变参数函数。
    • 内核代码中不允许递归

四、Sub-group 的重要性
    • sub-group 中的 Work-items 可以使用 shuffle 操作直接进行通信，无需显式的内存操作。
    • sub-group 中的 Work-items 可以使用 sub-group barriers 进行同步，并使用 sub-group memory fences 保证内存一致性。
    • sub-group 中的 Work-items 可以访问 sub-group 中的函数和算法，提供常见并行模式的快速实现。

template<int Dimensions = 1>
nd_range(range<Dimensions> globalSize, range<Dimensions> workGroupSize)
nd_range(range<Dimensions> globalSize, range<Dimensions> workGroupSize, id<Dimensions> offset)
*/

const int sequence_length = 4;
const int head_num = 4;
const int head_size = 2;

void Add(sycl::queue &q, std::vector<float> &a, std::vector<float> &b, std::vector<float> &c) {
    sycl::buffer<float, 1> bufferA(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> bufferB(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> bufferC(c.data(), sycl::range<1>(c.size()));

    int rows = 1;
    int cols = 4096;
    int iStride = 4096;
    int elementsPerThread = 8;
    float *device_data = sycl::malloc_device<float>(cols, q);
    q.fill(device_data, 1.0f, cols);
    sycl::buffer<float> bufSum {0.0f}; 
    sycl::buffer<float> part_sum(sycl::range<1>(512));

    // q.submit([&](sycl::handler &cgh) {
    //     auto out = sycl::stream(10240, 7680, cgh);
    //     auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
    //     auto part_sum_data = part_sum.get_access<sycl::access::mode::write>(cgh);
    //     cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(512), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
    //         for (int row = 0; row < rows; ++row) {
    //             int idx_col = item_ct1.get_global_id(0);
    //             float ss = 0.0f;
    //             for (int i = 0; i < elementsPerThread; i++) {
    //                 float val = (float)device_data[row * iStride + idx_col * elementsPerThread + i];
    //                 ss += val * val;
    //             }
    //             part_sum_data[idx_col] = ss;
    //         }
    //     });
    // }).wait();

    // q.submit([&](sycl::handler &cgh) {
    //     auto out = sycl::stream(10240, 7680, cgh);
    //     auto part_sum_data = part_sum.get_access<sycl::access::mode::write>(cgh);
    //     cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
    //         for (int row = 0; row < rows; ++row) {
    //             int idx_col = item_ct1.get_global_id(0);
    //             float sss = 0.0f;
    //             for (int i = 0; i < 16; i++) {
    //                 sss += part_sum_data[idx_col + 32 * i];
    //             }
    //             part_sum_data[idx_col] = sss;

    //             if (idx_col == 0) {
    //                 for (int i = 0; i < 32; ++i)
    //                     out << "part_sum_data[" << i << "]: " << part_sum_data[i]  << sycl::endl;
    //             }
    //         }
    //     });
    // }).wait();

    // q.submit([&](sycl::handler &cgh) {
    //     auto out = sycl::stream(10240, 7680, cgh);
    //     auto part_sum_data = part_sum.get_access<sycl::access::mode::write>(cgh);
    //     cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(8), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
    //         for (int row = 0; row < rows; ++row) {
    //             int idx_col = item_ct1.get_global_id(0);
    //             float sss = 0.0f;
    //             for (int i = 0; i < 4; i++) {
    //                 sss += part_sum_data[idx_col + 8 * i];
    //             }
    //             part_sum_data[idx_col] = sss;

    //             if (idx_col == 0) {
    //                 for (int i = 0; i < 8; ++i)
    //                     out << "part_sum_data[" << i << "]: " << part_sum_data[i]  << sycl::endl;
    //             }
    //         }
    //     });
    // }).wait();

    // q.submit([&](sycl::handler &cgh) {
    //     auto out = sycl::stream(10240, 7680, cgh);
    //     auto part_sum_data = part_sum.get_access<sycl::access::mode::write>(cgh);
    //     cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(8), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
    //         for (int row = 0; row < rows; ++row) {
    //             int idx_col = item_ct1.get_global_id(0);
    //             float ss = 0.0f;
    //             ss = part_sum_data[0] + part_sum_data[1] + part_sum_data[2] + part_sum_data[3] + 
    //                  part_sum_data[4] + part_sum_data[5] + part_sum_data[6] + part_sum_data[7];
    //             if (idx_col == 0) {
    //                 out << "ss: " << ss  << sycl::endl;
    //             }
    //         }
    //     });
    // }).wait();

    q.submit([&](sycl::handler &cgh) {
        auto out = sycl::stream(10240, 7680, cgh);
        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto part_sum_data = part_sum.get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(512), sycl::range<1>(1)), [=](sycl::nd_item<1> item_ct1) {
            for (int row = 0; row < rows; ++row) {
                int idx_col = item_ct1.get_global_id(0);
                if (idx_col < 512) {
                    float ss = 0.0f;
                    for (int i = 0; i < elementsPerThread; i++) {
                        float val = (float)device_data[row * iStride + idx_col * elementsPerThread + i];
                        ss += val * val;
                    }
                    part_sum_data[idx_col] = ss;
                }

                item_ct1.barrier(sycl::access::fence_space::global_space);
                item_ct1.barrier(sycl::access::fence_space::local_space);
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

                if (idx_col < 32) {
                    float ss = 0.0f;
                    for (int i = 0; i < 16; i++) {
                        ss += part_sum_data[idx_col + 32 * i];
                    }
                    part_sum_data[idx_col] = ss;
                }

                item_ct1.barrier(sycl::access::fence_space::global_space);
                item_ct1.barrier(sycl::access::fence_space::local_space);
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

                if (idx_col < 8) {
                    float ss = 0.0f;
                    for (int i = 0; i < 4; i++) {
                        ss += part_sum_data[idx_col + 8 * i];
                    }
                    part_sum_data[idx_col] = ss;
                }

                item_ct1.barrier(sycl::access::fence_space::global_space);
                item_ct1.barrier(sycl::access::fence_space::local_space);
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

                // for(unsigned int s = 1; s < 32; s *= 2) {
                //     if (idx_col % (2 * s) == 0) {
                //         part_sum_data[idx_col] += part_sum_data[idx_col + s];
                //     }
                //     item_ct1.barrier(sycl::access::fence_space::global_space);
                //     item_ct1.barrier(sycl::access::fence_space::local_space);
                // }

                // for (int s = cols / 2; s > 16; s >>= 1) {
                //     if (idx_col < s) {
                //         part_sum_data[idx_col] += part_sum_data[idx_col + s];
                //     }
                //     item_ct1.barrier(sycl::access::fence_space::global_space);
                //     item_ct1.barrier(sycl::access::fence_space::local_space);
                //     item_ct1.barrier();
                // }

                if (idx_col == 0) {
                    float ss = part_sum_data[0] + part_sum_data[1] + part_sum_data[2] + part_sum_data[3] + 
                               part_sum_data[4] + part_sum_data[5] + part_sum_data[6] + part_sum_data[7];
                    out << "ss: " << ss  << sycl::endl;
                    // for (int i = 0; i < 32; ++i)
                    //     out << "0 part_sum_data[" << i << "]: " << part_sum_data[i]  << sycl::endl;
                    // out << item_ct1.get_group() << sycl::endl;
                }

                if (idx_col < 32) {
                    out << "part_sum_data[" << idx_col << "]: " << part_sum_data[idx_col]  << sycl::endl;
                }
            }
        });
    }).wait();
}

int main() {
    sycl::queue q;

    std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    std::cout << "Supported Sub-Group Sizes : ";
    for (int i=0; i<sg_sizes.size(); i++)
        std::cout << sg_sizes[i] << " "; std::cout << "\n";

    auto max_sg_size = std::max_element(sg_sizes.begin(), sg_sizes.end());
    std::cout << "Max Sub-Group Size        : " << max_sg_size[0] << "\n";

    std::vector<float> a(sequence_length * head_num * head_size);
    std::vector<float> b(sequence_length * head_num * head_size);
    std::vector<float> c(sequence_length * head_num * head_size);

    for (int i = 0; i < sequence_length * head_num * head_size; ++i)
        a[i] = b[i] = i * 0.11;

    Add(q, a, b, c);

    for (int i = 0; i < sequence_length * head_num * head_size; ++i)
        std::cout << c[i] << " ";
    std::cout << std::endl;

    return 0;
}
