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

constexpr size_t globalSize = 256; // global size
constexpr size_t workGroupSize = 64; // work-group size

using namespace sycl;

int main() {
    queue q;
    std::vector<int> v(globalSize);

    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

    auto sg_sizes = q.get_device().get_info<info::device::sub_group_sizes>();
    std::cout << "Supported Sub-Group Sizes : ";
    for (int i=0; i<sg_sizes.size(); i++)
        std::cout << sg_sizes[i] << " "; std::cout << "\n";

    auto max_sg_size = std::max_element(sg_sizes.begin(), sg_sizes.end());
    std::cout << "Max Sub-Group Size        : " << max_sg_size[0] << "\n";

    buffer buf(v);
    q.submit([&](handler &h) {
        auto out = stream(1024, 768, h);

        // nd-range kernel with user specified sub_group size
        h.parallel_for(nd_range<1>(globalSize, workGroupSize), [=](nd_item<1> item) {
        // h.parallel_for(nd_range<1>(globalSize, workGroupSize), [=](nd_item<1> item)[[intel::reqd_sub_group_size(32)]] {
            auto sg = item.get_sub_group();
            out << "sub_group id: " << sg.get_group_id() << " of "
                << sg.get_group_range() << ", size=" << sg.get_local_range() << " "
                << sg.get_local_id() << "\n";
        });
    }).wait();

    for (int i = 0; i < globalSize; i++)
        std::cout << v[i] << " ";
    std::cout << std::endl;

    return 0;
}
