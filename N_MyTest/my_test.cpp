#include <CL/sycl.hpp>
#include <iostream>

constexpr int N = 16;

sycl::device getMyDevice() {
    auto platforms = sycl::platform::get_platforms();
    for (auto &platform : platforms) {
        auto devices = platform.get_devices();
        for (auto &device : devices) {
            if (device.has(sycl::aspect::ext_intel_pci_address)) {
                auto BDF  = device.get_info<sycl::ext::intel::info::device::pci_address>();
                std::cout << "Device BDF : " << BDF  << std::endl;
                if (BDF  == "0000:5b:00.0") {
                    return device;
                }
            }
        }
    }
}

int main() {
    sycl::queue q(getMyDevice());
    int *data = sycl::malloc_shared<int>(N, q);

    q.parallel_for(N, [=](auto i) {
        data[i] = i;
    }).wait();

    for (int i = 0; i < N; i++)
        std::cout << data[i] << "\n";

    sycl::free(data, q);
    return 0;
}
