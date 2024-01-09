#include <CL/sycl.hpp>
#include <iostream>


// Print basic device information
// TODO: Add more device info
inline void ShowDeviceInfo(std::size_t idx, const sycl::device &device) {
    try{
        auto platform_name = device.get_platform().get_info<sycl::info::platform::name>();
        auto device_name = device.get_info<sycl::info::device::name>();
        auto vendor_name = device.get_info<sycl::info::device::vendor>();
        auto device_drv = device.get_info<sycl::info::device::driver_version>();
        auto device_type = static_cast<int>(device.get_info<sycl::info::device::device_type>());

        auto device_addrbits = device.get_info<sycl::info::device::address_bits>();
        auto device_freq = device.get_info<sycl::info::device::max_clock_frequency>();
        auto device_gmem = device.get_info<sycl::info::device::global_mem_size>();
        auto device_maxalloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
        auto device_syclver = device.get_info<sycl::info::device::version>();
        auto device_CUs = device.get_info<sycl::info::device::max_compute_units>();
        auto device_max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

        std::cout << "------------------------ Device specifications ------------------------" << std::endl;
        std::cout << "Device Index:        " << idx + 1 << std::endl;
        std::cout << "Platform:            " << platform_name << std::endl;
        std::cout << "Device:              " << device_name << '/' << vendor_name << std::endl;
        std::cout << "Driver version:      " << device_drv << std::endl;
        std::cout << "Device type:         " << device_type << std::endl;
        std::cout << "Address bits:        " << device_addrbits << std::endl;
        std::cout << "GPU clock rate:      " << device_freq << " MHz" << std::endl;
        std::cout << "Total global mem:    " << device_gmem/1024/1024 << " MB" << std::endl;
        std::cout << "Max allowed buffer:  " << device_maxalloc/1024/1024 << " MB" << std::endl;
        std::cout << "SYCL version:        " << device_syclver << std::endl;
        std::cout << "Total CUs:           " << device_CUs << std::endl;
        std::cout << "Max work group size: " << device_max_work_group_size << std::endl;
        std::cout << "-----------------------------------------------------------------------" << std::endl;
    }
    catch (sycl::exception const &exc) {
        std::cerr << "Could not get full device info: ";
        std::cerr << exc.what() << std::endl;
    }
}

int main() {
    try {
        // Get all available devices
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

        if (devices.empty()) {
            std::cout << "No available GPU devices found." << std::endl;
            return 1;
        }

        // Iterate through each GPU device and print detailed information
        for (std::size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            ShowDeviceInfo(i, device);
        }
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
