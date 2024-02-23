#include <CL/sycl.hpp>
#include <iostream>

/*
!!! Use `xpu-smi discovery` to find your device_id, like 0000:5b:00.0,
    then, replace the following BDF value by your BDF.
*/

constexpr int N = 16;

class DeviceSelector {
public:
    DeviceSelector(std::string PCI_BDF_Address) : PCI_BDF_Address_(PCI_BDF_Address){
        vendorName_ = "Intel";
    };

    int operator()(const sycl::device& dev) const {
        int rating = 0;

        if (dev.has(sycl::aspect::ext_intel_pci_address)) {
            if (dev.is_gpu() && (dev.get_info<sycl::info::device::name>().find(vendorName_) != std::string::npos) &&
                    dev.get_info<sycl::ext::intel::info::device::pci_address>() == PCI_BDF_Address_)
                rating = 4;
                return rating;
        }

        if (dev.is_gpu() && (dev.get_info<sycl::info::device::name>().find(vendorName_) != std::string::npos))
            rating = 3;
        else if (dev.is_gpu()) rating = 2;
        else if (dev.is_cpu()) rating = 1;
        return rating;
    };

private:
    std::string vendorName_;
    std::string PCI_BDF_Address_;
};

sycl::device getMyDevice(std::string PCI_BDF_Address) {
    auto platforms = sycl::platform::get_platforms();
    for (auto &platform : platforms) {
        auto devices = platform.get_devices();
        for (auto &device : devices) {
            if (device.has(sycl::aspect::ext_intel_pci_address)) {
                auto BDF  = device.get_info<sycl::ext::intel::info::device::pci_address>();
                std::cout << "Device BDF : " << BDF  << std::endl;
                if (BDF  == PCI_BDF_Address) {
                    return device;
                }
            }
        }
    }
}

int main() {
    std::string PCI_BDF_Address = "0000:5b:00.0";

    // Option 1
    sycl::queue q1(getMyDevice(PCI_BDF_Address));
    std::cout << "Device: " << q1.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Device: " << q1.get_device().get_info<sycl::ext::intel::info::device::pci_address>() << "\n";

    // Option 2
    DeviceSelector selector(PCI_BDF_Address);
    sycl::queue q2(selector);
    std::cout << "Device: " << q2.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Device: " << q2.get_device().get_info<sycl::ext::intel::info::device::pci_address>() << "\n";
    return 0;
}
