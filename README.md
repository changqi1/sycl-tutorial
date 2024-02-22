# sycl-tutorial

## How to build

```bash
source /opt/intel/oneapi/setvars.sh
export CC=icx
export CXX=icpx

cd sycl-tutorial
mkdir build && cd build
cmake ..
make -j
```
