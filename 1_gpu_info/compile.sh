#!/bin/bash

icpx gpu_info.cpp -o gpu_info -fsycl -fsycl-device-code-split=per_kernel