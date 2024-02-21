#!/bin/bash

icpx array_operation.cpp -o array_operation -fsycl -fsycl-device-code-split=per_kernel