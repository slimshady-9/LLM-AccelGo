#!/usr/bin/env bash
(cd "$(dirname "$0")/cuda" && nvcc -std=c++14 -O3 -Xcompiler -fPIC -shared -arch="${CUDA_ARCH:-sm_90}" -o libspecdecode.so spec_decode.cu -lcudart) && LD_LIBRARY_PATH="$(dirname "$0")/cuda:${LD_LIBRARY_PATH:-}" go run .
