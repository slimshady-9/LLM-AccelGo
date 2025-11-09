#!/usr/bin/env bash
set -euo pipefail; nvcc -std=c++14 -shared -Xlinker -soname,libspecdecode.so -o cuda/libspecdecode.so cuda/spec_decode.cu -lcudart && LD_LIBRARY_PATH="$(pwd)/cuda:${LD_LIBRARY_PATH:-}" go run . "$@"
