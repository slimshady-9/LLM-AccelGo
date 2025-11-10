#include <torch/extension.h>
#include <cstdint>

// Forward declaration of CUDA launcher
int64_t launch_prefix_accept(torch::Tensor logits, torch::Tensor proposal);

// pybind11 module interface (importable as the extension name)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA prefix acceptance: returns accepted prefix length";
  m.def("accept_len", &launch_prefix_accept, "Compute accepted prefix length (CUDA)");
}

// TorchScript operator registration for LibTorch and torch.ops usage
TORCH_LIBRARY(prefixacc, m) {
  // Use `int` in the schema string (Py-side), but the C++ kernel must use int64_t
  m.def("accept_len(Tensor logits, Tensor proposal) -> int");
}

TORCH_LIBRARY_IMPL(prefixacc, CUDA, m) {
  m.impl("accept_len", &launch_prefix_accept);
}

