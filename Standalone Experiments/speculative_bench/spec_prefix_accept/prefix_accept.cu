// CUDA prefix acceptance kernel and launcher
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <limits>
#include <cstdint>

template <typename scalar_t>
__global__ void argmax_and_prefix_kernel(const scalar_t* __restrict__ logits,  // [K,V]
                                         const int64_t* __restrict__ prop,    // [K]
                                         int*          __restrict__ first_mismatch, // scalar
                                         int K, int V) {
  extern __shared__ float shmem[]; // size = threads*(sizeof(float)+sizeof(int))
  float* svals = shmem;
  int*   sidx  = reinterpret_cast<int*>(svals + blockDim.x);

  int row = blockIdx.x; // 0..K-1
  if (row >= K) return;

  const scalar_t* row_ptr = logits + row * V;

  float local_max = -std::numeric_limits<float>::infinity();
  int   local_idx = -1;

  // Stride over the vocab dimension
  for (int col = threadIdx.x; col < V; col += blockDim.x) {
    float val;
    if constexpr (std::is_same<scalar_t, __half>::value) {
      val = __half2float(reinterpret_cast<const __half*>(row_ptr)[col]);
    } else {
      val = static_cast<float>(row_ptr[col]);
    }
    if (val > local_max) { local_max = val; local_idx = col; }
  }

  svals[threadIdx.x] = local_max;
  sidx [threadIdx.x] = local_idx;
  __syncthreads();

  // Parallel reduction to find the block’s argmax
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (svals[threadIdx.x + stride] > svals[threadIdx.x]) {
        svals[threadIdx.x] = svals[threadIdx.x + stride];
        sidx [threadIdx.x] = sidx [threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int vtok = sidx[0];        // verifier argmax at this row
    int64_t dtok = prop[row];  // draft proposal token id for this row
    if (vtok != static_cast<int>(dtok)) {
      // record the smallest row index with mismatch
      atomicMin(first_mismatch, row);
    }
  }
}

int launch_prefix_accept(torch::Tensor logits,   // [K,V] float16/float32 CUDA
                         torch::Tensor proposal  // [K]   int64           CUDA/CPU
                         ) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(logits.dim() == 2, "logits shape must be [K,V]");
  TORCH_CHECK(proposal.dim() == 1, "proposal shape must be [K]");

  auto logits_c = logits.contiguous();
  int K = static_cast<int>(logits_c.size(0));
  int V = static_cast<int>(logits_c.size(1));
  TORCH_CHECK(K >= 0 && V > 0, "invalid shapes for logits");

  // Ensure proposal is CUDA int64 and contiguous
  torch::Tensor prop = proposal;
  if (prop.scalar_type() != torch::kLong) prop = prop.to(torch::kLong);
  if (!prop.is_cuda()) prop = prop.to(torch::kCUDA);
  prop = prop.contiguous();
  TORCH_CHECK(prop.size(0) == K, "proposal length must match logits K");

  auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto first_mismatch = torch::full({1}, K, opts_i); // init to K (assume all match)

  int threads = 256;
  int blocks  = K;
  size_t shmem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int));

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

  if (logits_c.scalar_type() == torch::kFloat16) {
    const __half* ptr = reinterpret_cast<const __half*>(logits_c.data_ptr<at::Half>());
    argmax_and_prefix_kernel<__half><<<blocks, threads, shmem, stream.stream()>>>(
      ptr,
      prop.data_ptr<int64_t>(),
      first_mismatch.data_ptr<int>(),
      K, V
    );
  } else if (logits_c.scalar_type() == torch::kFloat32) {
    const float* ptr = logits_c.data_ptr<float>();
    argmax_and_prefix_kernel<float><<<blocks, threads, shmem, stream.stream()>>>(
      ptr,
      prop.data_ptr<int64_t>(),
      first_mismatch.data_ptr<int>(),
      K, V
    );
  } else {
    TORCH_CHECK(false, "logits must be float16 or float32");
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  // Synchronize the current stream to ensure result is ready for host read
  AT_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

  int h = first_mismatch.cpu().item<int>(); // 0..K
  // accepted prefix length = index of first mismatch (K if none)
  return h;
}

