from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="prefix_accept",
    ext_modules=[
        CUDAExtension(
            name="prefix_accept",
            sources=["binding.cpp", "prefix_accept.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

