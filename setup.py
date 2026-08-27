from pathlib import Path

from setuptools import find_packages, setup


def build_extensions():
    # The serving client and tests can be installed without CUDA/PyTorch. The
    # extension is built only inside the vLLM CUDA image by BUILD_FUSED_OP=1.
    import os

    if os.environ.get("BUILD_FUSED_OP") != "1":
        return []
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    return [
        CUDAExtension(
            name="fused_swiglu._C",
            sources=["csrc/fused_swiglu.cpp", "csrc/fused_swiglu_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ]


cmdclass = {}
if build_extensions():
    from torch.utils.cpp_extension import BuildExtension

    cmdclass["build_ext"] = BuildExtension

setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    py_modules=["sitecustomize"],
    ext_modules=build_extensions(),
    cmdclass=cmdclass,
)
