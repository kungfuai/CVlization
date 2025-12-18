from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ops_dir = Path(__file__).parent / "ops"
cutlass_dir = ops_dir / "cutlass"

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=--verbose,--warn-on-local-memory-usage",
    "-lineinfo",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-DCUDA_FP8_ENABLED=1",  # Enable FP8 with CUDA 12.8+
    "-Xcompiler",
    "-fPIC"
]

# GPU architectures: Ampere (80, 86), Ada (89), Hopper (90), Blackwell (120)
cc_flag = [
    "-gencode", "arch=compute_120,code=sm_120",
    "-gencode", "arch=compute_90,code=sm_90",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_80,code=sm_80"
]

ext_modules = [
    CUDAExtension(
        name="turbo_diffusion_ops",
        sources=[
            "ops/bindings.cpp",
            "ops/quant/quant.cu",
            "ops/norm/rmsnorm.cu",
            "ops/norm/layernorm.cu",
            "ops/gemm/gemm.cu"
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags + ["-DEXECMODE=0"] + cc_flag + ["--threads", "4"],
        },
        include_dirs=[
            cutlass_dir / "include",
            cutlass_dir / "tools" / "util" / "include",
            ops_dir
        ],
        libraries=["cuda"],
    )
]

setup(
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks")
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
