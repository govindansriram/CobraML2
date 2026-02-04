import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Get absolute paths relative to this file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)

MIN_ARCH = 80

def get_cuda_arch_flags():
    # User override: TORCH_CUDA_ARCH_LIST="8.0;8.9" pip install .
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []

    # Auto-detect from current GPU
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        arch = major * 10 + minor
        if arch < MIN_ARCH:
            raise RuntimeError(f"sm_{arch} not supported. Minimum: sm_{MIN_ARCH}")
        return [f'-gencode=arch=compute_{arch},code=sm_{arch}']

    # No GPU: fat binary
    return [
        '-gencode=arch=compute_80,code=sm_80',
        '-gencode=arch=compute_86,code=sm_86',
        '-gencode=arch=compute_89,code=sm_89',
        '-gencode=arch=compute_90,code=sm_90',
    ]

setup(
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='cobraml._C',
            sources=[
                os.path.join(THIS_DIR, 'csrc/fmha_binding.cu'),
            ],
            include_dirs=[
                os.path.join(ROOT_DIR, 'include'),
                os.path.join(ROOT_DIR, 'external/cutlass/include'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--expt-relaxed-constexpr',
                ] + get_cuda_arch_flags()
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)