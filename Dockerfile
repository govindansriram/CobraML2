# =============================================================================
# CobraML2 Development Environment
# CUDA 13.0 + GCC 13 (C++20) on Ubuntu 22.04
# =============================================================================
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set locale to avoid encoding issues
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# =============================================================================
# Install build essentials and GCC 13 for full C++20 support
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    software-properties-common \
    # Version control
    git \
    # CMake (we'll install a newer version below)
    wget \
    ca-certificates \
    # Debugging and profiling tools
    gdb \
    valgrind \
    linux-tools-generic \
    # Misc utilities
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Add GCC 13 PPA and install (best C++20/23 support)
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc-13 \
    g++-13 \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 13 as the default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-13 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100

# =============================================================================
# Install CMake 3.28+ (required for modern CUDA + C++20 features)
# =============================================================================
ARG CMAKE_VERSION=3.28.3
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# =============================================================================
# Configure CUDA to use GCC 13 as host compiler
# =============================================================================
ENV CUDAHOSTCXX=/usr/bin/g++-13
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# Set working directory
# =============================================================================
WORKDIR /workspace

# =============================================================================
# Verify installations
# =============================================================================
RUN echo "=== Environment Verification ===" && \
    echo "GCC version:" && gcc --version | head -1 && \
    echo "G++ version:" && g++ --version | head -1 && \
    echo "CMake version:" && cmake --version | head -1 && \
    echo "NVCC version:" && nvcc --version | tail -1 && \
    echo "CUDA Host Compiler: ${CUDAHOSTCXX}"

# Default command: open a shell
CMD ["/bin/bash"]

