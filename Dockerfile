# Use an official Ubuntu base image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CUTLASS_PATH=/opt/cutlass
ENV CUDACXX=/usr/local/cuda-12.8/bin/nvcc

# Install essential build tools, dependencies, and SSH server
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    curl \
    libtbb-dev \
    openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir /var/run/sshd

# Set up SSH key authentication using RunPod's PUBLIC_KEY env var
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Configure SSH for key authentication
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Generate SSH host keys (this is what the working pod shows)
RUN ssh-keygen -A

# Install Google Benchmark
RUN git clone https://github.com/google/benchmark.git /tmp/benchmark && \
    cd /tmp/benchmark && \
    cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -S . -B build && \
    cmake --build build --config Release && \
    cmake --install build --prefix /usr && \
    rm -rf /tmp/benchmark

# Clone and install CUTLASS
RUN git clone https://github.com/NVIDIA/cutlass.git ${CUTLASS_PATH}

# Set C++ standard to C++17
ENV CXXFLAGS="-std=c++17"

# Start SSH service and set up public key from environment
CMD ["/bin/bash", "-c", "if [ -n \"$PUBLIC_KEY\" ]; then echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys; fi && service ssh start && tail -f /dev/null"]