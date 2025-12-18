# Docker Development Environment

CUDA 13.0 + GCC 13 (C++20) on Ubuntu 22.04.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with driver 525+

## Quick Start

```bash
# Build image
docker compose build

# Interactive shell
docker compose run --rm dev

# Inside container: build and test
cmake -B build -DCOBRAML2_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build
```

## Helper Script

```bash
./scripts/docker-dev.sh build   # Build image
./scripts/docker-dev.sh shell   # Interactive shell
./scripts/docker-dev.sh cmake   # Configure build
./scripts/docker-dev.sh make    # Compile
./scripts/docker-dev.sh test    # Run tests
```

## Environment

| Tool   | Version |
|--------|---------|
| CUDA   | 13.0    |
| GCC    | 13      |
| CMake  | 3.28    |
| Ubuntu | 22.04   |

Source is mounted at `/workspace`. Build artifacts persist in a Docker volume.

