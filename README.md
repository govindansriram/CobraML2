![/assets/](/assets/logo.png)

## About
A header only lib containing SOTA flash attention kernels, mega kernels, and full model serving.
Built from scratch, documenting every step along the way.

## Installation

### Prerequisites
- Python >= 3.10
- CUDA toolkit with `nvcc`
- An NVIDIA GPU with compute capability >= 8.0 (Ampere+)

So far all code has only been tested on systems with `CUDA >= 12.8` and `Ubuntu 22.04`

## Get from source
```bash
git clone https://github.com/govindansriram/CobraML2.git
cd CobraML2

sudo chmod +x ./runner.sh
```

## Building the Python Package

### Setup

The runner handles venv creation, CUDA-matched torch installation, and building in one go:

```bash
# First time: creates .venv and installs torch matching your system CUDA version
./runner.sh --python-setup

# Build and install cobraml in editable mode
./runner.sh --build-python
```

## Testing

### C++ tests

The C++ tests require the `.venv` to exist at the project root, since CMake uses it to locate PyTorch headers. Run `./runner.sh --python-setup` at least once before building C++ targets.

```bash
# Build and run a specific test
./runner.sh -r test_fmha_cc

# Run all tests
./runner.sh -a

# Run with benchmarking enabled
./runner.sh -c -b -r test_fmha_cc

# Filter specific test cases
./runner.sh -r test_fmha_cc -- --gtest_filter=*causal*
```

### Python tests

Requires the Python package to be built first (`./runner.sh --build-python`).

```bash
# Run all Python tests
./runner.sh --python-test

# Run with benchmarking
./runner.sh --python-benchmark

# Pass extra args to pytest
./runner.sh --python-test -- -k "test_fmha_fp32[4-512-16-64-True]"
```

## Roadmap

1. Flash Attention 1 (done)
   - between 2 and 4 x faster then pytorch naive MHA
2. KV cache and inference serving 
3. Flash Attention 2
4. Flash Attention 3
5. Matmul

## Using the Runner

The `runner.sh` script is the main entry point for building, testing, profiling, and formatting the project.

### Options

| Flag | Description |
|------|-------------|
| `-h, --help` | Show help message |
| `-c, --clean` | Clean build (removes build directory) |
| `-b, --benchmark` | Enable benchmarking |
| `-t, --target <name>` | Build specific target |
| `-r, --run <name>` | Build and run specific target |
| `-a, --run-all` | Build and run all tests via ctest |
| `-f, --format [file]` | Format all files, or a specific file |
| `-p, --profile <name>` | Build and profile target with ncu |
| `-o, --output <name>` | Custom name for .ncu-rep file |
| `--profile-opts <opts>` | Additional ncu options |
| `--no-tests` | Disable building tests |
| `--python-setup` | Create .venv and install build deps |
| `--build-python` | Build and install cobraml Python package |
| `--python-test` | Run Python tests |
| `--python-benchmark` | Run Python benchmarks |
| `--` | Pass remaining args to executable |

### Examples

**Build everything:**
```bash
./runner.sh
```

**Clean build:**
```bash
./runner.sh -c
```

**Build specific target:**
```bash
./runner.sh -t test_fmha_cc
```

**Build and run a test:**
```bash
./runner.sh -r test_fmha_cc
```

**Run all tests:**
```bash
./runner.sh -a
```

**Run with gtest filter:**
```bash
./runner.sh -r test_fmha_cc -- --gtest_filter=*Perf*
```

**Clean build with benchmarking enabled:**
```bash
./runner.sh -c -b -r test_fmha_cc
```

**Profile a kernel with ncu:**
```bash
./runner.sh -p test_fmha_cc
```

**Profile with custom output name:**
```bash
./runner.sh -p test_fmha_cc -o my_profile
```

**Profile specific kernel:**
```bash
./runner.sh -p test_fmha_cc --profile-opts '--kernel-name fmha'
```

### Linting

All files must be formatted to follow the style specified by `clang-format`.

Ensure clang-format is installed by running `clang-format --version`.

**Format all files:**
```bash
./runner.sh -f
```

**Format a specific file:**
```bash
./runner.sh -f include/cobraml2/kernels/fmha_cc.cuh
```

## Contributing

...