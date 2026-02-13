![/assets/](/assets/logo.png)

## About
LLM Inference Serving platform with flash attention kernels, mega kernels, and much more.
Built from scratch, documenting every step along the way.

## Installation

### Prerequisites
- Python >= 3.10
- CUDA toolkit with `nvcc`
- An NVIDIA GPU with compute capability >= 8.0 (Ampere+)

So far all code has only been tested on systems with `CUDA >= 12.8` and `Ubuntu 22.04`

### Build from source

#### Initial setup
```bash
git clone https://github.com/govindansriram/CobraML2.git
cd CobraML2

sudo chmod +x ./runner.sh
```

#### Python package

Install torch for your CUDA version, then build cobraml:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install --no-build-isolation -e ".[dev]"
```

Replace `cu130` with your CUDA version: `cu124`, `cu126`, `cu128`, etc. Check with `nvcc --version`.

#### C++ targets

The C++ build uses CMake to locate PyTorch headers from the `.venv`. You don't need to build the full Python package first, but torch must be installed in the venv.

#### Testing

##### C++ tests

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

##### Python tests

Requires the Python package to be built first.

```bash
# Run all tests
pytest

# Run with benchmarking
pytest --benchmark

# Filter specific test cases
pytest -k "test_fmha_fp32[4-512-16-64-True]"
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

#### C++

All C++ files must be formatted with `clang-format`.

```bash
./runner.sh -f
./runner.sh -f include/cobraml2/kernels/fmha_cc.cuh
```

#### Python

```bash
ruff check python/
ruff format python/
```

## Contributing

...