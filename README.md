![/assets/](/assets/logo.png)

## About
A header only lib containing SOTA flash attention kernels, mega kernels, and full model serving.
Built from scratch, documenting every step along the way.

## Installation

So far all code has only been tested on systems with `CUDA >= 12.8` and `Ubuntu 22.04`

### Build from source
```bash
git clone https://github.com/govindansriram/CobraML2.git
cd CobraML2

sudo chmod +x ./runner.sh
```

You can now build the executables by running:
```bash
./runner.sh
```

And run them using:
```bash
./runner.sh -r exe_name
```

## PyTorch Integration

...

## Roadmap

1. MHA
   - Iter 1: 287.925 GFLOPs
2. Flash Attention 1
   - Iter 1: 7715.79 GFLOPs
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