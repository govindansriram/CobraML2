![/assets/](/assets/logo.png)

## About
A header only lib containing SOTA flash attention kernels, mega kernels, and full model serving.
Built from scratch, documenting every step along the way.

## Installation

So far all code has only been tested on systems with `cuda > 12.8` and `ubuntu 22.04`

## Build from source

```bash
git clone https://github.com/govindansriram/CobraML2.git
cd /CobraML2

sudo chmod +x ./runner.sh
```

You can now build the exe's by running

```bash
./runner.sh
```

And run them using

```bash
./runner.sh -r exe_name
```

## Pytorch Integration
...

## Roadmap
1) MHA
    - Iter 1: 287.925 GFLOPs
2) Flash Attention 1
    - Iter 1: 6776.64 GFLOPs
3) Flash Attention 2
4) Flash Attention 3
5) Matmul
    

## Contributing
...

### linting

All files must be formatted to follow the style specified by `clang-format`. 

Ensure `clang-format` is installed. You can confirm this by running `clang-format --version`.

Formatting can be applied to all files by running

```bash
./runner.sh -f
```

Or one file 

```bash
./runner.sh -f /path/to/my/file.cuh
```