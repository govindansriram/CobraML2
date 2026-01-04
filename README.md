![/assets/](/assets/logo.png)

## About
Working our way up to SOTA flash attention kernels, mega kernels, and full model serving, from scratch.

## Build
...

## Pytorch Integration
...

## Roadmap
1) MHA
2) Flash Attention 1
    1) Version 1
    2) performance improvements
3) Flash Attention 2
4) Flash Attention 3
5) Matmul
    

## Contributing
...

### linting
```bash
find include tests -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' | xargs clang-format -i
```