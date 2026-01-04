


## linting
```bash
find include tests -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' | xargs clang-format -i
```