#!/bin/bash

set -e

BUILD_DIR="build"
CLEAN=false
TESTS=ON
BENCHMARK=OFF
TARGET=""
RUN_TARGET=""
RUN_ALL=false
FORMAT=false
FORMAT_FILE=""
PROFILE_TARGET=""
PROFILE_OPTS=""
PROFILE_OUTPUT=""
RUN_ARGS=()

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help"
    echo "  -c, --clean               Clean build (remove build directory)"
    echo "  -b, --benchmark           Enable benchmarking"
    echo "  -t, --target <target>     Build specific target (e.g., test_mha)"
    echo "  -r, --run <target>        Build and run specific target"
    echo "  -a, --run-all             Build and run all tests (ctest)"
    echo "  -f, --format [file]       Format all files, or specific file if provided"
    echo "  -p, --profile <target>    Build and profile target with ncu"
    echo "  -o, --output <name>       Name for the .ncu-rep file (default: target name)"
    echo "  --profile-opts <opts>     Additional ncu options"
    echo "  --no-tests                Disable building tests"
    echo "  --                        Remaining args passed to executable/ctest"
    echo ""
    echo "Examples:"
    echo "  $0                        # Build all"
    echo "  $0 -c                     # Clean build"
    echo "  $0 -t test_mha            # Build only test_mha"
    echo "  $0 -r test_mha            # Build and run test_mha"
    echo "  $0 -a                     # Build and run all tests"
    echo "  $0 -f                     # Format all source files"
    echo "  $0 -f src/main.cu         # Format specific file"
    echo "  $0 -c -b -r test_mha      # Clean build with benchmarking, then run"
    echo "  $0 -r test_mha -- --gtest_filter=*Perf*"
    echo "  $0 -p test_fmha_cc        # Build and profile with ncu"
    echo "  $0 -p test_fmha_cc -o my_profile  # Custom report name"
    echo "  $0 -p test_fmha_cc --profile-opts '--kernel-name fmha'"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -b|--benchmark)
            BENCHMARK=ON
            shift
            ;;
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -r|--run)
            RUN_TARGET="$2"
            TARGET="$2"
            shift 2
            ;;
        -a|--run-all)
            RUN_ALL=true
            shift
            ;;
        -f|--format)
            FORMAT=true
            # Check if next arg is a file (not another flag)
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                FORMAT_FILE="$2"
                shift
            fi
            shift
            ;;
        -p|--profile)
            PROFILE_TARGET="$2"
            TARGET="$2"
            shift 2
            ;;
        -o|--output)
            PROFILE_OUTPUT="$2"
            shift 2
            ;;
        --profile-opts)
            PROFILE_OPTS="$2"
            shift 2
            ;;
        --no-tests)
            TESTS=OFF
            shift
            ;;
        --)
            shift
            RUN_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Format if requested
if [ "$FORMAT" = true ]; then
    if [ -n "$FORMAT_FILE" ]; then
        echo "Formatting: $FORMAT_FILE"
        clang-format -i "$FORMAT_FILE"
    else
        echo "Formatting all source files..."
        find include tests -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' | xargs clang-format -i
    fi
    echo "Done formatting."
    # Exit if only formatting was requested
    if [ "$CLEAN" = false ] && [ -z "$TARGET" ] && [ -z "$RUN_TARGET" ] && [ "$RUN_ALL" = false ]; then
        exit 0
    fi
fi

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

# Configure
echo "Configuring (BENCHMARK=${BENCHMARK})..."
cmake -B "$BUILD_DIR" \
    -DCOBRAML2_BUILD_TESTS="$TESTS" \
    -DBENCHMARK="$BENCHMARK"

# Build
if [ -n "$TARGET" ]; then
    echo "Building target: $TARGET"
    cmake --build "$BUILD_DIR" --target "$TARGET" --parallel
else
    echo "Building all..."
    cmake --build "$BUILD_DIR" --parallel
fi

# Run all tests if requested
if [ "$RUN_ALL" = true ]; then
    echo ""
    echo "Running all tests..."
    echo "----------------------------------------"
    ctest --test-dir "$BUILD_DIR" --output-on-failure "${RUN_ARGS[@]}"
fi

# Run specific target if requested
if [ -n "$RUN_TARGET" ]; then
    echo ""
    echo "Running: $RUN_TARGET ${RUN_ARGS[*]}"
    echo "----------------------------------------"
    "$BUILD_DIR/tests/$RUN_TARGET" "${RUN_ARGS[@]}"
fi

# Profile specific target if requested
if [ -n "$PROFILE_TARGET" ]; then
    REPORT_NAME="${PROFILE_OUTPUT:-$PROFILE_TARGET}"
    echo ""
    echo "Profiling: $PROFILE_TARGET ${RUN_ARGS[*]}"
    echo "----------------------------------------"
    ncu \
        -o "$REPORT_NAME" \
        -f \
        --set=full \
        --import-source=yes \
        $PROFILE_OPTS \
        "$BUILD_DIR/tests/$PROFILE_TARGET" "${RUN_ARGS[@]}"
    echo ""
    echo "Report saved: ${REPORT_NAME}.ncu-rep"
fi