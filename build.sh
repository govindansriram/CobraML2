#!/bin/bash
set -e

# Define the build directory
BUILD_DIR="build"
LIB_DIR="lib"
LOG_DIR="logs"
BENCHMARK=false
THREAD_SANITIZE=false
ADDITIONAL_COMPILE_OPTIONS=""
THREAD_COUNT=1
SESSION_NAME="default"

# Function to display usage
usage() {
    echo "Usage: $0 [-t THREAD_SANITIZE] [-b BENCHMARK] [-a BENCHMARK]"
    echo "  -b      benchmark mode"
    echo "  -s     turn on thread sanitization for test mode"
    echo "  -a      additional compile options (e.g., -march=native;-ffast-math)"
    echo "  -t      how many threads to run the application on defaults to 1"
    echo "  -n      the name of the benchmarking session"
    exit 1
}

while getopts "bst:a:n:" opt; do
    case $opt in
        b) BENCHMARK=true ;;
        t) THREAD_COUNT="$OPTARG";;
        s) THREAD_SANITIZE=true ;;
        a) ADDITIONAL_COMPILE_OPTIONS="$OPTARG";;
        n) SESSION_NAME="$OPTARG"
    esac
done

# Check if the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating $BUILD_DIR directory..."
    mkdir "$BUILD_DIR"
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "Creating $LOG_DIR directory..."
    mkdir "$LOG_DIR"
fi

cd "$BUILD_DIR" || exit

CMAKE_CMD="cmake -DTHREAD_COUNT=$THREAD_COUNT"

if [ -n "$ADDITIONAL_COMPILE_OPTIONS" ]; then
    CMAKE_CMD+=" -DADDITIONAL_COMPILE_OPTIONS=\"$ADDITIONAL_COMPILE_OPTIONS\""
fi

echo

if [ "$BENCHMARK" = true ]; then
   echo "BENCHMARK MODE"
   CMAKE_CMD+=" -DENABLE_TESTING=OFF"
else
  echo "TEST MODE"
  CMAKE_CMD+=" -DENABLE_TESTING=ON"

  echo "Running Address Sanitization"

  if [ "$THREAD_SANITIZE" = true ]; then
    echo "Running thread Sanitization"
    CMAKE_CMD+=" -DIS_THREAD=ON"
  else
    CMAKE_CMD+=" -DIS_THREAD=OFF"
  fi
fi

echo
echo "Running: $CMAKE_CMD .."
echo

eval "$CMAKE_CMD .."
make

echo
echo "finished build"
echo

if $BENCHMARK; then
    pattern="benchmark*"
else
    pattern="test*"
fi

for exe in $pattern; do
    # Check if it's an executable file
    if [ -x "$exe" ] && [ ! -d "$exe" ]; then
        echo "Running $exe..."
        echo "----------------------------------------------------"
        echo

        if $BENCHMARK; then
            PTH="../benchmarks/reports/$exe/$SESSION_NAME"
            mkdir -p "$PTH"
            REPORT_PTH="$PTH/report.json"
            CONFIG_PTH="$PTH/config.txt"
            truncate -s 0 "$CONFIG_PTH"
            echo "$ADDITIONAL_COMPILE_OPTIONS" >> "$CONFIG_PTH"
            echo "$THREAD_COUNT" >> "$CONFIG_PTH"
            ./"$exe" --benchmark_counters_tabular=true --benchmark_format=console --benchmark_out="$REPORT_PTH" --benchmark_filter="BatchedDotProductAll"
        else
            ./"$exe"
        fi

        echo
    fi
done