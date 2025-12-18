#!/bin/bash
# =============================================================================
# CobraML2 Docker Development Helper Script
# Usage:
#   ./scripts/docker-dev.sh build    - Build the Docker image
#   ./scripts/docker-dev.sh shell    - Start interactive development shell
#   ./scripts/docker-dev.sh cmake    - Configure CMake build
#   ./scripts/docker-dev.sh make     - Build the project
#   ./scripts/docker-dev.sh test     - Run tests
#   ./scripts/docker-dev.sh clean    - Clean build artifacts
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

IMAGE_NAME="cobraml2-dev:cuda13-gcc13"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        log_warn "NVIDIA Docker runtime not detected. GPU access may not work."
        log_warn "Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

case "${1:-help}" in
    build)
        log_info "Building Docker image..."
        docker compose build
        log_info "Image built successfully: $IMAGE_NAME"
        ;;
    
    shell)
        check_nvidia_docker
        log_info "Starting development shell..."
        docker compose run --rm dev /bin/bash
        ;;
    
    cmake)
        check_nvidia_docker
        log_info "Configuring CMake..."
        docker compose run --rm dev cmake -B build -DCOBRAML2_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        log_info "CMake configuration complete. Run './scripts/docker-dev.sh make' to build."
        ;;
    
    make)
        check_nvidia_docker
        log_info "Building project..."
        docker compose run --rm dev cmake --build build --parallel
        log_info "Build complete."
        ;;
    
    test)
        check_nvidia_docker
        log_info "Running tests..."
        docker compose run --rm dev ctest --test-dir build --output-on-failure
        ;;
    
    clean)
        log_info "Cleaning build artifacts..."
        docker compose run --rm dev rm -rf build
        docker volume rm cobraml2_cobraml2-build 2>/dev/null || true
        log_info "Clean complete."
        ;;
    
    exec)
        # Run arbitrary command in container
        shift
        docker compose run --rm dev "$@"
        ;;
    
    help|*)
        echo "CobraML2 Docker Development Helper"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  build   - Build the Docker image"
        echo "  shell   - Start interactive development shell"
        echo "  cmake   - Configure CMake build (with tests enabled)"
        echo "  make    - Build the project"
        echo "  test    - Run tests"
        echo "  clean   - Clean build artifacts"
        echo "  exec    - Run arbitrary command (e.g., $0 exec nvcc --version)"
        echo ""
        echo "Quick start:"
        echo "  $0 build && $0 cmake && $0 make && $0 test"
        ;;
esac

