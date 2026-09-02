#!/bin/bash
# usage:
#   bfx/build_linux.sh <DistName> [Release|RelWithDebInfo|Debug] [--backend <list>] [--cuda-version <124|128>] [--parallel <n>] [--noclean]
#
# examples:
#   # everything (what CI does): all backends, Release, CUDA 12.8
#   bfx/build_linux.sh libonnxruntime-1.29.0_linux_vfx2023_cu128_<build-id>
#
#   # quick local iteration: webgpu only, Debug, all cores, reusing the previous build tree
#   bfx/build_linux.sh scratch Debug --backend webgpu --parallel 0 --noclean
#
#   # chasing a crash that only reproduces optimized: same codegen as Release, but with symbols
#   bfx/build_linux.sh scratch RelWithDebInfo --backend cuda --parallel 0 --noclean
#
# notes:
#   - this mirrors bfx/build_win.ps1's options. there is no 'dml' backend on linux (DirectML is a
#     windows-only API), so --backend's choices here are 'cuda' and 'webgpu'.
#   - webgpu on linux runs Dawn's *Vulkan* backend (windows uses D3D12). Dawn is linked statically
#     into libonnxruntime_providers_webgpu.so, but the host still needs a vulkan loader
#     (libvulkan.so.1) plus an ICD/driver at *runtime*.
#   - our bfx custom ops are DML-only, so they are NOT present in a webgpu build.
#   - --parallel defaults to 1 so the default (CI) invocation builds exactly like it always has.
#     '--parallel 0' means one job per core - safe and much faster for any build without CUDA in it.
#   - --cuda-version only matters when 'cuda' is in --backend; it is ignored otherwise.
set -e

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \?//'
    exit 1
}

# RelWithDebInfo is Release codegen (-O2, NDEBUG) plus symbols - use it to debug a crash that only
# reproduces in an optimized build.
BUILD_CONFIG="Release"
BACKENDS="cuda,webgpu"
CUDA_VERSION="128"
# build.py --parallel, where 0 means 'one job per core'. defaults to 1, which is what CI has always
# used - nvcc on the flash attention kernels OOMs the machine otherwise. a build with no CUDA in it
# has no such problem, so pass '--parallel 0' for those; it is far faster.
PARALLEL=1
# keep the existing build/ tree: incremental rebuild, and no re-download of the CUDA SDKs
NOCLEAN=0

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      BACKENDS="$2"; shift 2 ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --parallel)     PARALLEL="$2"; shift 2 ;;
        --noclean)      NOCLEAN=1; shift ;;
        -h|--help)      usage ;;
        -*)             echo "unknown option: $1" >&2; usage ;;
        *)              POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -lt 1 || ${#POSITIONAL[@]} -gt 2 ]]; then
    echo "expected <DistName> and an optional build config" >&2
    usage
fi
DIST_NAME="${POSITIONAL[0]}"
if [[ ${#POSITIONAL[@]} -eq 2 ]]; then BUILD_CONFIG="${POSITIONAL[1]}"; fi

case "$BUILD_CONFIG" in
    Release|RelWithDebInfo|Debug) ;;
    *) echo "invalid build config: ${BUILD_CONFIG}. must be Release, RelWithDebInfo or Debug" >&2; exit 1 ;;
esac

USE_CUDA=0
USE_WEBGPU=0
IFS=',' read -ra BACKEND_LIST <<< "$BACKENDS"
for b in "${BACKEND_LIST[@]}"; do
    case "$b" in
        cuda)   USE_CUDA=1 ;;
        webgpu) USE_WEBGPU=1 ;;
        dml)    echo "the 'dml' backend is windows-only (DirectML); use bfx/build_win.ps1 for it" >&2; exit 1 ;;
        *)      echo "invalid backend: '${b}'. must be one of: cuda, webgpu" >&2; exit 1 ;;
    esac
done
if [[ $USE_CUDA -eq 0 && $USE_WEBGPU -eq 0 ]]; then
    echo "--backend must name at least one of: cuda, webgpu" >&2
    exit 1
fi

if [[ $USE_CUDA -eq 1 ]]; then
    if [[ "$CUDA_VERSION" == "124" ]]; then
        CUDA_SDK_NAME=cuda-sdk-linux-v12.4.1
        CUDNN_NAME=cudnn-linux-x86_64-8.9.7.29_cuda12-archive
        CMAKE_CUDA_FLAGS=""
        CMAKE_CUDA_ARCHITECTURES="60-real;61-real;70-real;75-real;80-real;86-real;89-real;90a-real;90-real;90-virtual"
    elif [[ "$CUDA_VERSION" == "128" ]]; then
        CUDA_SDK_NAME=cuda-sdk-linux-v12.8.1
        CUDNN_NAME=cudnn-linux-x86_64-9.10.2.21_cuda12-archive
        CMAKE_CUDA_FLAGS="-static-global-template-stub=false" # this needed for compilation when switching to CU 12.8 from 12.4
        CMAKE_CUDA_ARCHITECTURES="60-real;61-real;70-real;75-real;80-real;86-real;89-real;90a-real;90-real;90-virtual;120-real;120-virtual"
    else
        echo "Unknown cuda version passed to build script: ${CUDA_VERSION}. Must be 124 or 128" >&2
        exit 1
    fi
fi

echo "starting onnxruntime build: ${DIST_NAME}"
echo "  config:   ${BUILD_CONFIG}"
echo "  backends: ${BACKENDS}"
if [[ $USE_CUDA -eq 1 ]]; then echo "  cuda:     ${CUDA_VERSION}"; fi
echo "  parallel: ${PARALLEL}"
if [[ $NOCLEAN -eq 1 ]]; then echo "  clean:    false"; else echo "  clean:    true"; fi

echo GCC info:
which gcc
gcc --version
echo CMAKE info:
which cmake
cmake --version
which ldd
ldd --version

# clear any previous build.. (--noclean keeps it, for incremental rebuilds)
if [[ $NOCLEAN -eq 0 ]]; then rm -rf build; fi
mkdir -p build

CUDA_HOME=""
CUDNN_HOME=""

if [[ $USE_CUDA -eq 1 ]]; then
    echo '-- fetching dependencies --'
    # fetch CUDA dependencies from BinaryArtifacts
    BINARY_ARTIFACTS="mescola:Boris FX/Engineering/BinaryArtifacts"

    cd build
    CUDA_HOME=$(pwd)/${CUDA_SDK_NAME}
    CUDNN_HOME=$(pwd)/${CUDNN_NAME}

    if [[ -d "${CUDA_SDK_NAME}" ]]; then
        echo "-- ${CUDA_SDK_NAME} already unpacked, skipping fetch --"
    else
        rclone copy "${BINARY_ARTIFACTS}/${CUDA_SDK_NAME}.tgz" .
        tar -xf ${CUDA_SDK_NAME}.tgz
    fi

    if [[ -d "${CUDNN_NAME}" ]]; then
        echo "-- ${CUDNN_NAME} already unpacked, skipping fetch --"
    else
        rclone copy "${BINARY_ARTIFACTS}/${CUDNN_NAME}.txz" .
        tar -xf ${CUDNN_NAME}.txz
    fi
    cd ..
else
    echo '-- skipping CUDA dependency fetch (no cuda build selected) --'
fi

eval "$(conda shell.bash hook)"
conda create -n ort_build python=3.12 -y 2>/dev/null || true
conda activate ort_build

# args shared by every backend; backend-specific args are appended below
BUILD_ARGS=(--config "$BUILD_CONFIG" --build_shared_lib --skip_tests --parallel "$PARALLEL")
CMAKE_EXTRA_DEFINES=(onnxruntime_BUILD_UNIT_TESTS=OFF)

if [[ $USE_WEBGPU -eq 1 ]]; then
    BUILD_ARGS+=(--use_webgpu shared_lib)
fi

if [[ $USE_CUDA -eq 1 ]]; then
    BUILD_ARGS+=(--use_cuda --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" --nvcc_threads 1 --flash_nvcc_threads 1)
    if [[ -n "$CMAKE_CUDA_FLAGS" ]]; then CMAKE_EXTRA_DEFINES+=("CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}"); fi
    CMAKE_EXTRA_DEFINES+=("CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}" onnxruntime_USE_FLASH_ATTENTION:BOOL=ON)
fi

# --cmake_extra_defines is variadic, so it has to come last
BUILD_ARGS+=(--cmake_extra_defines "${CMAKE_EXTRA_DEFINES[@]}")

echo '-- running build --'
echo "-- command: ./build.sh ${BUILD_ARGS[*]}"
./build.sh "${BUILD_ARGS[@]}"

BUILD_LIB_DIR=build/Linux/${BUILD_CONFIG}

if [[ $USE_CUDA -eq 1 ]]; then
    if [[ "$CUDA_VERSION" == "124" ]]; then
        echo TODO: add full cuDNN 8 versions to libonnxruntime_providers_cuda.so
    else
        # convert libonnxruntime_providers_cuda.so to use full cudnn library names to avoid conflicts with hosts that have their own cuDNN!
        patchelf --replace-needed libcudnn.so.9                             libcudnn.so.9.10.2                                  ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
        patchelf --replace-needed libcudnn_adv.so.9                         libcudnn_adv.so.9.10.2                              ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
        patchelf --replace-needed libcudnn_ops.so.9                         libcudnn_ops.so.9.10.2                              ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
        patchelf --replace-needed libcudnn_cnn.so.9                         libcudnn_cnn.so.9.10.2                              ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
        patchelf --replace-needed libcudnn_graph.so.9                       libcudnn_graph.so.9.10.2                            ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
        patchelf --replace-needed libcudnn_engines_runtime_compiled.so.9    libcudnn_engines_runtime_compiled.so.9.10.2         ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
        patchelf --replace-needed libcudnn_engines_precompiled.so.9         libcudnn_engines_precompiled.so.9.10.2              ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
        patchelf --replace-needed libcudnn_heuristic.so.9                   libcudnn_heuristic.so.9.10.2                        ${BUILD_LIB_DIR}/libonnxruntime_providers_cuda.so
    fi
fi

# put into release dir
DIST_DIR=build/${DIST_NAME}
rm -rf ${DIST_DIR} ${DIST_DIR}.zip
mkdir -p ${DIST_DIR}/lib

LIBS=(libonnxruntime.so libonnxruntime_providers_shared.so)
if [[ $USE_CUDA -eq 1 ]];   then LIBS+=(libonnxruntime_providers_cuda.so); fi
if [[ $USE_WEBGPU -eq 1 ]]; then LIBS+=(libonnxruntime_providers_webgpu.so); fi
for LIB in "${LIBS[@]}"; do
    cp ${BUILD_LIB_DIR}/${LIB} ${DIST_DIR}/lib
done
cp -r include ${DIST_DIR}/.

cd build
zip -r $(basename ${DIST_NAME}).zip $(basename ${DIST_NAME})
cd ..

echo "onnxruntime build completed, artifacts generated:"
echo "  ${DIST_DIR}.zip"
