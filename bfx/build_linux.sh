#!/bin/bash
set -e

DIST_NAME=$1
CUDA_VERSION=$2

echo starting onnxruntime build: $DIST_NAME, running CUDA $CUDA_VERSION

echo GCC info:
which gcc
gcc --version
echo CMAKE info:
which cmake
cmake --version
which ldd
ldd --version

rm -rf build
mkdir build

# fetch CUDA dependencies from BinaryArtifacts
BINARY_ARTIFACTS="mescola:Boris FX/Engineering/BinaryArtifacts"

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
    echo "Unknown CUDA_VERSION passed to build script: ${CUDA_VERSION}. Must be 124 or 128"
    exit 1
fi

cd build
rclone copy "${BINARY_ARTIFACTS}/${CUDA_SDK_NAME}.tgz" .
tar -xf ${CUDA_SDK_NAME}.tgz
CUDA_HOME=$(pwd)/${CUDA_SDK_NAME}

rclone copy "${BINARY_ARTIFACTS}/${CUDNN_NAME}.txz" .
tar -xf ${CUDNN_NAME}.txz
CUDNN_HOME=$(pwd)/${CUDNN_NAME}
cd ..

eval "$(conda shell.bash hook)"
conda activate base

./build.sh --config Release \
    --build_shared_lib \
    --use_cuda \
    --cuda_home $CUDA_HOME \
    --cudnn_home $CUDNN_HOME \
    --skip_tests \
    --cmake_extra_defines \
        CMAKE_CUDA_FLAGS=$CMAKE_CUDA_FLAGS \
        CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES \
        onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_FLASH_ATTENTION=OFF

# put into release dir
mkdir build/dist_release
mkdir build/dist_release/lib
cp build/Linux/Release/libonnxruntime.so build/dist_release/lib
# cp build/Linux/Release/libonnxruntime.so.1.18.1 build/dist_release/lib
cp build/Linux/Release/libonnxruntime_providers_shared.so build/dist_release/lib
cp build/Linux/Release/libonnxruntime_providers_cuda.so build/dist_release/lib
cp -r include build/dist_release/.

mv build/dist_release build/$DIST_NAME
cd build
zip -r $DIST_NAME.zip $(basename $DIST_NAME)
cd ..
