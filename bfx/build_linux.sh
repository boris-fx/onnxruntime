#!/bin/bash
set -a

DIST_NAME=$1

echo starting onnxruntime on build: $DIST_NAME

rm -rf build
mkdir build

cd build
# fetch CUDA dependencies from BinaryArtifacts
BINARY_ARTIFACTS="mescola:Boris FX/Engineering/BinaryArtifacts"

CUDA_SDK_NAME='cuda_11.8.0_520.61.05_linux'
rclone copy ${BINARY_ARTIFACTS}/${CUDA_SDK_NAME}.tgz .
tar -xf ${CUDA_SDK_NAME}.tgz
CUDA_HOME=$(pwd)/cuda_sdk

CUDNN_NAME='cudnn-linux-x86_64-8.9.4.25_cuda11-archive'
rclone copy ${BINARY_ARTIFACTS}/${CUDNN_NAME}.txz .
tar -xzf ${CUDNN_NAME}.txz
CUDNN_HOME=$(pwd)/cudnn
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
        onnxruntime_BUILD_UNIT_TESTS=OFF

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
