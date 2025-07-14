#!/bin/bash
set -a

export PATH=/home/buildpc/ml/deps/cmake-3.26.1-linux-x86_64/bin:$PATH

if [[ "$NODE_NAME" == "borg" ]]; then
    CUDA_VERSION=128
    DIST_NAME_COMPATIBILITY_STR=vfx2023
    source scl_source enable gcc-toolset-9
elif [[ "$NODE_NAME" == "steel" ]]; then
    CUDA_VERSION=124
    DIST_NAME_COMPATIBILITY_STR=vfx2022
    source scl_source enable devtoolset-9
else
    echo "Unknown NODE_NAME env variable: ${NODE_NAME}. Script is only configured to build on 'borg' and 'steel' nodes"
    exit 1
fi

ORT_VERSION=1.20.2
BUILD_ID=$(date '+%Y-%m-%d')_$(git rev-parse --short HEAD)_${BUILD_NUMBER}
DIST_NAME=libonnxruntime-${ORT_VERSION}_linux_${DIST_NAME_COMPATIBILITY_STR}_cu${CUDA_VERSION}_${BUILD_ID}

./bfx/build_linux.sh $DIST_NAME $CUDA_VERSION

rclone copy build/${DIST_NAME}.zip "mescola:Boris FX/Engineering/BinaryArtifacts"
