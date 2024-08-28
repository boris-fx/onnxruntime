#!/bin/bash
set -a

export PATH=/home/buildpc/ml/deps/cmake-3.26.1-linux-x86_64/bin:$PATH

source scl_source enable devtoolset-9

ORT_VERSION=1.18.1
BUILD_ID=$(date '+%Y-%m-%d')_$(git rev-parse --short HEAD)_${BUILD_NUMBER}
DIST_NAME=libonnxruntime-${ORT_VERSION}_linux_cu118_${BUILD_ID}

./bfx/build_linux.sh $DIST_NAME

rclone copy build/${DIST_NAME}.zip "mescola:Boris FX/Engineering/BinaryArtifacts"
