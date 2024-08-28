#!/bin/bash
set -a

export PATH=/home/buildpc/ml/deps/cmake-3.26.1-linux-x86_64/bin:$PATH

source scl_source enable devtoolset-9

BUILD_ID=$(date '+%Y-%m-%d')_$(git rev-parse --short HEAD)_${BUILD_NUMBER}

./bfx/build_linux.sh
"${WORKSPACE}/script/build_sdk_linux.sh" $build_id
