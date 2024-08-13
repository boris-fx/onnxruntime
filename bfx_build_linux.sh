source scl_source enable devtoolset-9
PATH=/home/buildpc/ml/deps/cmake-3.26.1-linux-x86_64/bin:$PATH
#do this before the script??
#conda init bash
#conda activate
./build.sh --config Release \
    --build_shared_lib \
    --use_cuda \
    --cuda_home /home/buildpc/ml/sdk_deps/cuda_sdk_dir \
    --cudnn_home /home/buildpc/ml/sdk_deps/cudnn_dir \
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

# now, convert to distributable!
dist_name=onnxruntime-linux-x64-1.18.1-5eeac2d-cu118
mv build/dist_release build/$dist_name

# zip it up!
cd build
zip -r $dist_name.zip $(basename $dist_name)
cd ..
