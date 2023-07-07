# run inside a MSVC shell w/ python interpreter present
# tested using MSVC 2022 because it comes w/ a new enough version of cmake

Remove-Item -r -Force .\build
.\build.bat --config Debug --cmake_generator "Visual Studio 17 2022" --build_shared_lib --parallel --use_dml --use_cuda --cuda_home 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8' --cudnn_home Z:/code/bfx/deps/cudnn-windows-x86_64-8.9.1.23_cuda11-archive --skip_tests --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

# can incremental build too after initial call to .\build.bat
# cmake --build .\build\Windows\Debug -j12

# rm -r -Force .\build\dist_debug
mkdir .\build\dist_debug
mkdir .\build\dist_debug\lib
Copy-Item .\build\Windows\Debug\Debug\onnxruntime.dll .\build\dist_debug\lib\.
Copy-Item .\build\Windows\Debug\Debug\onnxruntime.lib .\build\dist_debug\lib\.
Copy-Item .\build\Windows\Debug\Debug\onnxruntime_providers_cuda.dll .\build\dist_debug\lib\.
Copy-Item .\build\Windows\Debug\Debug\onnxruntime_providers_cuda.lib .\build\dist_debug\lib\.
Copy-Item .\build\Windows\Debug\Debug\onnxruntime_providers_shared.dll .\build\dist_debug\lib\.
Copy-Item .\build\Windows\Debug\Debug\onnxruntime_providers_shared.lib .\build\dist_debug\lib\.
Copy-Item .\build\Windows\Debug\Debug\DirectML.dll .\build\dist_debug\lib\.
Copy-Item -r .\include .\build\dist_debug\.

# to copy libraries to consuming binary dir!
# cp .\build\dist_debug\lib\*.dll $env:BFX_ROOT/ai/build/.
