# run inside a MSVC shell w/ python interpreter present
# tested using MSVC 2022 because it comes w/ a new enough version of cmake

# Remove-Item -r -Force .\build
.\build.bat --config Release --cmake_generator "Visual Studio 17 2022" --build_shared_lib --parallel --use_dml --use_cuda --cuda_home 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8' --cudnn_home Z:/code/bfx/deps/cudnn-windows-x86_64-8.9.1.23_cuda11-archive --skip_tests

mkdir .\build\dist_release
mkdir .\build\dist_release\lib
Copy-Item .\build\Windows\Release\Release\onnxruntime.dll .\build\dist_release\lib\.
Copy-Item .\build\Windows\Release\Release\onnxruntime.lib .\build\dist_release\lib\.
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_cuda.dll .\build\dist_release\lib\.
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_cuda.lib .\build\dist_release\lib\.
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_shared.dll .\build\dist_release\lib\.
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_shared.lib .\build\dist_release\lib\.
Copy-Item .\build\Windows\Release\Release\DirectML.dll .\build\dist_release\lib\.
Copy-Item -r .\include .\build\dist_release\.

# cp .\build\dist_release\lib\*.dll <consuming code build dir>
