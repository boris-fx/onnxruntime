# run inside a MSVC shell w/ python interpreter present
# tested using MSVC 2022 because it comes w/ a new enough version of cmake
$ErrorActionPreference = 'Stop'; # quit on error..
Set-StrictMode -Version Latest;
$PSDefaultParameterValues['*:ErrorAction']='Stop';
function CheckForErrors { if (-not $?) { throw 'Failure!'; } }

$DIST_NAME = $args[0]

Write-Output "starting onnxruntime build: ${DIST_NAME}"

# clear any previous build..
if (Test-Path build) { Remove-Item -r -Force build }
mkdir build

Push-Location build
    # fetch CUDA dependencies from BinaryArtifacts
    $BINARY_ARTIFACTS = 'mescola:Boris FX/Engineering/BinaryArtifacts'

    $CUDA_SDK_NAME='cuda-sdk-win-v11.8'
    rclone copy ($BINARY_ARTIFACTS + '/' + $CUDA_SDK_NAME + '.zip') .
    tar -xzf ($CUDA_SDK_NAME + '.zip')
    $CUDA_HOME = "$(Get-Location)\${CUDA_SDK_NAME}"

    $CUDNN_NAME='cudnn-windows-x86_64-8.9.1.23_cuda11-archive'
    rclone copy ($BINARY_ARTIFACTS + '/' + $CUDNN_NAME + '.zip') .
    tar -xzf ($CUDNN_NAME + '.zip')
    $CUDNN_HOME = "$(Get-Location)\${CUDNN_NAME}"
Pop-Location

conda activate base; CheckForErrors;

'-- MSVC compiler info --';
cl; CheckForErrors;

# cmake on path
'-- cmake info --';
cmake --version; CheckForErrors;

# conda environment available
'-- conda info --';
conda --version; CheckForErrors;

'-- python info --';
python --version; CheckForErrors;
where.exe python

'-- running build --';
# now run onnxruntime build script
.\build.bat `
    --config Release `
    --cmake_generator "Visual Studio 16 2019" `
    --build_shared_lib `
    --parallel `
    --use_dml `
    --use_cuda `
    --cuda_home $CUDA_HOME `
    --cudnn_home $CUDNN_HOME `
    --skip_tests `
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

# can incremental build too after initial call to .\build.bat
# cmake --build .\build\Windows\Release -j12 --config Release

# rm -r -Force .\build\dist_release
$DIST_DIR=".\build\${DIST_NAME}"
mkdir $DIST_DIR
$DIST_LIB_DIR="${DIST_DIR}\lib"
mkdir $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\onnxruntime.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\onnxruntime.lib $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_cuda.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_cuda.lib $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_shared.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\onnxruntime_providers_shared.lib $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\DirectML.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\Release\Release\DirectML.Debug.dll $DIST_LIB_DIR
Copy-Item -r .\include $DIST_DIR

# generate manifest for libraries with DLL hashes
Copy-Item bfx/bfx_ml.ort_dml_deps.runtime.manifest.in  $DIST_LIB_DIR
Push-Location $DIST_LIB_DIR
mt.exe -manifest bfx_ml.ort_dml_deps.runtime.manifest.in -hashupdate -out:bfx_ml.ort_dml_deps.runtime.manifest
Remove-Item bfx_ml.ort_dml_deps.runtime.manifest.in
Pop-Location

# make zip archive!
$DIST_PATH = "$(Get-Location)\${DIST_DIR}"
Compress-Archive -Path $DIST_PATH -DestinationPath "${DIST_PATH}.zip" -Force

Write-Output "onnxruntime build completed, artifact generated: build\${DIST_NAME}.zip"
