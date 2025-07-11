# run inside a MSVC shell w/ python interpreter present
# tested using MSVC 2022 because it comes w/ a new enough version of cmake
$ErrorActionPreference = 'Stop'; # quit on error..
Set-StrictMode -Version Latest;
$PSDefaultParameterValues['*:ErrorAction']='Stop';
function CheckForErrors { if (-not $?) { throw 'Failure!'; } }

$DIST_NAME = if ($args.Count -gt 0) { $args[0] } else { throw 'DIST_NAME argument is required' }
# should be either 'Release' or 'Debug'
$BUILD_CONFIG = if ($args.Count -gt 1) { $args[1] } else { 'Release' }

Write-Output "starting onnxruntime build: ${DIST_NAME}"

# clear any previous build..
if (Test-Path build) { Remove-Item -r -Force build }
mkdir build

$CUDA_SDK_NAME='cuda-sdk-win-v12.8.1'
$CUDNN_NAME='cudnn-windows-x86_64-9.10.2.21_cuda12-archive'

Push-Location build
    # fetch CUDA dependencies from BinaryArtifacts
    $BINARY_ARTIFACTS = 'mescola:Boris FX/Engineering/BinaryArtifacts'

    rclone copy ($BINARY_ARTIFACTS + '/' + $CUDA_SDK_NAME + '.zip') .
    tar -xzf ($CUDA_SDK_NAME + '.zip')
    $CUDA_HOME = "$(Get-Location)\${CUDA_SDK_NAME}" -replace '\\', '/'

    rclone copy ($BINARY_ARTIFACTS + '/' + $CUDNN_NAME + '.zip') .
    tar -xzf ($CUDNN_NAME + '.zip')
    $CUDNN_HOME = "$(Get-Location)\${CUDNN_NAME}" -replace '\\', '/'
Pop-Location

$CMAKE_CUDA_FLAGS = "-static-global-template-stub=false" # this needed for compilation when switching to CU 12.8 from 12.4
$CMAKE_CUDA_ARCHITECTURES = "60-real;61-real;70-real;75-real;80-real;86-real;89-real;90a-real;90-real;90-virtual;120-real;120-virtual"

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
    --config $BUILD_CONFIG `
    --cmake_generator "Ninja" `
    --build_shared_lib `
    --parallel `
    --use_dml `
    --use_cuda `
    --cuda_home $CUDA_HOME `
    --cudnn_home $CUDNN_HOME `
    --skip_tests `
    --cmake_extra_defines `
        CMAKE_CUDA_FLAGS=$CMAKE_CUDA_FLAGS `
        CMAKE_CUDA_ARCHITECTURES=$CMAKE_CUDA_ARCHITECTURES `
        onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_FLASH_ATTENTION=OFF

# can incremental build too after initial call to .\build.bat
# cmake --build .\build\Windows\$BUILD_CONFIG -j12 --config $BUILD_CONFIG

$DIST_DIR=".\build\${DIST_NAME}"
mkdir $DIST_DIR
$DIST_LIB_DIR="${DIST_DIR}\lib"
mkdir $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime.lib $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime_providers_cuda.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime_providers_cuda.lib $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime_providers_shared.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime_providers_shared.lib $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\DirectML.dll $DIST_LIB_DIR
Copy-Item .\build\Windows\$BUILD_CONFIG\DirectML.Debug.dll $DIST_LIB_DIR
if ($BUILD_CONFIG -eq 'Debug') {
    Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime.pdb $DIST_LIB_DIR
    Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime_providers_cuda.pdb $DIST_LIB_DIR
    Copy-Item .\build\Windows\$BUILD_CONFIG\onnxruntime_providers_shared.pdb $DIST_LIB_DIR
    Copy-Item .\build\Windows\$BUILD_CONFIG\DirectML.pdb $DIST_LIB_DIR
    Copy-Item .\build\Windows\$BUILD_CONFIG\DirectML.Debug.pdb $DIST_LIB_DIR
}
Copy-Item -r .\include $DIST_DIR

# generate manifest for libraries with DLL hashes
Copy-Item bfx/bfx_ml.ort_deps.runtime.manifest.in  $DIST_LIB_DIR
Push-Location $DIST_LIB_DIR
mt.exe -manifest bfx_ml.ort_deps.runtime.manifest.in -hashupdate -out:bfx_ml.ort_deps.runtime.manifest
Remove-Item bfx_ml.ort_deps.runtime.manifest.in
Pop-Location

# make zip archive!
$DIST_PATH = "$(Get-Location)\${DIST_DIR}"
Compress-Archive -Path $DIST_PATH -DestinationPath "${DIST_PATH}.zip" -Force

Write-Output "onnxruntime build completed, artifact generated: build\${DIST_NAME}.zip"
