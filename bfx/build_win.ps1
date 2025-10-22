# run inside a MSVC shell w/ python interpreter present
# tested using MSVC 2022 because it comes w/ a new enough version of cmake
$ErrorActionPreference = 'Stop'; # quit on error..
Set-StrictMode -Version Latest;
$PSDefaultParameterValues['*:ErrorAction']='Stop';
function CheckForErrors { if (-not $?) { throw 'Failure!'; } }

$DIST_NAME = if ($args.Count -gt 0) { $args[0] } else { throw 'DIST_NAME argument is required' }
# should be either 'Release' or 'Debug'
$BUILD_CONFIG = if ($args.Count -gt 1) { $args[1] } else { 'Release' }

# make sure cl.exe is NOT available
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    Write-Host "The command cl.exe is not available, this is expected. This script will load the appropriate VCVARS environments"
} else {
    throw "The command cl.exe is available. This is not expected! This script will load the appropriate VCVARS environments, there should not be one already activated"
}

# allow env to specify VCVARS variables, otherwise fall back to default for MSVC 2019 community
$VCVARS_X86_64 = if ($env:VCVARS_X86_64) { $env:VCVARS_X86_64 } else { 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat' }
$VCVARS_ARM64  = if ($env:VCVARS_ARM64) { $env:VCVARS_ARM64 } else { 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsamd64_arm64.bat' }
if (-not (Test-Path -Path $VCVARS_X86_64 -PathType Leaf)) { throw "specified X86_64 VCVARS path not found: ${VCVARS_X86_64}" }
if (-not (Test-Path -Path $VCVARS_ARM64 -PathType Leaf)) { throw "specified ARM64 VCVARS path not found: ${VCVARS_ARM64}" }
Write-Output "VCVARS for x86_64: ${VCVARS_X86_64}"
Write-Output "VCVARS for arm64: ${VCVARS_ARM64}"

# TODO: if these files are not found, error!
# TODO: CLI flag to build for only one arch
# TODO: make sure cl.exe is NOT available initially
# TODO: check for these vals in $env, and these are only fallbacks

Write-Output "starting onnxruntime build: ${DIST_NAME}"

# clear any previous build..
if (Test-Path build) { Remove-Item -r -Force build }
mkdir build

$CUDA_SDK_NAME='cuda-sdk-win-v12.8.1'
$CUDNN_NAME='cudnn-windows-x86_64-9.10.2.21_cuda12-archive'

'-- fetching dependencies --'
Push-Location build
    # fetch CUDA dependencies from BinaryArtifacts
    $BINARY_ARTIFACTS = 'mescola:Boris FX/Engineering/BinaryArtifacts'

    rclone copy ($BINARY_ARTIFACTS + '/' + $CUDA_SDK_NAME + '.zip') .
    tar -xzf ($CUDA_SDK_NAME + '.zip')
    $CUDA_HOME = "$(Get-Location)\${CUDA_SDK_NAME}" -replace '\\', '/'

    # https://github.com/microsoft/onnxruntime/issues/22728
    # using the latest MSVC SDK, this error pops up
    # build\cuda-sdk-win-v12.8.1\include\cuda\std\detail\libcxx\include\cmath(1032): error #221-D: floating-point value does not fit in required floating-point type : relating to INFINITY macro evaluating to ((float)(1e+300))
    # we could either force an earlier MSVC SDK, or tweak this line in-place to make the error go away.
python -c @"
f = '${CUDA_HOME}/include/cuda/std/detail/libcxx/include/cmath'
str_in = 'if (__r >= ::nextafter(static_cast<_RealT>(_MaxVal), INFINITY))'
str_out = 'if (__r >= ::nextafter(static_cast<_RealT>(_MaxVal), std::numeric_limits<float>::infinity()))'
with open(f, 'r') as f_in:
    f_str = f_in.read()
    f_str = f_str.replace(str_in, str_out)
with open(f, 'w') as f_out:
    f_out.write(f_str)
print(f'Modified: {f} inplace to fix build error! Hopefully we can remove this if we switch to a later CUDA SDK')
"@

    rclone copy ($BINARY_ARTIFACTS + '/' + $CUDNN_NAME + '.zip') .
    tar -xzf ($CUDNN_NAME + '.zip')
    $CUDNN_HOME = "$(Get-Location)\${CUDNN_NAME}" -replace '\\', '/'
Pop-Location

$CMAKE_CUDA_FLAGS = "-static-global-template-stub=false" # this needed for compilation when switching to CU 12.8 from 12.4
$CMAKE_CUDA_ARCHITECTURES = "60-real;61-real;70-real;75-real;80-real;86-real;89-real;90a-real;90-real;90-virtual;120-real;120-virtual"

conda activate base; CheckForErrors;

# cmake on path
'-- cmake info --';
cmake --version; CheckForErrors;

# conda environment available
'-- conda info --';
conda --version; CheckForErrors;

'-- python info --';
python --version; CheckForErrors;
where.exe python

$COMMON_BUILD_ARGS = "python tools\ci_build\build.py --config ${BUILD_CONFIG} --build_shared_lib --parallel --use_dml --skip_tests"
$COMMON_BUILD_DIR = "$(Get-Location)\build"

$DIST_DIR="$(Get-Location)\build\${DIST_NAME}"
$DIST_LIB_DIR="${DIST_DIR}\lib"
mkdir $DIST_DIR
mkdir $DIST_LIB_DIR
Copy-Item -r .\include $DIST_DIR

$X86_64_NAME = "x86_64"
$X86_64_BUILD_DIR = "${COMMON_BUILD_DIR}\${X86_64_NAME}"
$X86_64_DIST_LIB_DIR="${DIST_LIB_DIR}\${X86_64_NAME}"
$X86_64_CMAKE_EXTRA_DEFINES = "CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS} CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_USE_FLASH_ATTENTION=OFF"
$X86_64_ARGS = "--cmake_generator Ninja --use_cuda --cuda_home ${CUDA_HOME} --cudnn_home ${CUDNN_HOME} --cmake_extra_defines ${X86_64_CMAKE_EXTRA_DEFINES}"
$X86_64_BUILD_CMD = "${COMMON_BUILD_ARGS} --build_dir ${X86_64_BUILD_DIR}  ${X86_64_ARGS}"

$ARM64_NAME = "arm64"
$ARM64_BUILD_DIR = "${COMMON_BUILD_DIR}\${ARM64_NAME}"
$ARM64_DIST_LIB_DIR="${DIST_LIB_DIR}\${ARM64_NAME}"
# Building w/ Ninja fails on arm64, so build w/ MSBuild for MSVC 17
# probably would only require a few small tweaks to make Ninja run, but no need
# the x86_64 build fails when running MSVC generator, due to 'visual studio integration' not being present on the CUDA SDK artifact (cuda_sdk/extras/visual_studio_integration/MSBuildExtensions)
$ARM64_ARGS = '--cmake_generator "Visual Studio 17 2022" --arm64'
$ARM64_BUILD_CMD = "${COMMON_BUILD_ARGS} --build_dir ${ARM64_BUILD_DIR}  ${ARM64_ARGS}"

# arm64
    # build
    '-- running build (arm64) --';
    "-- command: `"${ARM64_BUILD_CMD}`""
    cmd /c "`"${VCVARS_ARM64}`" & ${ARM64_BUILD_CMD}"

    # package
    mkdir $ARM64_DIST_LIB_DIR
    $ARM64_BUILD_LIB_DIR="${ARM64_BUILD_DIR}\${BUILD_CONFIG}\${BUILD_CONFIG}"
    Copy-Item $ARM64_BUILD_LIB_DIR\onnxruntime.dll $ARM64_DIST_LIB_DIR
    Copy-Item $ARM64_BUILD_LIB_DIR\onnxruntime.lib $ARM64_DIST_LIB_DIR
    Copy-Item $ARM64_BUILD_LIB_DIR\DirectML.dll $ARM64_DIST_LIB_DIR
    Copy-Item $ARM64_BUILD_LIB_DIR\DirectML.Debug.dll $ARM64_DIST_LIB_DIR
    if ($BUILD_CONFIG -eq 'Debug') {
        Copy-Item $ARM64_BUILD_LIB_DIR\onnxruntime.pdb $ARM64_DIST_LIB_DIR
        Copy-Item $ARM64_BUILD_LIB_DIR\DirectML.pdb $ARM64_DIST_LIB_DIR
        Copy-Item $ARM64_BUILD_LIB_DIR\DirectML.Debug.pdb $ARM64_DIST_LIB_DIR
    }

    # generate manifest for libraries with DLL hashes
    Copy-Item bfx/bfx_ml.ort_deps.runtime.manifest.arm64.in $ARM64_DIST_LIB_DIR
    Push-Location $ARM64_DIST_LIB_DIR
    cmd /c "`"${VCVARS_ARM64}`" & mt.exe -manifest bfx_ml.ort_deps.runtime.manifest.arm64.in -hashupdate -out:bfx_ml.ort_deps.runtime.manifest"
    Remove-Item bfx_ml.ort_deps.runtime.manifest.arm64.in
    Pop-Location

# x86_64
    # build
    '-- running build (x86_64) --';
    "-- command: `"${X86_64_BUILD_CMD}`""
    cmd /c "`"${VCVARS_X86_64}`" & ${X86_64_BUILD_CMD}"

    # package
    mkdir $X86_64_DIST_LIB_DIR
    $X86_64_BUILD_LIB_DIR="${X86_64_BUILD_DIR}\${BUILD_CONFIG}"
    Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime.dll $X86_64_DIST_LIB_DIR
    Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime.lib $X86_64_DIST_LIB_DIR
    Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime_providers_cuda.dll $X86_64_DIST_LIB_DIR
    Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime_providers_cuda.lib $X86_64_DIST_LIB_DIR
    Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime_providers_shared.dll $X86_64_DIST_LIB_DIR
    Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime_providers_shared.lib $X86_64_DIST_LIB_DIR
    Copy-Item $X86_64_BUILD_LIB_DIR\DirectML.dll $X86_64_DIST_LIB_DIR
    Copy-Item $X86_64_BUILD_LIB_DIR\DirectML.Debug.dll $X86_64_DIST_LIB_DIR
    if ($BUILD_CONFIG -eq 'Debug') {
        Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime.pdb $X86_64_DIST_LIB_DIR
        Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime_providers_cuda.pdb $X86_64_DIST_LIB_DIR
        Copy-Item $X86_64_BUILD_LIB_DIR\onnxruntime_providers_shared.pdb $X86_64_DIST_LIB_DIR
        Copy-Item $X86_64_BUILD_LIB_DIR\DirectML.pdb $X86_64_DIST_LIB_DIR
        Copy-Item $X86_64_BUILD_LIB_DIR\DirectML.Debug.pdb $X86_64_DIST_LIB_DIR
    }

    # generate manifest for libraries with DLL hashes
    Copy-Item bfx/bfx_ml.ort_deps.runtime.manifest.x86_64.in $X86_64_DIST_LIB_DIR
    Push-Location $X86_64_DIST_LIB_DIR
    cmd /c "`"${VCVARS_X86_64}`" & mt.exe -manifest bfx_ml.ort_deps.runtime.manifest.x86_64.in -hashupdate -out:bfx_ml.ort_deps.runtime.manifest"
    Remove-Item bfx_ml.ort_deps.runtime.manifest.x86_64.in
    Pop-Location

# make final zip archive!
Compress-Archive -Path $DIST_DIR -DestinationPath "${DIST_DIR}.zip" -Force

Write-Output "onnxruntime build completed, artifacts generated:"
Write-Output "  ${DIST_DIR}.zip"
