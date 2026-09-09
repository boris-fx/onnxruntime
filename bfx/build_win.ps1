# run inside a MSVC shell w/ python interpreter present
# tested using MSVC 2022 because it comes w/ a new enough version of cmake
#
# usage:
#   bfx/build_win.ps1 <DistName> [Release|RelWithDebInfo|Debug] [-Arch <list>] [-Backend <list>] [-Parallel <n>] [-NoClean]
#
# examples:
#   # everything (what CI does): both arches, all backends, Release
#   bfx/build_win.ps1 libonnxruntime-1.29.0_win_cu128-dml-1.15.4_<build-id>
#
#   # quick local iteration: x86_64 / dml / Debug, all cores, reusing the previous build tree
#   bfx/build_win.ps1 scratch Debug -Arch x86_64 -Backend dml -Parallel 0 -NoClean
#
#   # chasing a crash that only reproduces optimized: same codegen as Release, but with PDBs
#   bfx/build_win.ps1 scratch RelWithDebInfo -Arch x86_64 -Backend dml -Parallel 0 -NoClean
#
# notes:
#   - CUDA is x86_64 only (no CUDA SDK for windows-on-arm); it is ignored for the arm64 build,
#     and the CUDA/cuDNN SDKs are only downloaded when an x86_64 CUDA build is actually selected.
#   - -Parallel defaults to 1 so the default (CI) invocation builds exactly like it always has.
#     '-Parallel 0' means one job per core - safe and much faster for any build without CUDA in it.
#   - the arm64 webgpu build needs host tablegen tools out of an x86_64 build tree (see the arm64
#     section below), so '-Arch arm64 -Backend webgpu' needs either x86_64 in -Arch as well, or a
#     previous x86_64 build left in place with -NoClean.
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$DistName,

    # RelWithDebInfo is Release codegen (/O2, NDEBUG) plus symbols - use it to debug a crash that
    # only reproduces in an optimized build. PDBs are packaged for any config that emits them.
    [Parameter(Position = 1)]
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')]
    [string]$BuildConfig = 'Release',

    [ValidateSet('x86_64', 'arm64')]
    [string[]]$Arch = @('x86_64', 'arm64'),

    [ValidateSet('cuda', 'dml', 'webgpu')]
    [string[]]$Backend = @('cuda', 'dml', 'webgpu'),

    # build.py --parallel, where 0 means 'one job per core'. defaults to 1, which is what CI has
    # always used - nvcc on the flash attention kernels OOMs the machine otherwise. a build with no
    # CUDA in it has no such problem, so pass -Parallel 0 for those; it is far faster.
    [ValidateRange(0, 1024)]
    [int]$Parallel = 1,

    # keep the existing build/ tree: incremental rebuild, and no re-download of the CUDA SDKs
    [switch]$NoClean
)

$ErrorActionPreference = 'Stop'; # quit on error..
Set-StrictMode -Version Latest;
$PSDefaultParameterValues['*:ErrorAction']='Stop';
function CheckForErrors { if (-not $?) { throw 'Failure!'; } }

# throw on a failed cmd.exe/native invocation - $ErrorActionPreference does not cover those
function CheckExitCode { param([string]$What) if ($LASTEXITCODE -ne 0) { throw "${What} failed with exit code ${LASTEXITCODE}" } }

$DIST_NAME = $DistName
$BUILD_CONFIG = $BuildConfig

$BUILD_X86_64 = $Arch -contains 'x86_64'
$BUILD_ARM64  = $Arch -contains 'arm64'
$USE_CUDA     = $Backend -contains 'cuda'
$USE_DML      = $Backend -contains 'dml'
$USE_WEBGPU   = $Backend -contains 'webgpu'

# CUDA only ever applies to the x86_64 build
$X86_64_USE_CUDA = $USE_CUDA -and $BUILD_X86_64
if ($USE_CUDA -and -not $BUILD_X86_64) {
    Write-Warning 'cuda was requested but x86_64 is not in -Arch; CUDA is x86_64-only, so it will not be built'
}

# make sure cl.exe is NOT available
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    Write-Host "The command cl.exe is not available, this is expected. This script will load the appropriate VCVARS environments"
} else {
    throw "The command cl.exe is available. This is not expected! This script will load the appropriate VCVARS environments, there should not be one already activated"
}

# allow env to specify VCVARS variables, otherwise fall back to default for MSVC 2022 community
$VCVARS_X86_64 = if ($env:VCVARS_X86_64) { $env:VCVARS_X86_64 } else { 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat' }
$VCVARS_ARM64  = if ($env:VCVARS_ARM64) { $env:VCVARS_ARM64 } else { 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsamd64_arm64.bat' }
# only require the toolchains the selected architectures actually need
if ($BUILD_X86_64) {
    if (-not (Test-Path -Path $VCVARS_X86_64 -PathType Leaf)) { throw "specified X86_64 VCVARS path not found: ${VCVARS_X86_64}" }
    Write-Output "VCVARS for x86_64: ${VCVARS_X86_64}"
}
if ($BUILD_ARM64) {
    if (-not (Test-Path -Path $VCVARS_ARM64 -PathType Leaf)) { throw "specified ARM64 VCVARS path not found: ${VCVARS_ARM64}" }
    Write-Output "VCVARS for arm64: ${VCVARS_ARM64}"
}

Write-Output "starting onnxruntime build: ${DIST_NAME}"
Write-Output "  config:   ${BUILD_CONFIG}"
Write-Output "  arch:     $($Arch -join ', ')"
Write-Output "  backends: $($Backend -join ', ')"
Write-Output "  clean:    $(-not $NoClean)"

conda activate base; CheckForErrors;

# conda environment available
'-- conda info --';
conda --version; CheckForErrors;

'-- python info --';
python --version; CheckForErrors;
where.exe python

# clear any previous build.. (-NoClean keeps it, for incremental rebuilds)
if (-not $NoClean -and (Test-Path build)) { Remove-Item -r -Force build }
if (-not (Test-Path build)) { mkdir build | Out-Null }

$CUDA_SDK_NAME='cuda-sdk-win-v12.8.1'
$CUDNN_NAME='cudnn-windows-x86_64-9.10.2.21_cuda12-archive'
$CUDA_HOME = ''
$CUDNN_HOME = ''

if ($X86_64_USE_CUDA) {
'-- fetching dependencies --'
Push-Location build
    # fetch CUDA dependencies from BinaryArtifacts
    $BINARY_ARTIFACTS = 'mescola:Boris FX/Engineering/BinaryArtifacts'

    $CUDA_HOME = "$(Get-Location)\${CUDA_SDK_NAME}" -replace '\\', '/'
    $CUDNN_HOME = "$(Get-Location)\${CUDNN_NAME}" -replace '\\', '/'

    if (Test-Path $CUDA_SDK_NAME) {
        "-- ${CUDA_SDK_NAME} already unpacked, skipping fetch --"
    } else {
        rclone copy ($BINARY_ARTIFACTS + '/' + $CUDA_SDK_NAME + '.zip') .
        tar -xzf ($CUDA_SDK_NAME + '.zip')

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
    }

    if (Test-Path $CUDNN_NAME) {
        "-- ${CUDNN_NAME} already unpacked, skipping fetch --"
    } else {
        rclone copy ($BINARY_ARTIFACTS + '/' + $CUDNN_NAME + '.zip') .
        tar -xzf ($CUDNN_NAME + '.zip')
    }
Pop-Location
} else {
    '-- skipping CUDA dependency fetch (no x86_64 cuda build selected) --'
}

$CMAKE_CUDA_FLAGS = "-static-global-template-stub=false" # this needed for compilation when switching to CU 12.8 from 12.4
$CMAKE_CUDA_ARCHITECTURES = "60-real;61-real;70-real;75-real;80-real;86-real;89-real;90a-real;90-real;90-virtual;120-real;120-virtual"

# args shared by both architectures; --parallel and --build_dir are appended per-arch
$COMMON_BUILD_ARGS_LIST = @('python', 'tools\ci_build\build.py', '--config', $BUILD_CONFIG, '--build_shared_lib', '--skip_tests')
if ($USE_DML)    { $COMMON_BUILD_ARGS_LIST += '--use_dml' }
# static_lib links the WebGPU EP into onnxruntime.dll (so there is no onnxruntime_providers_webgpu.dll
# to ship), and BUILD_DAWN_SHARED_LIBRARY makes Dawn its own webgpu_dawn.dll. that pairing is what lets a
# client create its own WGPUDevice/WGPUBuffer and hand them to ORT: both sides call the same Dawn.
# the plugin form ('--use_webgpu shared_lib') hides every Dawn symbol behind CreateEpFactories, and cmake
# rejects combining it with a Dawn DLL outright - see cmake/CMakeLists.txt:1036-1042.
if ($USE_WEBGPU) { $COMMON_BUILD_ARGS_LIST += @('--use_webgpu', 'static_lib') }
$COMMON_BUILD_ARGS = $COMMON_BUILD_ARGS_LIST -join ' '
$COMMON_BUILD_DIR = "$(Get-Location)\build"

$DIST_DIR="$(Get-Location)\build\${DIST_NAME}"
$DIST_LIB_DIR="${DIST_DIR}\lib"
if (Test-Path $DIST_DIR) { Remove-Item -r -Force $DIST_DIR }
mkdir $DIST_DIR | Out-Null
mkdir $DIST_LIB_DIR | Out-Null
Copy-Item -r .\include $DIST_DIR

# PDBs only exist for configs that emit them (Debug, RelWithDebInfo), and which of DirectML's ship
# varies too, so package whatever is actually there rather than keying off the config name.
function Copy-Pdb {
    param([string]$Path, [string]$Dest)
    if (Test-Path $Path) { Copy-Item $Path $Dest }
}

# ORT installs no WebGPU headers at all - the only thing under include/onnxruntime/core/providers/webgpu
# is a dummy "was WebGPU in this build" signal header. A client that creates its own WGPUDevice and
# WGPUBuffers needs Dawn's headers, which live in Dawn's build tree in two include roots: the checked-in
# one (webgpu/) and the generated one (dawn/, which is where dawn_proc_table.h is produced).
# $CmakeBinDir is the cmake binary dir for the arch, i.e. the directory containing _deps.
function Copy-DawnHeaders {
    param([string]$CmakeBinDir, [string]$DestInclude)
    Copy-Item -Recurse -Force "${CmakeBinDir}\_deps\dawn-src\include\*" $DestInclude
    Copy-Item -Recurse -Force "${CmakeBinDir}\_deps\dawn-build\gen\include\*" $DestInclude
    # the WebGPU provider option keys (webgpuDevice, webgpuInstance, dawnProcTable, deviceId,
    # preserveDevice, ...) are in internal ORT source rather than its public include tree, so vendor
    # that one header in alongside them.
    $OPTS_DST = Join-Path $DestInclude 'onnxruntime\core\providers\webgpu'
    if (-not (Test-Path $OPTS_DST)) { mkdir $OPTS_DST | Out-Null }
    Copy-Item 'onnxruntime\core\providers\webgpu\webgpu_provider_options.h' $OPTS_DST
}

# the checked-in manifest templates list every DLL we can ever ship for an arch. a partial build
# won't produce all of them, so drop the missing entries before handing it to mt.exe -hashupdate.
function Write-DepsManifest {
    param([string]$Template, [string]$LibDir, [string]$Vcvars)
    $STAGED_NAME = Split-Path -Leaf $Template
    $KEPT = Get-Content $Template | Where-Object {
        if ($_ -match '<file\s+name="([^"]+)"') { Test-Path (Join-Path $LibDir $Matches[1]) } else { $true }
    }
    Set-Content -Path (Join-Path $LibDir $STAGED_NAME) -Value $KEPT
    Push-Location $LibDir
    cmd /c "`"${Vcvars}`" & mt.exe -manifest ${STAGED_NAME} -hashupdate -out:bfx_ml.ort_deps.runtime.manifest"
    CheckExitCode 'mt.exe'
    Remove-Item $STAGED_NAME
    Pop-Location
}

$X86_64_NAME = "x86_64"
$X86_64_BUILD_DIR = "${COMMON_BUILD_DIR}\${X86_64_NAME}"
$X86_64_DIST_LIB_DIR="${DIST_LIB_DIR}\${X86_64_NAME}"

$ARM64_NAME = "arm64"
$ARM64_BUILD_DIR = "${COMMON_BUILD_DIR}\${ARM64_NAME}"
$ARM64_DIST_LIB_DIR="${DIST_LIB_DIR}\${ARM64_NAME}"

# x86_64
if ($BUILD_X86_64) {
    $X86_64_CMAKE_EXTRA_DEFINES_LIST = @('onnxruntime_BUILD_UNIT_TESTS=OFF')
    if ($USE_WEBGPU) { $X86_64_CMAKE_EXTRA_DEFINES_LIST += 'onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON' }
    $X86_64_ARGS_LIST = @('--cmake_generator', 'Ninja')
    if ($X86_64_USE_CUDA) {
        $X86_64_CMAKE_EXTRA_DEFINES_LIST += @("CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}", "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}", 'onnxruntime_USE_FLASH_ATTENTION:BOOL=ON')
        $X86_64_ARGS_LIST += @('--use_cuda', '--cuda_home', $CUDA_HOME, '--cudnn_home', $CUDNN_HOME, '--nvcc_threads', '1', '--flash_nvcc_threads', '1')
    }
    $X86_64_ARGS_LIST += @('--cmake_extra_defines') + $X86_64_CMAKE_EXTRA_DEFINES_LIST
    $X86_64_ARGS = $X86_64_ARGS_LIST -join ' '
    $X86_64_BUILD_CMD = "${COMMON_BUILD_ARGS} --parallel ${Parallel} --build_dir ${X86_64_BUILD_DIR}  ${X86_64_ARGS}"

    # build
    '-- running build (x86_64) --';
    "-- command: `"${X86_64_BUILD_CMD}`""
    cmd /c "`"${VCVARS_X86_64}`" & ${X86_64_BUILD_CMD}"
    CheckExitCode 'x86_64 build'

    # package
    mkdir $X86_64_DIST_LIB_DIR | Out-Null
    $X86_64_BUILD_LIB_DIR="${X86_64_BUILD_DIR}\${BUILD_CONFIG}"

    $X86_64_LIBS = @('onnxruntime', 'onnxruntime_providers_shared')
    if ($X86_64_USE_CUDA) { $X86_64_LIBS += 'onnxruntime_providers_cuda' }
    # NOTE: no onnxruntime_providers_webgpu here - with --use_webgpu static_lib the EP is linked into
    # onnxruntime.dll and only exists as a .lib. webgpu_dawn.dll is what ships instead; TODO below.
    foreach ($LIB in $X86_64_LIBS) {
        Copy-Item "${X86_64_BUILD_LIB_DIR}\${LIB}.dll" $X86_64_DIST_LIB_DIR
        Copy-Item "${X86_64_BUILD_LIB_DIR}\${LIB}.lib" $X86_64_DIST_LIB_DIR
        Copy-Pdb "${X86_64_BUILD_LIB_DIR}\${LIB}.pdb" $X86_64_DIST_LIB_DIR
    }
    if ($USE_WEBGPU) {
        # Dawn's D3D12 backend compiles shaders through DXC at runtime, so dxil.dll and dxcompiler.dll
        # must ship alongside onnxruntime_providers_webgpu.dll. cmake stages them into the build output
        # dir (onnxruntime_providers_webgpu_dll_deps in cmake/onnxruntime_providers_webgpu.cmake); ORT's
        # own packaging copies them the same way.
        Copy-Item "${X86_64_BUILD_LIB_DIR}\dxil.dll" $X86_64_DIST_LIB_DIR
        Copy-Item "${X86_64_BUILD_LIB_DIR}\dxcompiler.dll" $X86_64_DIST_LIB_DIR

        # cmake stages webgpu_dawn.dll next to onnxruntime.dll, but its import lib stays in Dawn's own
        # build tree. the client links this directly - it is how they call wgpuDeviceCreateBuffer etc.
        Copy-Item "${X86_64_BUILD_LIB_DIR}\webgpu_dawn.dll" $X86_64_DIST_LIB_DIR
        Copy-Item "${X86_64_BUILD_LIB_DIR}\_deps\dawn-build\src\dawn\native\webgpu_dawn.lib" $X86_64_DIST_LIB_DIR
        Copy-Pdb  "${X86_64_BUILD_LIB_DIR}\webgpu_dawn.pdb" $X86_64_DIST_LIB_DIR

        # ninja puts the cmake binary dir (the one holding _deps) at the same place as the build outputs
        Copy-DawnHeaders -CmakeBinDir $X86_64_BUILD_LIB_DIR -DestInclude "${DIST_DIR}\include"
    }
    if ($USE_DML) {
        Copy-Item $X86_64_BUILD_LIB_DIR\DirectML.dll $X86_64_DIST_LIB_DIR
        Copy-Item $X86_64_BUILD_LIB_DIR\DirectML.Debug.dll $X86_64_DIST_LIB_DIR
        Copy-Pdb $X86_64_BUILD_LIB_DIR\DirectML.pdb $X86_64_DIST_LIB_DIR
        Copy-Pdb $X86_64_BUILD_LIB_DIR\DirectML.Debug.pdb $X86_64_DIST_LIB_DIR
    }

    # generate manifest for libraries with DLL hashes
    Write-DepsManifest -Template "bfx/bfx_ml.ort_deps.runtime.manifest.${X86_64_NAME}.in" -LibDir $X86_64_DIST_LIB_DIR -Vcvars $VCVARS_X86_64
}

# arm64
if ($BUILD_ARM64) {
    # Building w/ Ninja fails on arm64, so build w/ MSBuild for MSVC 17
    # probably would only require a few small tweaks to make Ninja run, but no need
    # the x86_64 build fails when running MSVC generator, due to 'visual studio integration' not being present on the CUDA SDK artifact (cuda_sdk/extras/visual_studio_integration/MSBuildExtensions)
    $ARM64_ARGS_LIST = @('--cmake_generator', '"Visual Studio 17 2022"', '--arm64')
    $ARM64_CMAKE_EXTRA_DEFINES_LIST = @('onnxruntime_BUILD_UNIT_TESTS=OFF')
    if ($USE_WEBGPU) { $ARM64_CMAKE_EXTRA_DEFINES_LIST += 'onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON' }

    if ($USE_WEBGPU) {
        # Dawn/DXC (needed for --use_webgpu) builds its own copy of LLVM's tablegen tool. When cross-compiling
        # for arm64 with the Visual Studio generator, CMake cannot generate a project to build that tool (it
        # must run on the host, not the target), which causes MSBuild error MSB1009 for a missing
        # LLVM-tablegen-host.vcxproj. Reuse the tablegen executables already built during the native
        # x86_64 build to work around this (same approach ORT's own CI uses, see
        # tools/ci_build/github/azure-pipelines/stages/nodejs-win-packaging-stage.yml).
        '-- locating host tablegen tools from x86_64 build (needed for arm64 dawn/dxc cross-compile) --';
        if (-not (Test-Path $X86_64_BUILD_DIR)) {
            throw "an arm64 webgpu build needs host tablegen tools from an x86_64 build tree, and there is none at ${X86_64_BUILD_DIR}. either add x86_64 to -Arch, or re-run with -NoClean on top of a previous x86_64 build."
        }
        $LLVM_TABLEGEN = (Get-ChildItem -Path $X86_64_BUILD_DIR -Recurse -Filter 'llvm-tblgen.exe' | Select-Object -First 1).FullName
        $CLANG_TABLEGEN = (Get-ChildItem -Path $X86_64_BUILD_DIR -Recurse -Filter 'clang-tblgen.exe' | Select-Object -First 1).FullName
        if (-not $LLVM_TABLEGEN) { throw "could not locate llvm-tblgen.exe under ${X86_64_BUILD_DIR}" }
        if (-not $CLANG_TABLEGEN) { throw "could not locate clang-tblgen.exe under ${X86_64_BUILD_DIR}" }
        "-- LLVM_TABLEGEN: ${LLVM_TABLEGEN}"
        "-- CLANG_TABLEGEN: ${CLANG_TABLEGEN}"
        $ARM64_CMAKE_EXTRA_DEFINES_LIST += @("LLVM_TABLEGEN=${LLVM_TABLEGEN}", "CLANG_TABLEGEN=${CLANG_TABLEGEN}")
    }

    $ARM64_ARGS_LIST += @('--cmake_extra_defines') + $ARM64_CMAKE_EXTRA_DEFINES_LIST
    $ARM64_ARGS = $ARM64_ARGS_LIST -join ' '
    $ARM64_BUILD_CMD = "${COMMON_BUILD_ARGS} --parallel ${Parallel} --build_dir ${ARM64_BUILD_DIR}  ${ARM64_ARGS}"

    # build
    '-- running build (arm64) --';
    "-- command: `"${ARM64_BUILD_CMD}`""
    cmd /c "`"${VCVARS_ARM64}`" & ${ARM64_BUILD_CMD}"
    CheckExitCode 'arm64 build'

    # package
    mkdir $ARM64_DIST_LIB_DIR | Out-Null
    $ARM64_BUILD_LIB_DIR="${ARM64_BUILD_DIR}\${BUILD_CONFIG}\${BUILD_CONFIG}"

    $ARM64_LIBS = @('onnxruntime', 'onnxruntime_providers_shared')
    # see the x86_64 note: static-EP build has no onnxruntime_providers_webgpu.dll
    foreach ($LIB in $ARM64_LIBS) {
        Copy-Item "${ARM64_BUILD_LIB_DIR}\${LIB}.dll" $ARM64_DIST_LIB_DIR
        Copy-Item "${ARM64_BUILD_LIB_DIR}\${LIB}.lib" $ARM64_DIST_LIB_DIR
        Copy-Pdb "${ARM64_BUILD_LIB_DIR}\${LIB}.pdb" $ARM64_DIST_LIB_DIR
    }
    if ($USE_WEBGPU) {
        # see the x86_64 block above - Dawn's D3D12 backend needs DXC at runtime
        Copy-Item "${ARM64_BUILD_LIB_DIR}\dxil.dll" $ARM64_DIST_LIB_DIR
        Copy-Item "${ARM64_BUILD_LIB_DIR}\dxcompiler.dll" $ARM64_DIST_LIB_DIR

        Copy-Item "${ARM64_BUILD_LIB_DIR}\webgpu_dawn.dll" $ARM64_DIST_LIB_DIR
        Copy-Pdb  "${ARM64_BUILD_LIB_DIR}\webgpu_dawn.pdb" $ARM64_DIST_LIB_DIR

        # arm64 builds with the Visual Studio generator, not Ninja, so its layout differs from x86_64:
        # the cmake binary dir (holding _deps) is one level above the build outputs, and MSBuild adds a
        # per-config subdirectory of its own under the dawn targets.
        # TODO: confirm both of these against a real arm64 build - x86_64 is the only one verified so far.
        $ARM64_CMAKE_BIN_DIR = "${ARM64_BUILD_DIR}\${BUILD_CONFIG}"
        Copy-Item "${ARM64_CMAKE_BIN_DIR}\_deps\dawn-build\src\dawn\native\${BUILD_CONFIG}\webgpu_dawn.lib" $ARM64_DIST_LIB_DIR
        Copy-DawnHeaders -CmakeBinDir $ARM64_CMAKE_BIN_DIR -DestInclude "${DIST_DIR}\include"
    }
    if ($USE_DML) {
        Copy-Item $ARM64_BUILD_LIB_DIR\DirectML.dll $ARM64_DIST_LIB_DIR
        Copy-Item $ARM64_BUILD_LIB_DIR\DirectML.Debug.dll $ARM64_DIST_LIB_DIR
        Copy-Pdb $ARM64_BUILD_LIB_DIR\DirectML.pdb $ARM64_DIST_LIB_DIR
        Copy-Pdb $ARM64_BUILD_LIB_DIR\DirectML.Debug.pdb $ARM64_DIST_LIB_DIR
    }

    # generate manifest for libraries with DLL hashes
    Write-DepsManifest -Template "bfx/bfx_ml.ort_deps.runtime.manifest.${ARM64_NAME}.in" -LibDir $ARM64_DIST_LIB_DIR -Vcvars $VCVARS_ARM64
}

# make final zip archive!
Compress-Archive -Path $DIST_DIR -DestinationPath "${DIST_DIR}.zip" -Force

Write-Output "onnxruntime build completed, artifacts generated:"
Write-Output "  ${DIST_DIR}.zip"
