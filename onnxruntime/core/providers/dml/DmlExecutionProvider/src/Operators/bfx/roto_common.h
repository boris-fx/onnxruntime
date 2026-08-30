#pragma once

// Shared C++ helpers for the roto ops. Kept out of d3d_util.h so the roto port adds files rather
// than editing ones every other bfx op already compiles against.

#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace bfx_ops {
namespace roto {

// A PROCESS-LIFETIME CACHE OF COMPILED SHADERS, and the single most important thing in this port
// for performance.
//
// WHY IT HAS TO EXIST. Our kernels are registered with MLOperatorKernelOptions::None, and they
// have to be: AbiCustomRegistry::RegisterOperatorKernel rejects AllowDynamicInputShapes combined
// with a shape inferrer (E_INVALIDARG), and these ops need a shape inferrer to size their outputs.
// None sets requiresInputShapesAtCreation, and AbiOpKernel::Compute then does this
// (MLOperatorAuthorImpl.cpp, "In the edge case that the input size is changing across
// invocations"): if the input shapes differ from the ones the kernel instance was built with, it
// constructs an ENTIRE NEW KERNEL for that one call and throws it away afterwards.
//
// For the refine graphs that is not an edge case, it is every call - they are dynamic in F, P, S,
// Q, N and K, and refine drives them in a loop over a changing batch. So a constructor that
// creates a root signature and a compute PSO would do so on every single iteration, and PSO
// creation is milliseconds.
//
// Root signatures and PSOs depend ONLY on the shader, never on the shapes, so they can outlive the
// kernel instance. Keyed on the bytecode pointer, which is a unique per-shader address in the
// binary, and on the device, since D3D12 objects belong to one. Nothing here is per-shape: any
// scratch a kernel needs must come from IMLOperatorKernelContext::AllocateTemporaryData at Compute
// time instead, never from the constructor.
inline std::shared_ptr<ComputeShader> cachedShader(
    ComPtr<ID3D12Device> device,
    const ComputeShaderConfig& cfg)
{
    using Key = std::pair<ID3D12Device*, const void*>;
    static std::mutex s_mutex;
    static std::map<Key, std::shared_ptr<ComputeShader>> s_cache;

    const Key key{device.Get(), cfg.bytecode};

    std::lock_guard<std::mutex> lock(s_mutex);
    auto it = s_cache.find(key);
    if (it != s_cache.end())
    {
        return it->second;
    }
    auto shader = std::make_shared<ComputeShader>(device, cfg);
    s_cache.emplace(key, shader);
    return shader;
}

// ceil(a / b) for positive values, as a dispatch grid dimension.
inline uint32_t gridFor(int64_t count, int64_t blockSize)
{
    if (count <= 0) { return 0; }
    return static_cast<uint32_t>((count + blockSize - 1) / blockSize);
}

// D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION. CUDA allows 2^31-1 on x and 65535 on y/z;
// D3D12 caps ALL THREE at 65535, so a dimension that was safe in the CUDA kernel is not
// automatically safe here. Checked rather than clamped: silently dispatching a smaller grid would
// leave part of the output holding whatever the allocator handed back.
constexpr uint32_t kMaxThreadGroupsPerDimension = 65535;

inline void checkDispatchGrid(uint32_t x, uint32_t y, const char* what)
{
    ML_CHECK_VALID_ARGUMENT(
        x <= kMaxThreadGroupsPerDimension && y <= kMaxThreadGroupsPerDimension,
        "roto: dispatch grid exceeds the D3D12 limit of 65535 thread groups per dimension");
    (void)what;
}

} // namespace roto
} // namespace bfx_ops
