// bfx: not part of upstream onnxruntime. See core/providers/dml/bfx_dml_external_allocator.h

#pragma once

#include "core/providers/dml/bfx_dml_external_allocator.h"
#include "DmlCommittedResourceWrapper.h"

namespace Dml
{
    inline Microsoft::WRL::ComPtr<ID3D12Resource> BfxAllocResource(const BfxDmlExternalAllocator& external, size_t bytes, bool persistent)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        resource.Attach(external.alloc(external.ctx, bytes, persistent));
        ORT_THROW_HR_IF(E_OUTOFMEMORY, !resource);
        return resource;
    }

    // for the BucketizedBufferAllocator: its buffers go back through free()
    inline Microsoft::WRL::ComPtr<DmlResourceWrapper> BfxAllocResourceWrapper(const BfxDmlExternalAllocator& external, size_t bytes)
    {
        Microsoft::WRL::ComPtr<DmlResourceWrapper> wrapper;
        Dml::SafeMakeOrThrow<DmlCommittedResourceWrapper>(BfxAllocResource(external, bytes, false)).As(&wrapper);
        return wrapper;
    }
}
