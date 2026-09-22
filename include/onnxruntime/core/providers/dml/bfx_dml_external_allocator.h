// bfx: not part of upstream onnxruntime.
//
// Lets the client allocate the D3D12 buffers a DirectML execution provider uses - for tensors, weights and operators'
// persistent resources - in place of the provider's own pool of them (BucketizedBufferAllocator). A client can then
// share one pool of buffers between all of its sessions, and knows exactly what each of them holds.
//
// Set it per session w/ the session config entry kBfxDmlExternalAllocator, *before* the DirectML execution provider is
// appended to the session options: the provider reads its config when it is appended. The value is the address of a
// BfxDmlExternalAllocator as a decimal number, and the struct, and everything it points to, must outlive the session.

#pragma once

#include <cstddef>

struct ID3D12Resource;

struct BfxDmlExternalAllocator
{
    void* ctx;

    // A committed buffer of at least `bytes` bytes in a default heap, created w/ D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    // in D3D12_RESOURCE_STATE_COMMON - or one given back through free() before, reused. The provider takes over the
    // reference it is returned with. nullptr if out of memory.
    //
    // persistent: the provider keeps the buffer until it releases it, and it never comes back through free() - the
    // weights of fused graphs, which a graph fused at run time (when its input shapes change) also allocates mid-run.
    // Such a buffer is the provider's alone: it must be a new one, which the allocator does not hold on to.
    ID3D12Resource* (*alloc)(void* ctx, size_t bytes, bool persistent);

    // The provider is done with a buffer it got from alloc() (not a persistent one), except that GPU work it has already
    // recorded may still use it: the buffer can be handed out again for work queued behind that, but must not be
    // destroyed before that work has completed. The provider releases its own reference once this returns, so the
    // allocator takes one to keep it.
    void (*free)(void* ctx, ID3D12Resource* resource);
};

static const char* const kBfxDmlExternalAllocator = "ep.dml.bfx_external_allocator";
