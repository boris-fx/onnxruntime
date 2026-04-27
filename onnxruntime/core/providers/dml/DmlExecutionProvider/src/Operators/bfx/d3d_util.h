// some boilerplate related to d3d12!
#pragma once

namespace bfx_ops {

struct ComputeShaderConfig {
    const void* bytecode;
    size_t bytecode_size;
    int num_bindings;
    int num_constants;
};

// wrapper for single compute shader!
class ComputeShader {
public:

    explicit ComputeShader(ComPtr<ID3D12Device> device, const ComputeShaderConfig& cfg) {
        init(device, cfg.bytecode, cfg.bytecode_size, cfg.num_bindings, cfg.num_constants);
    }

    void run(
            ComPtr<ID3D12GraphicsCommandList> commandList,
            std::vector<int64_t> grid,
            std::vector<ComPtr<ID3D12Resource>> resources,
            void* constants) {

        const size_t num_resources = resources.size();
        std::vector<D3D12_RESOURCE_BARRIER> barriers(num_resources);

        std::vector<ID3D12Resource*> uav_resources(num_resources, nullptr);

        for (int i =0; i < resources.size(); i++) {
            barriers[i] = CD3DX12_RESOURCE_BARRIER::Transition(
                resources[i].Get(),
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            );
            uav_resources[i] = resources[i].Get();
        }

        commandList->ResourceBarrier((uint32_t)num_resources, barriers.data());

        // Set the root signature and pipeline state
        commandList->SetComputeRootSignature(m_rootSignature.Get());
        commandList->SetPipelineState(m_pipelineState.Get());

        std::vector<D3D12_RESOURCE_BARRIER> uav_barriers(num_resources);

        std::transform(
            uav_resources.begin(), uav_resources.end(),
            uav_barriers.data(),
            [](auto& resource) { return CD3DX12_RESOURCE_BARRIER::UAV(resource); } );
        commandList->ResourceBarrier((uint32_t)num_resources, uav_barriers.data());

        for (uint32_t i = 0; i < num_resources; i++)
        {
            // Set resource views
            if (uav_resources[i]) {
                commandList->SetComputeRootUnorderedAccessView(i, uav_resources[i]->GetGPUVirtualAddress());
            }
            else
            {
                commandList->SetComputeRootUnorderedAccessView(i, {});
            }
        }

        commandList->SetComputeRoot32BitConstants(
            (uint32_t)num_resources,
            m_num_constants,
            constants,
            0);

        // dispatch grid shape.
        // block shape is baked into shader code!
        commandList->Dispatch(
            grid.size() > 0 ? static_cast<unsigned int>(grid[0]) : 1,
            grid.size() > 1 ? static_cast<unsigned int>(grid[1]) : 1,
            grid.size() > 2 ? static_cast<unsigned int>(grid[2]) : 1);

        // Execution done, transition resources to common state
        for (int i =0; i < resources.size(); i++) {
            barriers[i] = CD3DX12_RESOURCE_BARRIER::Transition(
                resources[i].Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
            );
        }

        commandList->ResourceBarrier((uint32_t)num_resources, barriers.data());

    }


private:

    void init(ComPtr<ID3D12Device> device, const void* bytecode, size_t bytecode_size, int num_bindings, int num_constants) {

        m_device = device;
        m_num_constants = num_constants;

        // Compute root signature.
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
        rootParameters.resize(num_bindings + 1);

        for (uint32_t i = 0; i < (uint32_t)num_bindings; i++)
        {
            rootParameters[i].InitAsUnorderedAccessView(i);
        }

        rootParameters[num_bindings].InitAsConstants(num_constants, 0);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());

        ComPtr<ID3DBlob> rootSignatureBlob;
        ComPtr<ID3DBlob> rootSignatureErrorBlob;
        ORT_THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(
            &desc,
            rootSignatureBlob.GetAddressOf(),
            rootSignatureErrorBlob.GetAddressOf()
        ));

        ORT_THROW_IF_FAILED(m_device->CreateRootSignature(0,
            rootSignatureBlob->GetBufferPointer(),
            rootSignatureBlob->GetBufferSize(),
            IID_ID3D12RootSignature,
            &m_rootSignature
        ));

        // Describe and create the compute pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_rootSignature.Get();
        computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(bytecode, bytecode_size);

        ORT_THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));
    }

    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    int m_num_constants;
};

ComPtr<ID3D12Resource> CreateD3D12ResourceOfByteSize(
    ID3D12Device* d3dDevice,
    size_t resourceByteSize,
    D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_FLAGS resourceFlags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    )
{
    resourceByteSize = std::max(resourceByteSize, size_t(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));

    // DML needs the resources' sizes to be a multiple of 4 bytes
    (resourceByteSize += 3) &= ~3;

    D3D12_HEAP_PROPERTIES const heapProperties =
    {
        heapType, // Type, Default to D3D12_HEAP_TYPE_DEFAULT.
        D3D12_CPU_PAGE_PROPERTY_UNKNOWN, // CPUPageProperty
        D3D12_MEMORY_POOL_UNKNOWN, // MemoryPoolPreference
        1, // CreationNodeMask
        1 // VisibleNodeMask
    };

    D3D12_RESOURCE_DESC const resourceDesc =
    {
        D3D12_RESOURCE_DIMENSION_BUFFER, // Dimension
        0, // Alignment
        static_cast<uint64_t>(resourceByteSize), // Width
        1, // Height
        1, // DepthOrArraySize
        1, // MipLevels
        DXGI_FORMAT_UNKNOWN, // Format
        {1, 0}, // SampleDesc
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR, // Layout
        resourceFlags // Flags, Default to D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
    };

    ComPtr<ID3D12Resource> gpuResource;
    ORT_THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        resourceState, // Default to D3D12_RESOURCE_STATE_COMMON
        nullptr,
        __uuidof(ID3D12Resource),
        /*out*/ &gpuResource
    ));

    return gpuResource;
}

class ComputeBuffer {
public:
    explicit ComputeBuffer(ComPtr<ID3D12Device> device, size_t sz) {
        m_resource = CreateD3D12ResourceOfByteSize(device.Get(), sz);
    }

    ComPtr<ID3D12Resource> resource() {
        return m_resource;
    }

private:

ComPtr<ID3D12Resource> m_resource;

};

void resourceBarrier(ComPtr<ID3D12GraphicsCommandList> commandList, ComPtr<ID3D12Resource> r, D3D12_RESOURCE_STATES pre, D3D12_RESOURCE_STATES post)
{
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(r.Get(), pre, post);
    commandList->ResourceBarrier(1, &barrier);
}

void copyBufferRegion(
        ComPtr<ID3D12GraphicsCommandList> commandList,
        ComPtr<ID3D12Resource> dst, size_t dst_offset,
        ComPtr<ID3D12Resource> src, size_t src_offset,
        size_t sz) {

    resourceBarrier(commandList, src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    resourceBarrier(commandList, dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);

    commandList->CopyBufferRegion(dst.Get(), dst_offset, src.Get(), src_offset, sz);

    resourceBarrier(commandList, src, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    resourceBarrier(commandList, dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

}
