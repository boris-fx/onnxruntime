#pragma once

// #include <functional>

namespace bfx_ops {

template <typename op_type>
class op_factory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        try
        {
            auto op = wil::MakeOrThrow<op_type>(context);
            op.CopyTo(kernel);
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }
};

template <typename op_type>
struct shape_inferrer : public WRL::Base<IMLOperatorShapeInferrer> {
STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
{
    try
    {
        ComPtr<IMLOperatorShapeInferenceContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));

        MLShapeInferenceContext inferenceContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{inferenceContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{inferenceContext};
        typename op_type::op_params params(kernelInfo, shapeInfo);

        auto shapes = op_type::infer_shapes(params);

        for (uint32_t i =0; i < shapes.size(); i++) {
            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(i, onnxruntime::narrow<uint32_t>(shapes[i].size()), shapes[i].data()));
        }
    }
    catch (...)
    {
        return E_FAIL;
    }

    return S_OK;
}
};

template <typename op_type>
struct shape_inferrer2 : public WRL::Base<IMLOperatorShapeInferrer> {
STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
{
    try
    {
        auto shapes = op_type::infer_shapes(context);
        for (uint32_t i =0; i < shapes.size(); i++) {
            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(i, onnxruntime::narrow<uint32_t>(shapes[i].size()), shapes[i].data()));
        }
    }
    catch (...)
    {
        return E_FAIL;
    }

    return S_OK;
}
};

void register_operator_kernel(IMLOperatorRegistry* registry, const char* opName, IMLOperatorKernelFactory* kernelFactory, IMLOperatorShapeInferrer* shapeInferrer) {
    MLOperatorKernelDescription kernelDescription = {};
    kernelDescription.domain = "bfx";
    kernelDescription.name = opName;
    kernelDescription.minimumOperatorSetVersion = 1;
    kernelDescription.executionType = MLOperatorExecutionType::D3D12;
    kernelDescription.options = MLOperatorKernelOptions::None;
    kernelDescription.executionOptions = 0;

    ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
    ORT_THROW_IF_FAILED(registry->QueryInterface(IID_PPV_ARGS(&registryPrivate)));
    ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
        &kernelDescription,
        kernelFactory,
        shapeInferrer,
        nullptr, // supportQuery
        false, // isInternalOperator
        false, // alias
        false));//, // supportsGraph
        // nullptr,
        // nullptr,
        // 0));
}

template <typename op_type>
void register_operator_kernel(IMLOperatorRegistry* registry)
{
    auto shapeInferrer = wil::MakeOrThrow<shape_inferrer<op_type>>();
    auto factory = wil::MakeOrThrow<op_factory<custom_op<op_type>>>();
    register_operator_kernel(registry, op_type::op_name, factory.Get(), shapeInferrer.Get());
}

template <typename op_type>
void register_operator_kernel2(IMLOperatorRegistry* registry)
{
    auto shapeInferrer = wil::MakeOrThrow<shape_inferrer2<op_type>>();
    auto factory = wil::MakeOrThrow<op_factory<op_type>>();
    register_operator_kernel(registry, op_type::op_name, factory.Get(), shapeInferrer.Get());
}

// util
std::vector<uint32_t> GetTensorDimensions(IMLOperatorTensor* tensor)
{
    auto inputDimsSize = tensor->GetDimensionCount();
    auto dims = std::vector<uint32_t>(inputDimsSize);
    ORT_THROW_IF_FAILED(tensor->GetShape(static_cast<uint32_t>(dims.size()), dims.data()));
    return dims;
}

template <typename op_type>
class custom_op : public WRL::Base<IMLOperatorKernel>
{
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;

    typename op_type::op_params m_params = {};


public:

    custom_op(IMLOperatorKernelCreationContext* context)
    {
        // load params
        // this is an opaque object that is passed to the custom op implementation at runtime
        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};
        m_params = op_type::op_params(kernelInfo, shapeInfo);

        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());

        ComPtr<ID3D12GraphicsCommandList> commandList;
        executionObject.As(&commandList);

        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));


        // make sure all input data types match!
        MLOperatorEdgeDescription edgeDesc0 = kernelInfo.GetInputEdgeDescription(0);
        assert(edgeDesc0.edgeType == MLOperatorEdgeType::Tensor);
        auto dataType = edgeDesc0.tensorDataType;
        // for (uint32_t i = 1; i < op_type::num_inputs; i++) {
            // auto edgeDesc_i = kernelInfo.GetInputEdgeDescription(i);
            // assert(dataType == edgeDesc_i.tensorDataType);
        // }

        // Compute root signature.
        const int uavCount = op_type::num_inputs + op_type::num_outputs; // 3 bound UAVs: input, offset, mask, output
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
        rootParameters.resize(uavCount + 1);

        for (uint32_t i = 0; i < (uint32_t)uavCount; i++)
        {
            rootParameters[i].InitAsUnorderedAccessView(i);
        }

        rootParameters[uavCount].InitAsConstants(op_type::num_els_constants, 0);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());

        ComPtr<ID3DBlob> rootSignatureBlob;
        ComPtr<ID3DBlob> rootSignatureErrorBlob;
        ORT_THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(
            &desc,
            rootSignatureBlob.GetAddressOf(),
            rootSignatureErrorBlob.GetAddressOf()
        ));

        ORT_THROW_IF_FAILED(m_device->CreateRootSignature(
            0,
            rootSignatureBlob->GetBufferPointer(),
            rootSignatureBlob->GetBufferSize(),
            IID_ID3D12RootSignature,
            &m_rootSignature
        ));

        // Describe and create the compute pipeline state object (PSO).
        D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
        computePsoDesc.pRootSignature = m_rootSignature.Get();

        switch (dataType)
        {
            case MLOperatorTensorDataType::Float:
            {
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(op_type::bytecode_fp32, op_type::bytecode_fp32_size);
                break;
            }

            case MLOperatorTensorDataType::Float16:
            {
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(op_type::bytecode_fp16, op_type::bytecode_fp16_size);
                break;
            }

            default:
                printf("ERROR! unsupported data type for bfx custom compute kernel!");
                break;
        }

        ORT_THROW_IF_FAILED(m_device->CreateComputePipelineState(&computePsoDesc, IID_ID3D12PipelineState, &m_pipelineState));

    }

    // Computes the outputs of the kernel.  This may be called multiple times
    // simultaneously within the same instance of the class.  Implementations
    // of this method must be thread-safe.
    STDMETHOD(Compute)(IMLOperatorKernelContext* context)
    {
        try
        {
            std::vector<ComPtr<ID3D12Resource>> input_resources;
            std::vector<std::vector<uint32_t>> input_dims;
            for (int i = 0; i < op_type::num_inputs; i++) {
                // Get the input tensor
                ComPtr<IMLOperatorTensor> t;
                ORT_THROW_IF_FAILED(context->GetInputTensor(i, t.GetAddressOf()));
                if (t->IsCpuData()) { return E_UNEXPECTED; }
                auto dims = GetTensorDimensions(t.Get());

                ComPtr<IUnknown> u;
                ComPtr<ID3D12Resource> r;
                t->GetDataInterface(u.GetAddressOf());
                ORT_THROW_IF_FAILED(u.As(&r));

                input_resources.push_back(r);
                input_dims.push_back(dims);
            }

            std::vector<ComPtr<ID3D12Resource>> output_resources;
            std::vector<std::vector<uint32_t>> output_dims;
            for (int i = 0; i < op_type::num_outputs; i++) {
                // Get the output tensor
                ComPtr<IMLOperatorTensor> t;
                ORT_THROW_IF_FAILED(context->GetOutputTensor(i, t.GetAddressOf()));
                if (t->IsCpuData()) { return E_UNEXPECTED; }
                auto dims = GetTensorDimensions(t.Get());

                ComPtr<IUnknown> u;
                ComPtr<ID3D12Resource> r;
                t->GetDataInterface(u.GetAddressOf());
                ORT_THROW_IF_FAILED(u.As(&r));

                output_resources.push_back(r);
                output_dims.push_back(dims);
            }

            ComPtr<IUnknown> executionObject;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            context->GetExecutionInterface(executionObject.GetAddressOf());
            executionObject.As(&commandList);

            return Compute(
                commandList.Get(),
                context,
                input_resources, input_dims,
                output_resources, output_dims
            );

        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }

    HRESULT Compute(
        ID3D12GraphicsCommandList* commandList,
        IMLOperatorKernelContext* context,
        std::vector<ComPtr<ID3D12Resource>>& input_resources,
        std::vector<std::vector<uint32_t>>& input_dims,
        std::vector<ComPtr<ID3D12Resource>>& output_resources,
        std::vector<std::vector<uint32_t>>& output_dims)
    {
        try
        {
            run(
                input_resources,
                input_dims,
                output_resources,
                output_dims,
                commandList);
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }

    void run(
        std::vector<ComPtr<ID3D12Resource>>& input_resources,
        std::vector<std::vector<uint32_t>>& input_dims,
        std::vector<ComPtr<ID3D12Resource>>& output_resources,
        std::vector<std::vector<uint32_t>>& output_dims,
        ID3D12GraphicsCommandList* commandList)
    {

        const size_t num_resources = input_resources.size() + output_resources.size();
        std::vector<D3D12_RESOURCE_BARRIER> barriers(num_resources);

        std::vector<ID3D12Resource*> uav_resources(num_resources, nullptr);

        for (int i =0; i < input_resources.size(); i++) {
            barriers[i] = CD3DX12_RESOURCE_BARRIER::Transition(
                input_resources[i].Get(),
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            );
            uav_resources[i] = input_resources[i].Get();
        }

        for (int i = 0; i < output_resources.size(); i++) {
            barriers[i + input_resources.size()] = CD3DX12_RESOURCE_BARRIER::Transition(
                output_resources[i].Get(),
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            );
            uav_resources[i + input_resources.size()] = output_resources[i].Get();
        }

        commandList->ResourceBarrier((uint32_t)num_resources, barriers.data());

        // Set the root signature and pipeline state
        commandList->SetComputeRootSignature(m_rootSignature.Get());
        commandList->SetPipelineState(m_pipelineState.Get());

        Dispatch(uav_resources, commandList);

        // Transition resources to common state
        for (int i =0; i < input_resources.size(); i++) {
            barriers[i] = CD3DX12_RESOURCE_BARRIER::Transition(
                input_resources[i].Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
            );
        }

        for (int i = 0; i < output_resources.size(); i++) {
            barriers[i + input_resources.size()] = CD3DX12_RESOURCE_BARRIER::Transition(
                output_resources[i].Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
            );
        }
        commandList->ResourceBarrier((uint32_t)num_resources, barriers.data());
    }

    void Dispatch(
        std::vector<ID3D12Resource*>& resources,
        // TConstants& constants,
        ID3D12GraphicsCommandList* commandList)
    {

        const size_t TSize = resources.size();

        std::vector<D3D12_RESOURCE_BARRIER> uav_barriers(TSize);

        std::transform(
            resources.begin(), resources.end(),
            uav_barriers.data(),
            [](auto& resource) { return CD3DX12_RESOURCE_BARRIER::UAV(resource); } );
        commandList->ResourceBarrier((uint32_t)TSize, uav_barriers.data());

        for (uint32_t i = 0; i < TSize; i++)
        {
            // Set resource views
            if (resources[i]) {
                commandList->SetComputeRootUnorderedAccessView(i, resources[i]->GetGPUVirtualAddress());
            }
            else
            {
                commandList->SetComputeRootUnorderedAccessView(i, {});
            }
        }

        op_type::run(
            m_params,
            [&](void* constants_ptr) {
                commandList->SetComputeRoot32BitConstants(
                    (uint32_t)TSize,
                    op_type::num_els_constants,
                    constants_ptr,
                    0);
            },
            [&](uint32_t x, uint32_t y, uint32_t z) { commandList->Dispatch(x, y, z); }
        );

        // TODO: why '2' here??
        // commandList->ResourceBarrier(2, uav_barriers.data());
    }


};

class custom_op2 : public WRL::Base<IMLOperatorKernel>
{
protected:
    ComPtr<ID3D12Device> m_device;

public:

    virtual int numInputs() = 0;
    virtual int numOutputs() = 0;

    virtual void run(
        ComPtr<ID3D12GraphicsCommandList> commandList,
        IMLOperatorKernelContext* context,
        std::vector<ComPtr<ID3D12Resource>>& input_resources,
        std::vector<std::vector<uint32_t>>& input_dims,
        std::vector<ComPtr<ID3D12Resource>>& output_resources,
        std::vector<std::vector<uint32_t>>& output_dims) = 0;

    explicit custom_op2(IMLOperatorKernelCreationContext* context) {
        // load params
        // this is an opaque object that is passed to the custom op implementation at runtime
        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};

        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());

        ComPtr<ID3D12GraphicsCommandList> commandList;
        executionObject.As(&commandList);

        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));
    }

    STDMETHOD(Compute)(IMLOperatorKernelContext* context) {
        try
        {
            std::vector<ComPtr<ID3D12Resource>> input_resources;
            std::vector<std::vector<uint32_t>> input_dims;
            for (int i = 0; i < numInputs(); i++) {
                // Get the input tensor
                ComPtr<IMLOperatorTensor> t;
                ORT_THROW_IF_FAILED(context->GetInputTensor(i, t.GetAddressOf()));
                if (t->IsCpuData()) { return E_UNEXPECTED; }
                auto dims = GetTensorDimensions(t.Get());

                ComPtr<IUnknown> u;
                ComPtr<ID3D12Resource> r;
                t->GetDataInterface(u.GetAddressOf());
                ORT_THROW_IF_FAILED(u.As(&r));

                input_resources.push_back(r);
                input_dims.push_back(dims);
            }

            std::vector<ComPtr<ID3D12Resource>> output_resources;
            std::vector<std::vector<uint32_t>> output_dims;
            for (int i = 0; i < numOutputs(); i++) {
                // Get the output tensor
                ComPtr<IMLOperatorTensor> t;
                ORT_THROW_IF_FAILED(context->GetOutputTensor(i, t.GetAddressOf()));
                if (t->IsCpuData()) { return E_UNEXPECTED; }
                auto dims = GetTensorDimensions(t.Get());

                ComPtr<IUnknown> u;
                ComPtr<ID3D12Resource> r;
                t->GetDataInterface(u.GetAddressOf());
                ORT_THROW_IF_FAILED(u.As(&r));

                output_resources.push_back(r);
                output_dims.push_back(dims);
            }

            ComPtr<IUnknown> executionObject;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            context->GetExecutionInterface(executionObject.GetAddressOf());
            executionObject.As(&commandList);

            run(
                commandList,
                context,
                input_resources, input_dims,
                output_resources, output_dims);
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }
};

}
