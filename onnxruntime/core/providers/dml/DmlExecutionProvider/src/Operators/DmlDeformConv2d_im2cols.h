#pragma once

#include "../../../OperatorAuthorHelper/OperatorHelper.h"
#include "../MLOperatorAuthorImpl.h"

#include "../External/D3DX12/d3dx12.h"
#include <d3d12.h>

// NOTE: When this operator's implementation is moved into DML, the associated FP16 fallback
//       should be removed from IsCustomOpShader(...) in
//       onnxruntime\core\providers\dml\DmlExecutionProvider\src\ExecutionProvider.cpp

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
namespace deform_conv2d_im2cols_fp32 {
    #include "GeneratedShaders/deform_conv2d_im2cols_fp32.h"
}

namespace deform_conv2d_im2cols_fp16 {
    #include "GeneratedShaders/deform_conv2d_im2cols_fp16.h"
}

#include <wrl/client.h>
#include <wrl/implements.h>

#include <sstream>

using namespace Microsoft::WRL;

// Helper to derive dimensions and attributes from either the shape inferrer or the kernel constructor.
struct DmlDeformConv2d_im2cols_parameters
{
    // attributes
    int32_t kernel_h = 0;
    int32_t kernel_w = 0;
    int32_t stride_h = 0;
    int32_t stride_w = 0;
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    int32_t dil_h = 0;
    int32_t dil_w = 0;
    int32_t n_offset_grps = 0;
    int32_t use_mask = 0;

    // shapes, etc..
    int32_t n = 0; // batch size.
    int32_t in_ch = 0;
    int32_t in_h = 0;
    int32_t in_w = 0;

    int32_t out_h = 0;
    int32_t out_w = 0;

    // DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_UNKNOWN;
    MLOperatorTensorDataType dataType;

    DmlDeformConv2d_im2cols_parameters(){}

    DmlDeformConv2d_im2cols_parameters(
        const OperatorHelper::IKernelInformationAdapter& kernelInfo,
        const OperatorHelper::IShapeInformationAdapter& shapeInfo)
    {
        auto& attributes = kernelInfo.GetAttributes();

        kernel_h = static_cast<int32_t>(attributes.GetAttribute<int64_t>("kernel_h"));
        kernel_w = static_cast<int32_t>(attributes.GetAttribute<int64_t>("kernel_w"));
        pad_h = static_cast<int32_t>(attributes.GetAttribute<int64_t>("pad_h"));
        pad_w = static_cast<int32_t>(attributes.GetAttribute<int64_t>("pad_w"));
        stride_h = static_cast<int32_t>(attributes.GetAttribute<int64_t>("stride_h"));
        stride_w = static_cast<int32_t>(attributes.GetAttribute<int64_t>("stride_w"));
        dil_h = static_cast<int32_t>(attributes.GetAttribute<int64_t>("dil_h"));
        dil_w = static_cast<int32_t>(attributes.GetAttribute<int64_t>("dil_w"));
        n_offset_grps = static_cast<int32_t>(attributes.GetAttribute<int64_t>("n_offset_grps"));
        use_mask = static_cast<int32_t>(attributes.GetAttribute<int64_t>("use_mask"));

        // input 0: image (required; tensor)
        {
            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(0);
            ML_CHECK_VALID_ARGUMENT(rank == 4, "Input shape must be 4D.");
            auto dims = shapeInfo.GetInputTensorShape(0);
            assert(dims.size() == rank);
            n = static_cast<int32_t>(dims[0]);
            in_ch = static_cast<int32_t>(dims[1]);
            in_h = static_cast<int32_t>(dims[2]);
            in_w = static_cast<int32_t>(dims[3]);

            MLOperatorEdgeDescription edgeDesc = kernelInfo.GetInputEdgeDescription(0);
            assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);
            dataType = edgeDesc.tensorDataType;
        }

        // input 1: offset
        {
            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(1);
            ML_CHECK_VALID_ARGUMENT(rank == 4, "Offset shape must be 4D.");
            auto dims = shapeInfo.GetInputTensorShape(1);
            assert(dims.size() == rank);
            assert(n == dims[0]);
            assert(n_offset_grps * kernel_h * kernel_w * 2 == dims[1]);
            out_h = static_cast<int32_t>(dims[2]);
            out_w = static_cast<int32_t>(dims[3]);

            MLOperatorEdgeDescription edgeDesc = kernelInfo.GetInputEdgeDescription(1);
            assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);
            assert(dataType == edgeDesc.tensorDataType);
        }

        // input 2: mask
        if (use_mask) {
            uint32_t rank = shapeInfo.GetInputTensorDimensionCount(2);
            ML_CHECK_VALID_ARGUMENT(rank == 4, "mask shape must be 4D.");
            auto dims = shapeInfo.GetInputTensorShape(2);
            assert(dims.size() == rank);
            assert(n == dims[0]);
            assert(n_offset_grps * kernel_h * kernel_w == dims[1]);
            assert(out_h == dims[2]);
            assert(out_w == dims[3]);

            MLOperatorEdgeDescription edgeDesc = kernelInfo.GetInputEdgeDescription(1);
            assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);
            assert(dataType == edgeDesc.tensorDataType);
        }
    }

};

/*

namespace GridSampleHelpers
{
    // Divides and rounds
    inline uint32_t CeilDivide(uint32_t dividend, uint32_t divisor)
    {
        uint64_t temp = static_cast<uint64_t>(dividend) + divisor - 1;
        return static_cast<uint32_t>(temp / divisor);
    }

    // Gets the next number of elements to dispatch to the GPU within a loop handling a large
    // total number of tensor elements and threads.
    void GetNextDispatchSize(
        uint32_t elementCount,
        uint32_t elementsPerThread,
        uint32_t numThreads,
        _Out_ uint32_t& dispatch,
        _Out_ uint32_t& pendingElementCount
    )
    {
        // Max threads per workgroup is 2^10 (1024). Max dispatch per dimension is 2^16. Taken together, we can dispatch a maximum of
        // 2^26 (268,435,456) threads along a single dimension. This should suffice for a majority of the workload. Therefore, even
        // though it is possible to dispatch up to (2^16)^3 workgroups simultaneously, we stick to the simpler 1D dispatch alternative.
        assert(numThreads <= D3D12_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP);

        const uint32_t maxThreadsPerDispatch = numThreads * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

        const uint32_t requiredThreadCount = CeilDivide(elementCount, elementsPerThread);

        // Compute max dispatchable elements
        const uint32_t availableThreadCount = std::min(requiredThreadCount, maxThreadsPerDispatch);

        // Compute required thread group count
        uint32_t workGroupCount1D = CeilDivide(availableThreadCount, numThreads);

        // Compute min dispatch size
        dispatch = workGroupCount1D;

        // With the dispatch size computed, compute the dispatched element count
        const uint32_t dispatchedElementCount = workGroupCount1D * numThreads * elementsPerThread;

        // Update the pending element count
        pendingElementCount = (dispatchedElementCount < elementCount) ? elementCount - dispatchedElementCount : 0;
    }
}

*/

class DmlDeformConv2d_im2cols_operator : public WRL::Base<IMLOperatorKernel>
{
private:
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    DmlDeformConv2d_im2cols_parameters m_params = {};

    struct ShaderConstants
    {
        int32_t n_kernels; // total number of elements in output!
        int32_t n; // batch size
        int32_t n_in_channels;
        int32_t in_h;
        int32_t in_w;
        int32_t out_h;
        int32_t out_w;
        int32_t kernel_h;
        int32_t kernel_w;
        int32_t pad_h;
        int32_t pad_w;
        int32_t dil_h;
        int32_t dil_w;
        int32_t stride_h;
        int32_t stride_w;
        int32_t n_offset_grps;
        int32_t use_mask;
    };
    const static uint32_t ShaderConstants_numEls = 17; // num 32-bit values to set

public:

    DmlDeformConv2d_im2cols_operator(IMLOperatorKernelCreationContext* context)
    {
        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());

        ComPtr<ID3D12GraphicsCommandList> commandList;
        executionObject.As(&commandList);

        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));

        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};
        m_params = DmlDeformConv2d_im2cols_parameters(kernelInfo, shapeInfo);

        MLOperatorEdgeDescription inputEdgeDesc;
        ORT_THROW_IF_FAILED(context->GetInputEdgeDescription(0, &inputEdgeDesc));
        assert(inputEdgeDesc.edgeType == MLOperatorEdgeType::Tensor);
        assert(inputEdgeDesc.tensorDataType == m_params.dataType);

        // m_params
        // printf("TODO: assert all dtypes match whats expected from the params!\n");

        MLOperatorEdgeDescription offsetEdgeDesc;
        ORT_THROW_IF_FAILED(context->GetInputEdgeDescription(1, &offsetEdgeDesc));
        assert(offsetEdgeDesc.edgeType == MLOperatorEdgeType::Tensor);
        assert(offsetEdgeDesc.tensorDataType == m_params.dataType);

        if (m_params.use_mask) {
            MLOperatorEdgeDescription maskEdgeDesc;
            ORT_THROW_IF_FAILED(context->GetInputEdgeDescription(2, &maskEdgeDesc));
            assert(maskEdgeDesc.edgeType == MLOperatorEdgeType::Tensor);
            assert(maskEdgeDesc.tensorDataType == m_params.dataType);
        }

        PrepareKernel();
    }

    void PrepareKernel()
    {
        // Compute root signature.
        const int uavCount = 4; // 3 bound UAVs: input, offset, mask, output
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters;
        rootParameters.resize(uavCount + 1);

        for (uint32_t i = 0; i < uavCount; i++)
        {
            rootParameters[i].InitAsUnorderedAccessView(i);
        }

        rootParameters[uavCount].InitAsConstants(ShaderConstants_numEls, 0);

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

        switch (m_params.dataType)
        {
            case MLOperatorTensorDataType::Float:
            {
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(deform_conv2d_im2cols_fp32::g_deform_conv2d_im2cols, sizeof(deform_conv2d_im2cols_fp32::g_deform_conv2d_im2cols));
                break;
            }

            case MLOperatorTensorDataType::Float16:
            {
                computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(deform_conv2d_im2cols_fp16::g_deform_conv2d_im2cols, sizeof(deform_conv2d_im2cols_fp16::g_deform_conv2d_im2cols));
                break;
            }

            default:
                printf("ERROR! unsupported data type for deform_conv2d_im2cols!");
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

            // Get the input tensor
            ComPtr<IMLOperatorTensor> inputTensor;
            ORT_THROW_IF_FAILED(context->GetInputTensor(0, inputTensor.GetAddressOf()));

            // Get the offset tensor
            ComPtr<IMLOperatorTensor> offsetTensor;
            ORT_THROW_IF_FAILED(context->GetInputTensor(1, offsetTensor.GetAddressOf()));

            // Get the mask tensor
            ComPtr<IMLOperatorTensor> maskTensor;
            ORT_THROW_IF_FAILED(context->GetInputTensor(2, maskTensor.GetAddressOf()));

            // Get the output tensor
            ComPtr<IMLOperatorTensor> outputTensor;
            context->GetOutputTensor(0, outputTensor.GetAddressOf());

            if (outputTensor->IsCpuData() || inputTensor->IsCpuData() || offsetTensor->IsCpuData() || maskTensor->IsCpuData())
            {
                return E_UNEXPECTED;
            }

            ComPtr<IUnknown> executionObject;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            context->GetExecutionInterface(executionObject.GetAddressOf());
            executionObject.As(&commandList);

            // Get the input and output shape sizes
            auto inputDims = GetTensorDimensions(inputTensor.Get());
            auto offsetDims = GetTensorDimensions(offsetTensor.Get());
            auto maskDims = GetTensorDimensions(maskTensor.Get());
            auto outputDims = GetTensorDimensions(outputTensor.Get());

            ComPtr<IUnknown> inputUnknown;
            ComPtr<ID3D12Resource> inputResource;
            inputTensor->GetDataInterface(inputUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(inputUnknown.As(&inputResource));

            ComPtr<IUnknown> offsetUnknown;
            ComPtr<ID3D12Resource> offsetResource;
            offsetTensor->GetDataInterface(offsetUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(offsetUnknown.As(&offsetResource));

            ComPtr<IUnknown> maskUnknown;
            ComPtr<ID3D12Resource> maskResource;
            maskTensor->GetDataInterface(maskUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(maskUnknown.As(&maskResource));

            ComPtr<IUnknown> outputUnknown;
            ComPtr<ID3D12Resource> outputResource;
            outputTensor->GetDataInterface(outputUnknown.GetAddressOf());
            ORT_THROW_IF_FAILED(outputUnknown.As(&outputResource));

            return Compute(
                commandList.Get(),
                context,
                inputResource.Get(),
                inputDims,
                offsetResource.Get(),
                offsetDims,
                maskResource.Get(),
                maskDims,
                outputResource.Get(),
                outputDims
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
        ID3D12Resource* inputResource,
        gsl::span<const uint32_t> inputDims,
        ID3D12Resource* offsetResource,
        gsl::span<const uint32_t> offsetDims,
        ID3D12Resource* maskResource,
        gsl::span<const uint32_t> maskDims,
        ID3D12Resource* outputResource,
        gsl::span<const uint32_t> outputDims)
    {
        try
        {
            run(
                inputResource,
                inputDims,
                offsetResource,
                offsetDims,
                maskResource,
                maskDims,
                outputResource,
                outputDims,
                commandList);
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }

    void run(
        ID3D12Resource* inputResource,
        gsl::span<const uint32_t> inputDims,
        ID3D12Resource* offsetResource,
        gsl::span<const uint32_t> offsetDims,
        ID3D12Resource* maskResource,
        gsl::span<const uint32_t> maskDims,
        ID3D12Resource* outputResource,
        gsl::span<const uint32_t> outputDims,
        ID3D12GraphicsCommandList* commandList)
    {

        // Transition resources from common to UAV state
        D3D12_RESOURCE_BARRIER barriers[4];

        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            inputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            offsetResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
            maskResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        barriers[3] = CD3DX12_RESOURCE_BARRIER::Transition(
            outputResource,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

        inputResource->SetName(L"InputResource");
        offsetResource->SetName(L"OffsetResource");
        maskResource->SetName(L"MaskResource");
        outputResource->SetName(L"OutputResource");

        commandList->ResourceBarrier(4, barriers);

        // Set the root signature and pipeline state
        commandList->SetComputeRootSignature(m_rootSignature.Get());
        commandList->SetPipelineState(m_pipelineState.Get());

        // Each iteration of the below loop represents 1 level in the Stockham DFT
        // Dispatch in a loop
        ShaderConstants constants = {};

        constants.n = m_params.n;
        constants.n_in_channels = m_params.in_ch;
        constants.in_h = m_params.in_h;
        constants.in_w = m_params.in_w;
        constants.out_h = m_params.out_h;
        constants.out_w = m_params.out_w;
        constants.kernel_h = m_params.kernel_h;
        constants.kernel_w = m_params.kernel_w;
        constants.pad_h = m_params.pad_h;
        constants.pad_w = m_params.pad_w;
        constants.dil_h = m_params.dil_h;
        constants.dil_w = m_params.dil_w;
        constants.stride_h = m_params.stride_h;
        constants.stride_w = m_params.stride_w;
        constants.n_offset_grps = m_params.n_offset_grps;
        constants.use_mask = m_params.use_mask;

        constants.n_kernels =
            constants.n * constants.n_in_channels * constants.out_h * constants.out_w;

        std::array<ID3D12Resource*, 4> uav_resources = { inputResource, offsetResource, maskResource, outputResource };
        Dispatch(uav_resources, constants, commandList);

        // Transition resources to common state
        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
                inputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
                offsetResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
                maskResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        barriers[3] = CD3DX12_RESOURCE_BARRIER::Transition(
                outputResource,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON
                );

        commandList->ResourceBarrier(4, barriers);

    }

    std::vector<uint32_t> GetTensorDimensions(IMLOperatorTensor* tensor)
    {
        auto inputDimsSize = tensor->GetDimensionCount();
        auto dims = std::vector<uint32_t>(inputDimsSize);
        ORT_THROW_IF_FAILED(tensor->GetShape(static_cast<uint32_t>(dims.size()), dims.data()));
        return dims;
    }

    template <typename TConstants, uint32_t TSize>
    void Dispatch(
        std::array<ID3D12Resource*, TSize>& resources,
        TConstants& constants,
        ID3D12GraphicsCommandList* commandList)
    {

        D3D12_RESOURCE_BARRIER uav_barriers[TSize];

        std::transform(
            resources.begin(), resources.end(),
            uav_barriers,
            [](auto& resource) { return CD3DX12_RESOURCE_BARRIER::UAV(resource); } );
        commandList->ResourceBarrier(TSize, uav_barriers);

        for (uint32_t i = 0; i < TSize; i++)
        {
            // Set resource views
            if (resources[i]) {
                commandList->SetComputeRootUnorderedAccessView(
                    i, // root parameter index
                    resources[i]->GetGPUVirtualAddress()
                );
            }
            else
            {
                commandList->SetComputeRootUnorderedAccessView(
                    i, // root parameter index
                    {}
                );

            }
        }

        // auto pendingElementCount = constants.ElementCount;

        // Dispatch up to the maximum number of threads per iteration until
        // all elements are completed
        // while (pendingElementCount > 0)
        // {
            // constants.StartIndex = constants.ElementCount - pendingElementCount;

            const uint32_t numThreads = 256;

            const uint32_t numGroups = (constants.n_kernels + numThreads - 1) / numThreads;

            if (numGroups > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) {
                printf("ERROR! TODO: handle deform_conv2d_im2cols invocation larger than max D3D12 thread group size\n");
            }

            // GridSampleHelpers::GetNextDispatchSize(
            //     pendingElementCount,
            //     1,
            //     64,
            //     dispatchSizeX,
            //     pendingElementCount
            // );

            // Set root constants
            commandList->SetComputeRoot32BitConstants(
                TSize, // root parameter index
                ShaderConstants_numEls, // Constant count
                &constants,
                0 // offset
            );

            commandList->Dispatch(numGroups, 1, 1);
        // }

        commandList->ResourceBarrier(2, uav_barriers);

    }
};

struct DeformConv2d_im2cols_shapeInferrer : public WRL::Base<IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept
    {
        try
        {
            ComPtr<IMLOperatorShapeInferenceContextPrivate> contextPrivate;
            ORT_THROW_IF_FAILED(context->QueryInterface(IID_PPV_ARGS(&contextPrivate)));

            MLShapeInferenceContext inferenceContext(context);
            OperatorHelper::KernelInformationAdapter kernelInfo{inferenceContext};
            OperatorHelper::ShapeInformationAdapter shapeInfo{inferenceContext};
            DmlDeformConv2d_im2cols_parameters params(kernelInfo, shapeInfo);

            std::array<uint32_t, 2> outputDims = {
                (uint32_t)(params.in_ch * params.kernel_h * params.kernel_w),
                (uint32_t)(params.out_h * params.out_w)
            };

            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(0, onnxruntime::narrow<uint32_t>(outputDims.size()), outputDims.data()));
        }
        catch (...)
        {
            return E_FAIL;
        }

        return S_OK;
    }
};

class DmlDeformConv2d_im2cols_operatorFactory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        IMLOperatorKernel** kernel)
    {
        try
        {
            auto dftOperator = wil::MakeOrThrow<DmlDeformConv2d_im2cols_operator>(context);
            dftOperator.CopyTo(kernel);
            return S_OK;
        }
        catch (...)
        {
            return E_FAIL;
        }
    }

    static void RegisterDeformConv2d_im2cols_Kernel(IMLOperatorRegistry* registry)
    {
        MLOperatorKernelDescription kernelDescription = {};
        kernelDescription.domain = "bfx";
        kernelDescription.name = "deform_conv2d_im2cols";
        kernelDescription.minimumOperatorSetVersion = 1;
        kernelDescription.executionType = MLOperatorExecutionType::D3D12;

        // TODO: are these constraints needed??

        /*

        // T1: tensor(float16), tensor(float)
        MLOperatorEdgeTypeConstrant t1Constraint;
        t1Constraint.typeLabel = "T1";
        std::vector<MLOperatorEdgeDescription> t1AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float16 },
        };
        t1Constraint.allowedTypes = t1AllowedEdges.data();
        t1Constraint.allowedTypeCount = static_cast<uint32_t>(t1AllowedEdges.size());

        // T2 : tensor(int32), tensor(int64)
        MLOperatorEdgeTypeConstrant t2Constraint;
        t2Constraint.typeLabel = "T2";
        std::vector<MLOperatorEdgeDescription> t2AllowedEdges
        {
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
            MLOperatorEdgeDescription { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float16 },
        };
        t2Constraint.allowedTypes = t2AllowedEdges.data();
        t2Constraint.allowedTypeCount = static_cast<uint32_t>(t2AllowedEdges.size());

        std::vector<MLOperatorEdgeTypeConstrant> typeConstraints{ t1Constraint, t2Constraint };
        kernelDescription.typeConstraints = typeConstraints.data();
        kernelDescription.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        */

       /*

        MLOperatorAttributeNameValue kernel_h_AttributeValue;
        kernel_h_AttributeValue.name = "kernel_h";
        kernel_h_AttributeValue.type = MLOperatorAttributeType::Int;
        kernel_h_AttributeValue.valueCount = 1;
        static const int64_t kernel_h[] = { 0 };
        kernel_h_AttributeValue.ints = kernel_h;

        */

        /*

        MLOperatorAttributeNameValue modeAttributeValue;
        modeAttributeValue.name = AttrName::Mode;
        modeAttributeValue.type = MLOperatorAttributeType::String;
        modeAttributeValue.valueCount = 1;
        static const char* modes[] = { "bilinear" };
        modeAttributeValue.strings = modes;

        MLOperatorAttributeNameValue paddingModeAttributeValue;
        paddingModeAttributeValue.name = AttrName::Mode;
        paddingModeAttributeValue.type = MLOperatorAttributeType::String;
        paddingModeAttributeValue.valueCount = 1;
        static const char* paddingModes[] = { "zeros" };
        paddingModeAttributeValue.strings = paddingModes;

        */

        // std::vector<MLOperatorAttributeNameValue> attributeDefaultValues{
            // kernel_h_AttributeValue,
            // modeAttributeValue,
            // paddingModeAttributeValue
        // };

        // kernelDescription.defaultAttributes = attributeDefaultValues.data();
        // kernelDescription.defaultAttributeCount = static_cast<uint32_t>(attributeDefaultValues.size());
        kernelDescription.options = MLOperatorKernelOptions::None;
        kernelDescription.executionOptions = 0;

        auto shareInferrer = wil::MakeOrThrow<DeformConv2d_im2cols_shapeInferrer>();
        auto factory = wil::MakeOrThrow<DmlDeformConv2d_im2cols_operatorFactory>();

        ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
        ORT_THROW_IF_FAILED(registry->QueryInterface(IID_PPV_ARGS(&registryPrivate)));

        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &kernelDescription,
            factory.Get(),
            shareInferrer.Get(),
            nullptr,
            false, // isInternalOperator
            false, // alias
            false, // supportsGraph
            nullptr,
            nullptr,
            0));

    }
};
