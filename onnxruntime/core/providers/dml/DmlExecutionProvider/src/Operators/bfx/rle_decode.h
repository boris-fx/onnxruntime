#pragma once

#include "../../../../OperatorAuthorHelper/OperatorHelper.h"
#include "../../MLOperatorAuthorImpl.h"

#include "../../External/D3DX12/d3dx12.h"
#include <d3d12.h>

#include <wrl/client.h>
#include <wrl/implements.h>

#include <sstream>

#include "bfx_ops.h"
#include "d3d_util.h"

using namespace Microsoft::WRL;

namespace bfx_ops {

#include "scan/scan_clear_buffer_hlsl.h"
#include "rle_decode_scatter_hlsl.h"

#include "scan/scan.h"

class rle_decode : public custom_op2
{
public:
    const static inline char* op_name = "rle_decode";

    std::shared_ptr<ComputeShader> m_shader_clear_buffer;
    std::shared_ptr<ComputeShader> m_shader_scatter;

    std::shared_ptr<ComputeBuffer> m_tmp0;

    std::shared_ptr<PrefixSum> m_pfx_sum;

    int m_n_els;

    static std::vector<std::vector<uint32_t>> infer_shapes(IMLOperatorShapeInferenceContext* ctx) {
        MLShapeInferenceContext ctx_(ctx);
        OperatorHelper::KernelInformationAdapter kernelInfo{ctx_};
        OperatorHelper::ShapeInformationAdapter shapeInfo{ctx_};

        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorDimensionCount(0) == 1, "N input must be 1D");
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorShape(0)[0] == 1, "N input should have shape (1,)");

        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorDimensionCount(1) == 1, "diffs input shape must be 1D");
        auto n_els = shapeInfo.GetInputTensorShape(1)[0];

        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorDimensionCount(1) == 1, "idxs input shape must be 1D");
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorShape(2)[0] == n_els, "idxs input should match shape of diffs input");

        return {{n_els}};
    }

    explicit rle_decode(IMLOperatorKernelCreationContext* context) : custom_op2(context) {
        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};

        // uint32_t rank = shapeInfo.GetInputTensorDimensionCount(0);
        // ML_CHECK_VALID_ARGUMENT(rank == 1, "Input shape must be 1D");
        // m_n_els = shapeInfo.GetInputTensorShape(0)[0];

        m_n_els = shapeInfo.GetInputTensorShape(1)[0];

        // get_diffs
        m_shader_clear_buffer.reset(new ComputeShader(m_device, scan_clear_buffer_hlsl::cfg));
        m_shader_scatter.reset(new ComputeShader(m_device, rle_decode_scatter_hlsl::cfg));

        // allocate intermediate!
        m_tmp0.reset(new ComputeBuffer(m_device, (m_n_els + 1) * sizeof(int32_t)));

        m_pfx_sum.reset(new PrefixSum(m_device, m_n_els + 1));
    }

    int numInputs() override { return 3; }
    int numOutputs() override { return 1; }

    void run(ComPtr<ID3D12GraphicsCommandList> commandList,
            IMLOperatorKernelContext* context,
            std::vector<ComPtr<ID3D12Resource>>& input_resources,
            std::vector<std::vector<uint32_t>>& input_dims,
            std::vector<ComPtr<ID3D12Resource>>& output_resources,
            std::vector<std::vector<uint32_t>>& output_dims) override {

        auto enc_n = input_resources[0];
        auto enc_d = input_resources[1];
        auto enc_i = input_resources[2];
        auto x = output_resources[0];

        auto tmp0 = m_tmp0->resource();

        const auto block_size = 256; // matches shader!
        const auto grid_size = (m_n_els + block_size - 1) / block_size;

        scan_clear_buffer_hlsl::constants constants0{m_n_els + 1};
        m_shader_clear_buffer->run(commandList, { grid_size}, { x }, &constants0);

        rle_decode_scatter_hlsl::constants constants1{m_n_els};
        m_shader_scatter->run(commandList, { grid_size }, { enc_n, enc_i, enc_d, x }, &constants1);

        m_pfx_sum->run(commandList, tmp0, x);

        // copy result shifted by one: convert exclusive pfx scan to inclusive!
        bfx_ops::copyBufferRegion(commandList, x, 0, tmp0, sizeof(int32_t), (m_n_els) * sizeof(int32_t));
    }
};

} // bfx_ops
