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

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
#include "rle_encode_get_diffs_hlsl.h"
#include "rle_encode_write_out_hlsl.h"

#include "scan/scan.h"

class rle_encode : public custom_op2
{
public:
    const static inline char* op_name = "rle_encode";

    std::shared_ptr<ComputeShader> m_shader_get_diffs;
    std::shared_ptr<ComputeShader> m_shader_write_out;

    std::shared_ptr<ComputeBuffer> m_tmp_d;
    std::shared_ptr<ComputeBuffer> m_tmp_i;

    std::shared_ptr<PrefixSum> m_pfx_sum;

    int m_n_els;

    static std::vector<std::vector<uint32_t>> infer_shapes(IMLOperatorShapeInferenceContext* ctx) {
        MLShapeInferenceContext ctx_(ctx);
        OperatorHelper::KernelInformationAdapter kernelInfo{ctx_};
        OperatorHelper::ShapeInformationAdapter shapeInfo{ctx_};

        uint32_t rank = shapeInfo.GetInputTensorDimensionCount(0);
        ML_CHECK_VALID_ARGUMENT(rank == 1, "Input shape must be 1D");
        auto n_els = shapeInfo.GetInputTensorShape(0)[0];

        return {{1}, {n_els}, {n_els}};
    }

    explicit rle_encode(IMLOperatorKernelCreationContext* context) : custom_op2(context) {
        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};

        uint32_t rank = shapeInfo.GetInputTensorDimensionCount(0);
        ML_CHECK_VALID_ARGUMENT(rank == 1, "Input shape must be 1D");
        m_n_els = shapeInfo.GetInputTensorShape(0)[0];

        // get_diffs
        m_shader_get_diffs.reset(new ComputeShader(m_device, rle_encode_get_diffs_hlsl::cfg));
        m_shader_write_out.reset(new ComputeShader(m_device, rle_encode_write_out_hlsl::cfg));

        // allocate intermediate!
        m_tmp_d.reset(new ComputeBuffer(m_device, m_n_els * sizeof(int32_t)));
        m_tmp_i.reset(new ComputeBuffer(m_device, (m_n_els + 1) * sizeof(int32_t)));

        m_pfx_sum.reset(new PrefixSum(m_device, m_n_els + 1));
    }

    int numInputs() override { return 1; }
    int numOutputs() override { return 3; }

    void run(ComPtr<ID3D12GraphicsCommandList> commandList,
            IMLOperatorKernelContext* context,
            std::vector<ComPtr<ID3D12Resource>>& input_resources,
            std::vector<std::vector<uint32_t>>& input_dims,
            std::vector<ComPtr<ID3D12Resource>>& output_resources,
            std::vector<std::vector<uint32_t>>& output_dims) override {

        auto x = input_resources[0];
        auto enc_n = output_resources[0];
        auto enc_d = output_resources[1];
        auto enc_i = output_resources[2];

        auto tmp_i = m_tmp_i->resource();
        auto tmp_d = m_tmp_d->resource();

        const auto block_size = 256; // matches shader!
        const auto grid_size = (m_n_els + block_size - 1) / block_size;
        rle_encode_get_diffs_hlsl::constants constants{m_n_els};
        m_shader_get_diffs->run(commandList, { grid_size }, { x, enc_i, tmp_d }, &constants);

        m_pfx_sum->run(commandList, tmp_i, enc_i);

        // write n out!
        bfx_ops::copyBufferRegion(commandList, enc_n, 0, tmp_i, m_n_els * sizeof(int32_t), sizeof(int32_t));

        rle_encode_write_out_hlsl::constants write_out_constants{m_n_els};
        m_shader_write_out->run(commandList, { grid_size }, { tmp_i, enc_i, tmp_d, enc_d }, &write_out_constants);
    }
};

} // bfx_ops
