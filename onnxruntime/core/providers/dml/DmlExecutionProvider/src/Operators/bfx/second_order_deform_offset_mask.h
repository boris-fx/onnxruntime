#pragma once

#include "../../../../OperatorAuthorHelper/OperatorHelper.h"
#include "../../MLOperatorAuthorImpl.h"

#include "../../External/D3DX12/d3dx12.h"
#include <d3d12.h>

#include <wrl/client.h>
#include <wrl/implements.h>

#include <sstream>

#include "bfx_ops.h"

using namespace Microsoft::WRL;

namespace bfx_ops {
namespace second_order_deform_offset_mask {

#include "second_order_deform_offset_mask_shader_constants.h"

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
namespace shader_fp32 {
    #include "GeneratedShaders/second_order_deform_offset_mask_fp32.h"
}

namespace shader_fp16 {
    #include "GeneratedShaders/second_order_deform_offset_mask_fp16.h"
}

class op
{
public:

    const static inline char* op_name = "second_order_deform_alignment_make_offset_and_mask";

    // input, offset, mask | output
    const static inline int32_t num_inputs = 3;
    const static inline int32_t num_outputs = 2;

    // number of 32-bit values copied to shader as 'uniforms' (to use the OpenGL term..)
    const static inline int32_t num_els_constants = num_els_constants;

    const static inline void* bytecode_fp32 = shader_fp32::g_second_order_deform_offset_mask;
    const static inline size_t bytecode_fp32_size = sizeof(shader_fp32::g_second_order_deform_offset_mask);
    const static inline void* bytecode_fp16 = shader_fp16::g_second_order_deform_offset_mask;
    const static inline size_t bytecode_fp16_size = sizeof(shader_fp16::g_second_order_deform_offset_mask);

    // Parses kernel & shape info to generate an execution configuration for the kernel
    class op_params
    {
    public:
        // attributes
        float max_residue_magnitude;

        // shapes, etc..
        int32_t n = 0; // batch size.
        int32_t n_ch_feats = 0;
        int32_t h = 0;
        int32_t w = 0;

        op_params(){}

        op_params(
                const OperatorHelper::IKernelInformationAdapter& kernelInfo,
                const OperatorHelper::IShapeInformationAdapter& shapeInfo)
        {
            auto& attributes = kernelInfo.GetAttributes();

            max_residue_magnitude = static_cast<float>(attributes.GetAttribute<float>("max_residue_magnitude"));

            // input 0: image (required; tensor)
            {
                uint32_t rank = shapeInfo.GetInputTensorDimensionCount(0);
                ML_CHECK_VALID_ARGUMENT(rank == 4, "Input shape must be 4D.");
                auto dims = shapeInfo.GetInputTensorShape(0);
                assert(dims.size() == rank);
                n = static_cast<int32_t>(dims[0]);
                n_ch_feats = static_cast<int32_t>(dims[1]);
                h = static_cast<int32_t>(dims[2]);
                w = static_cast<int32_t>(dims[3]);
            }

            // input 1: flow_1
            {
                uint32_t rank = shapeInfo.GetInputTensorDimensionCount(1);
                ML_CHECK_VALID_ARGUMENT(rank == 4, "Offset shape must be 4D.");
                auto dims = shapeInfo.GetInputTensorShape(1);
                assert(dims.size() == rank);
                assert(n == static_cast<int32_t>(dims[0]));
                assert(2 == static_cast<int32_t>(dims[1]));
                assert(h == static_cast<int32_t>(dims[2]));
                assert(w == static_cast<int32_t>(dims[3]));
            }

            // input 1: flow_2
            {
                uint32_t rank = shapeInfo.GetInputTensorDimensionCount(2);
                ML_CHECK_VALID_ARGUMENT(rank == 4, "Offset shape must be 4D.");
                auto dims = shapeInfo.GetInputTensorShape(1);
                assert(dims.size() == rank);
                assert(n == static_cast<int32_t>(dims[0]));
                assert(2 == static_cast<int32_t>(dims[1]));
                assert(h == static_cast<int32_t>(dims[2]));
                assert(w == static_cast<int32_t>(dims[3]));
            }
        }

        // copies runtime config to necessary shader constants!
        shader_constants make_shader_constants() {
            shader_constants constants = {};
            constants.max_residue_magnitude = max_residue_magnitude;
            constants.n = n;
            constants.n_ch_feats = n_ch_feats;
            constants.h = h;
            constants.w = w;

            return constants;
        }

    };

    static std::vector<std::vector<uint32_t>> infer_shapes(op_params& params) {
        return {{
            (uint32_t)(params.n),
            (uint32_t)(2 * params.n_ch_feats / 3),
            (uint32_t)(params.h),
            (uint32_t)(params.w)
        },{
            (uint32_t)(params.n),
            (uint32_t)(params.n_ch_feats / 3),
            (uint32_t)(params.h),
            (uint32_t)(params.w)
        }};
    }

    static void run(
            op_params& params,
            std::function<void(void*)> set_constants,
            std::function<void(uint32_t, uint32_t, uint32_t)> dispatch) {

        auto constants = params.make_shader_constants();
        set_constants(&constants);

        // matches value in hlsl kernel definition
        const uint32_t blockShapeX = 16;
        const uint32_t blockShapeY = 16;

        const uint32_t gridShapeX = (params.w + blockShapeX - 1) / blockShapeX;
        const uint32_t gridShapeY = (params.h + blockShapeY - 1) / blockShapeY;
        const uint32_t gridShapeC = params.n * params.n_ch_feats;

        dispatch(gridShapeX, gridShapeY, gridShapeC);
    }
};

} // second_order_deform_offset_mask
} // bfx_ops
