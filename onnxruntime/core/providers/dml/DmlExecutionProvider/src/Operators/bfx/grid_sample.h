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
namespace grid_sample {

#include "grid_sample_shader_constants.h"

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
namespace shader_fp32 {
    #include "GeneratedShaders/grid_sample_fp32.h"
}

namespace shader_fp16 {
    #include "GeneratedShaders/grid_sample_fp16.h"
}

class op
{
public:

    const static inline char* op_name = "grid_sample";

    // input, grid | output
    const static inline int32_t num_inputs = 2;
    const static inline int32_t num_outputs = 1;

    // number of 32-bit values copied to shader as 'uniforms' (to use the OpenGL term..)
    const static inline int32_t num_els_constants = num_els_constants;

    const static inline void* bytecode_fp32 = shader_fp32::g_grid_sample;
    const static inline size_t bytecode_fp32_size = sizeof(shader_fp32::g_grid_sample);
    const static inline void* bytecode_fp16 = shader_fp16::g_grid_sample;
    const static inline size_t bytecode_fp16_size = sizeof(shader_fp16::g_grid_sample);

    // Parses kernel & shape info to generate an execution configuration for the kernel
    class op_params
    {
    public:
        // attributes
        int32_t interpolation_mode;
        int32_t padding_mode;
        int32_t align_corners;

        // shapes, etc..
        int32_t n = 0; // batch size.
        int32_t n_ch = 0;
        int32_t in_h = 0;
        int32_t in_w = 0;
        int32_t out_h = 0;
        int32_t out_w = 0;

        op_params(){}

        op_params(
                const OperatorHelper::IKernelInformationAdapter& kernelInfo,
                const OperatorHelper::IShapeInformationAdapter& shapeInfo)
        {
            auto& attributes = kernelInfo.GetAttributes();

            interpolation_mode = static_cast<int32_t>(attributes.GetAttribute<int64_t>("interpolation_mode"));
            padding_mode = static_cast<int32_t>(attributes.GetAttribute<int64_t>("padding_mode"));
            align_corners = static_cast<int32_t>(attributes.GetAttribute<int64_t>("align_corners"));

            // input 0: image (required; tensor)
            {
                uint32_t rank = shapeInfo.GetInputTensorDimensionCount(0);
                ML_CHECK_VALID_ARGUMENT(rank == 4, "Input shape must be 4D.");
                auto dims = shapeInfo.GetInputTensorShape(0);
                assert(dims.size() == rank);
                n = static_cast<int32_t>(dims[0]);
                n_ch = static_cast<int32_t>(dims[1]);
                in_h = static_cast<int32_t>(dims[2]);
                in_w = static_cast<int32_t>(dims[3]);
            }

            // input 1: grid
            {
                uint32_t rank = shapeInfo.GetInputTensorDimensionCount(1);
                ML_CHECK_VALID_ARGUMENT(rank == 4, "Offset shape must be 4D.");
                auto dims = shapeInfo.GetInputTensorShape(1);
                assert(dims.size() == rank);
                assert(n == dims[0]);
                assert(2 == dims[1]);
                out_h = static_cast<int32_t>(dims[2]);
                out_w = static_cast<int32_t>(dims[3]);
            }
        }

        // copies runtime config to necessary shader constants!
        shader_constants make_shader_constants() {
            shader_constants constants = {};
            constants.interpolation_mode = interpolation_mode;
            constants.padding_mode = padding_mode;
            constants.align_corners = align_corners;
            constants.n = n;
            constants.n_ch = n_ch;
            constants.in_h = in_h;
            constants.in_w = in_w;
            constants.out_h = out_h;
            constants.out_w = out_w;

            return constants;
        }

    };

    static std::vector<std::vector<uint32_t>> infer_shapes(op_params& params) {
        return {{
            (uint32_t)(params.n),
            (uint32_t)(params.n_ch),
            (uint32_t)(params.out_h),
            (uint32_t)(params.out_w)
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

        const uint32_t gridShapeX = (params.out_w + blockShapeX - 1) / blockShapeX;
        const uint32_t gridShapeY = (params.out_h + blockShapeY - 1) / blockShapeY;
        const uint32_t gridShapeC = params.n * params.n_ch;

        dispatch(gridShapeX, gridShapeY, gridShapeC);
    }
};

} // grid_sample
} // bfx_ops
