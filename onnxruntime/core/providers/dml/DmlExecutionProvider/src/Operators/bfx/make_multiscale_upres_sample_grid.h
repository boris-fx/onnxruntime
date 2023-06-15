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
namespace make_multiscale_upres_sample_grid {

#include "make_multiscale_upres_sample_grid_shader_constants.h"

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
namespace shader {
    #include "GeneratedShaders/make_multiscale_upres_sample_grid.h"
}

class op
{
public:

    const static inline char* op_name = "make_multiscale_upres_sample_grid";

    // exec_config | output
    const static inline int32_t num_inputs = 1;
    const static inline int32_t num_outputs = 1;

    // number of 32-bit values copied to shader as 'uniforms' (to use the OpenGL term..)
    const static inline int32_t num_els_constants = num_els_constants;

    const static inline void* bytecode_fp32 = shader::g_make_multiscale_upres_sample_grid;
    const static inline size_t bytecode_fp32_size = sizeof(shader::g_make_multiscale_upres_sample_grid);
    const static inline void* bytecode_fp16 = nullptr;
    const static inline size_t bytecode_fp16_size = 0;

    // Parses kernel & shape info to generate an execution configuration for the kernel
    class op_params
    {
    public:

        // shapes, etc..
        int32_t n = 0; // batch size.
        int32_t tile_height = 0;
        int32_t tile_width = 0;

        op_params(){}

        op_params(
                const OperatorHelper::IKernelInformationAdapter& kernelInfo,
                const OperatorHelper::IShapeInformationAdapter& shapeInfo)
        {
            auto& attributes = kernelInfo.GetAttributes();

            n = static_cast<int32_t>(attributes.GetAttribute<int64_t>("n"));
            tile_height = static_cast<int32_t>(attributes.GetAttribute<int64_t>("tile_height"));
            tile_width = static_cast<int32_t>(attributes.GetAttribute<int64_t>("tile_width"));

            // input 0: image (required; tensor)
            {
                uint32_t rank = shapeInfo.GetInputTensorDimensionCount(0);
                ML_CHECK_VALID_ARGUMENT(rank == 4, "Input shape must be 4D.");
                auto dims = shapeInfo.GetInputTensorShape(0);
                assert(dims.size() == rank);
                assert(1 == dims[0]);
                assert(5 == dims[1]);
                assert(1 == dims[2]);
                assert(1 == dims[3]);
            }
        }

        // copies runtime config to necessary shader constants!
        shader_constants make_shader_constants() {
            shader_constants constants = {};
            constants.n = n;
            constants.tile_width = tile_width;
            constants.tile_height = tile_height;

            return constants;
        }

    };

    static std::vector<std::vector<uint32_t>> infer_shapes(op_params& params) {
        return {{
            (uint32_t)(params.n),
            (uint32_t)(2),
            (uint32_t)(params.tile_height),
            (uint32_t)(params.tile_width)
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

        const uint32_t gridShapeX = (params.tile_width + blockShapeX - 1) / blockShapeX;
        const uint32_t gridShapeY = (params.tile_height + blockShapeY - 1) / blockShapeY;

        dispatch(gridShapeX, gridShapeY, 1);
    }
};

} // make_multiscale_upres_sample_grid
} // bfx_ops
