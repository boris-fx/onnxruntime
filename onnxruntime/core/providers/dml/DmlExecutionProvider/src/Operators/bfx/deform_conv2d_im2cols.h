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
namespace deform_conv2d_im2cols {

#include "deform_conv2d_im2cols_shader_constants.h"

// The shader headers are produced using "GeneratedShaders/GenerateShaders.bat"
namespace shader_fp32 {
    #include "GeneratedShaders/deform_conv2d_im2cols_fp32.h"
}

namespace shader_fp16 {
    #include "GeneratedShaders/deform_conv2d_im2cols_fp16.h"
}

class op
{
public:

    const static inline char* op_name = "deform_conv2d_im2cols";

    // input, offset, mask | output
    const static inline int32_t num_inputs = 3;
    const static inline int32_t num_outputs = 1;

    // number of 32-bit values copied to shader as 'uniforms' (to use the OpenGL term..)
    const static inline int32_t num_els_constants = num_els_constants;

    const static inline void* bytecode_fp32 = shader_fp32::g_deform_conv2d_im2cols;
    const static inline size_t bytecode_fp32_size = sizeof(shader_fp32::g_deform_conv2d_im2cols);
    const static inline void* bytecode_fp16 = shader_fp16::g_deform_conv2d_im2cols;
    const static inline size_t bytecode_fp16_size = sizeof(shader_fp16::g_deform_conv2d_im2cols);

    // Parses kernel & shape info to generate an execution configuration for the kernel
    class op_params
    {
    public:
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

        op_params(){}

        op_params(
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
            }
        }

        // copies runtime config to necessary shader constants!
        shader_constants make_shader_constants() {
            shader_constants constants = {};
            constants.n = n;
            constants.n_in_channels = in_ch;
            constants.in_h = in_h;
            constants.in_w = in_w;
            constants.out_h = out_h;
            constants.out_w = out_w;
            constants.kernel_h = kernel_h;
            constants.kernel_w = kernel_w;
            constants.pad_h = pad_h;
            constants.pad_w = pad_w;
            constants.dil_h = dil_h;
            constants.dil_w = dil_w;
            constants.stride_h = stride_h;
            constants.stride_w = stride_w;
            constants.n_offset_grps = n_offset_grps;
            constants.use_mask = use_mask;

            constants.n_kernels =
                constants.n * constants.n_in_channels * constants.out_h * constants.out_w;

            return constants;
        }

    };

    static std::vector<std::vector<uint32_t>> infer_shapes(op_params& params) {
        return {{
            (uint32_t)(params.n),
            (uint32_t)(params.in_ch * params.kernel_h * params.kernel_w),
            (uint32_t)(params.out_h),
            (uint32_t)(params.out_w)
        }};
    }

    static void run(op_params& params, std::function<void(void*)> set_constants, std::function<void(uint32_t, uint32_t, uint32_t)> dispatch) {

        auto constants = params.make_shader_constants();
        set_constants(&constants);

        const uint32_t numThreads = 256; // matches value in hlsl kernel definition!

        const uint32_t numGroups = (constants.n_kernels + numThreads - 1) / numThreads;

        if (numGroups > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION) {
            printf("ERROR! TODO: handle deform_conv2d_im2cols invocation larger than max D3D12 thread group size\n");
        }

        dispatch(numGroups, 1, 1);
    }
};

} // deform_conv2d_im2cols
} // bfx_ops
