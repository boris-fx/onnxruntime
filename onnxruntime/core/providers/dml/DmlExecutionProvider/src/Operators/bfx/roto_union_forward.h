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
#include "roto_common.h"

using namespace Microsoft::WRL;

namespace bfx_ops {

// The shader header is produced using "GenerateShaders.bat"
#include "roto_union_forward_hlsl.h"

// bfx::roto_UnionForward, DirectML/D3D12 implementation.
//
// The op's identity is a CONTRACT with python/roto/refine_graph.py: the exporter registers an ONNX
// schema for (domain "bfx", op_type "roto_UnionForward", version 1) and emits the node against it,
// so this name has to match UNION_FORWARD_OPTYPE exactly - and the matching schema has to be
// declared in onnxruntime/core/graph/dml_ops/dml_defs.cc, or the model will not even resolve.
//
// custom_op2 rather than the templated custom_op<>: custom_op<> binds one PSO to exactly
// num_inputs + num_outputs UAVs and dispatches once with the input-0 element type selecting fp32
// or fp16 bytecode. This op is float32 throughout (see the schema), takes an int64 input which
// that switch has no case for, and wants its grid computed from two different tensors' shapes.
//
// No attributes: unlike roto_OccGrad there is no joint/own split here, because pass 1 does not run
// at all when there is no external target.
class roto_union_forward : public custom_op2
{
public:
    const static inline char* op_name = "roto_UnionForward";

    static std::vector<std::vector<uint32_t>> infer_shapes(IMLOperatorShapeInferenceContext* ctx)
    {
        MLShapeInferenceContext ctx_(ctx);
        OperatorHelper::KernelInformationAdapter kernelInfo{ctx_};
        OperatorHelper::ShapeInformationAdapter shapeInfo{ctx_};

        // log1m_out has the accumulator's shape exactly - (Q,P) in, (Q,P) out. Rows this object
        // does not cover are copied through rather than dropped, which is why the output is never
        // smaller than the input.
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorDimensionCount(0) == 2,
                                "roto_UnionForward: expected a rank-2 accumulator (Q,P)");
        return {shapeInfo.GetInputTensorShape(0)};
    }

    explicit roto_union_forward(IMLOperatorKernelCreationContext* context) : custom_op2(context)
    {
        // NOTHING SHAPE-DEPENDENT MAY BE BUILT HERE. This constructor runs again for every call
        // whose input shapes differ from the last - see roto::cachedShader for why - so it does a
        // hash lookup and nothing else.
        m_shader = roto::cachedShader(m_device, roto_union_forward_hlsl::cfg);
    }

    int numInputs() override { return 5; }
    int numOutputs() override { return 1; }

    void run(ComPtr<ID3D12GraphicsCommandList> commandList,
             IMLOperatorKernelContext* context,
             std::vector<ComPtr<ID3D12Resource>>& input_resources,
             std::vector<std::vector<uint32_t>>& input_dims,
             std::vector<ComPtr<ID3D12Resource>>& output_resources,
             std::vector<std::vector<uint32_t>>& output_dims) override
    {
        auto log1mIn   = input_resources[0];
        auto curve     = input_resources[1];
        auto ptsAll    = input_resources[2];
        auto posIdx    = input_resources[3];
        auto sharpness = input_resources[4];
        auto log1mOut  = output_resources[0];

        const auto& accShape   = input_dims[0];   // (Q, P)
        const auto& curveShape = input_dims[1];   // (F, S, 2)
        const auto& ptsShape   = input_dims[2];   // (Q, P, 2)
        const auto& idxShape   = input_dims[3];   // (F)

        ML_CHECK_VALID_ARGUMENT(accShape.size() == 2,
                                "roto_UnionForward: expected a rank-2 accumulator (Q,P)");
        ML_CHECK_VALID_ARGUMENT(curveShape.size() == 3 && curveShape[2] == 2,
                                "roto_UnionForward: expected a rank-3 curve (F,S,2)");
        ML_CHECK_VALID_ARGUMENT(ptsShape.size() == 3 && ptsShape[2] == 2,
                                "roto_UnionForward: expected rank-3 points (Q,P,2)");
        // pts is indexed by the OUTPUT row rather than pre-gathered, so its Q and P must be the
        // accumulator's - a mismatch would silently read a different frame's samples
        ML_CHECK_VALID_ARGUMENT(ptsShape[0] == accShape[0] && ptsShape[1] == accShape[1],
                                "roto_UnionForward: points (Q,P,2) disagree with the accumulator");
        ML_CHECK_VALID_ARGUMENT(idxShape.size() == 1 && idxShape[0] == curveShape[0],
                                "roto_UnionForward: pos_idx must be (F), matching the curve");

        const uint32_t Q = accShape[0];
        const uint32_t P = accShape[1];
        const uint32_t F = curveShape[0];
        const uint32_t S = curveShape[1];

        if (Q == 0 || P == 0)
        {
            return;
        }

        // F or S being zero means an object with nothing to render. The accumulator still has to
        // come out intact, so that is a COPY rather than an early return - anything else would
        // drop every other object's contribution. Mirrors roto_union_forward_cu's cudaMemcpyAsync.
        if (F == 0 || S == 0)
        {
            copyBufferRegion(commandList, log1mOut, 0, log1mIn, 0,
                             static_cast<size_t>(Q) * P * sizeof(float));
            return;
        }

        const uint32_t gridX = roto::gridFor(P, ROTO_UF_BLOCK);
        roto::checkDispatchGrid(gridX, Q, op_name);

        roto_union_forward_hlsl::constants c{
            static_cast<int>(F),
            static_cast<int>(S),
            static_cast<int>(P)};

        m_shader->run(
            commandList,
            {gridX, Q},
            {log1mIn, curve, ptsAll, posIdx, sharpness, log1mOut},
            &c);
    }

private:
    std::shared_ptr<ComputeShader> m_shader;
};

} // namespace bfx_ops
