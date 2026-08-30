#pragma once

#include "../../../../OperatorAuthorHelper/OperatorHelper.h"
#include "../../MLOperatorAuthorImpl.h"

#include "../../External/D3DX12/d3dx12.h"
#include <d3d12.h>

#include <wrl/client.h>
#include <wrl/implements.h>

#include <algorithm>
#include <sstream>

#include "bfx_ops.h"
#include "d3d_util.h"
#include "roto_common.h"

using namespace Microsoft::WRL;

namespace bfx_ops {

// The shader headers are produced using "GenerateShaders.bat"
#include "roto_occ_grad_hlsl.h"
#include "roto_occ_grad_reduce_hlsl.h"

// bfx::roto_OccGrad, DirectML/D3D12 implementation. TWO dispatches, not one - see
// roto_occ_grad_hlsl.h for why the absence of a float atomic in HLSL turns the CUDA kernel's
// single fused pass into scatter-to-private-rows plus a reduce.
//
// The op's identity is a CONTRACT with python/roto/refine_graph.py: the exporter registers an ONNX
// schema for (domain "bfx", op_type "roto_OccGrad", version 1), so this name has to match
// OCC_GRAD_OPTYPE exactly - and a matching schema has to be declared in
// onnxruntime/core/graph/dml_ops/dml_defs.cc or the model will not resolve at all.
//
// THE INPUT LIST IS THE UNION OF WHAT THE TWO MODES NEED, because an operator's input count is
// fixed and cannot vary by attribute. own mode reads no log1m/dldu; ext mode reads no
// own/present/tgt/wgt and none of the three weights. The caller is entitled to bind a small dummy
// for the dead half - which is the point, since `own` is (F,P) per object and the largest thing
// refine holds - so nothing below may inspect an input the selected mode does not read.
class roto_occ_grad : public custom_op2
{
public:
    const static inline char* op_name = "roto_OccGrad";

    static std::vector<std::vector<uint32_t>> infer_shapes(IMLOperatorShapeInferenceContext* ctx)
    {
        MLShapeInferenceContext ctx_(ctx);
        OperatorHelper::KernelInformationAdapter kernelInfo{ctx_};
        OperatorHelper::ShapeInformationAdapter shapeInfo{ctx_};

        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorDimensionCount(0) == 3,
                                "roto_OccGrad: expected a rank-3 curve (F,S,2)");

        // d_curve has the curve's shape; own_loss is rank 1, NOT rank 0 - the graph output it
        // feeds is bound by a host whose tensor shape is a vector<int>, and an empty one reads
        // back as {1}. Same reason ext_loss is shaped that way.
        return {shapeInfo.GetInputTensorShape(0), {1}};
    }

    explicit roto_occ_grad(IMLOperatorKernelCreationContext* context) : custom_op2(context)
    {
        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        m_ext = static_cast<int32_t>(kernelInfo.GetAttributes().GetAttribute<int64_t>("ext"));

        // NOTHING SHAPE-DEPENDENT MAY BE BUILT HERE - this runs again for every call whose input
        // shapes differ from the last. See roto::cachedShader. The partials temporary is
        // allocated per Compute, from the kernel context, for exactly this reason.
        m_shader = roto::cachedShader(m_device, roto_occ_grad_hlsl::cfg);
        m_reduce = roto::cachedShader(m_device, roto_occ_grad_reduce_hlsl::cfg);
    }

    int numInputs() override { return 13; }
    int numOutputs() override { return 2; }

    void run(ComPtr<ID3D12GraphicsCommandList> commandList,
             IMLOperatorKernelContext* context,
             std::vector<ComPtr<ID3D12Resource>>& input_resources,
             std::vector<std::vector<uint32_t>>& input_dims,
             std::vector<ComPtr<ID3D12Resource>>& output_resources,
             std::vector<std::vector<uint32_t>>& output_dims) override
    {
        auto curve      = input_resources[0];
        auto ptsAll     = input_resources[1];
        auto own        = input_resources[2];
        auto present    = input_resources[3];
        auto tgtAll     = input_resources[4];
        auto wgtAll     = input_resources[5];
        auto log1mAll   = input_resources[6];
        auto dlduAll    = input_resources[7];
        auto posIdx     = input_resources[8];
        auto sharpness  = input_resources[9];
        auto neighborWt = input_resources[10];
        auto ownWt      = input_resources[11];
        auto ownTotal   = input_resources[12];

        auto dCurve  = output_resources[0];
        auto ownLoss = output_resources[1];

        const auto& curveShape = input_dims[0];   // (F, S, 2)
        const auto& ptsShape   = input_dims[1];   // (Q, P, 2)
        const auto& idxShape   = input_dims[8];   // (F)

        ML_CHECK_VALID_ARGUMENT(curveShape.size() == 3 && curveShape[2] == 2,
                                "roto_OccGrad: expected a rank-3 curve (F,S,2)");
        ML_CHECK_VALID_ARGUMENT(ptsShape.size() == 3 && ptsShape[2] == 2,
                                "roto_OccGrad: expected rank-3 points (Q,P,2)");
        ML_CHECK_VALID_ARGUMENT(idxShape.size() == 1 && idxShape[0] == curveShape[0],
                                "roto_OccGrad: pos_idx must be (F), matching the curve");

        const uint32_t F = curveShape[0];
        const uint32_t S = curveShape[1];
        const uint32_t P = ptsShape[1];

        // `own` is only checked in own mode, because only own mode reads it - in ext mode it is a
        // dummy binding and asserting a shape on it would reject the very saving it exists for.
        if (m_ext == 0)
        {
            const auto& ownShape = input_dims[2];  // (F, P)
            ML_CHECK_VALID_ARGUMENT(ownShape.size() == 2 && ownShape[0] == F,
                                    "roto_OccGrad: own must be (F,P), matching the curve");
            // every (Q,P) stack is indexed by pos_idx rather than pre-sliced, so P has to agree
            // across them - a mismatch would read a neighbouring frame's samples and still look
            // plausible
            ML_CHECK_VALID_ARGUMENT(ownShape[1] == P,
                                    "roto_OccGrad: own (F,P) disagrees with points on P");
        }

        const uint32_t slots = S * 2;
        const uint32_t rowStride = slots + 1;      // + this block's own-loss partial

        // blocksX is CAPPED, which is what bounds the temporary and lets the scatter avoid global
        // atomics entirely - see roto_occ_grad_hlsl.h. At least 1 even when P is 0, so that the
        // shader still runs, writes its zeroed slab out, and the reduce produces the zeros that
        // ARE the whole answer at P == 0.
        const uint32_t blocksX = (F == 0 || S == 0)
            ? 0
            : std::max(1u, std::min<uint32_t>(roto::gridFor(P, ROTO_OG_BLOCK),
                                              ROTO_OG_MAX_BLOCKS_X));

        // The temporary lives for this Compute only; the context frees it. 4 bytes minimum
        // because a zero-size allocation is not a thing, and because the F == 0 path below still
        // has to bind something.
        const uint64_t partialFloats =
            static_cast<uint64_t>(blocksX) * F * rowStride;
        const size_t partialBytes =
            static_cast<size_t>(std::max<uint64_t>(partialFloats, 1ull) * sizeof(float));

        ComPtr<IUnknown> tempUnknown;
        ORT_THROW_IF_FAILED(context->AllocateTemporaryData(partialBytes, tempUnknown.GetAddressOf()));
        ComPtr<ID3D12Resource> partials;
        ORT_THROW_IF_FAILED(tempUnknown.As(&partials));

        if (blocksX > 0)
        {
            const uint32_t gridX = blocksX;
            roto::checkDispatchGrid(gridX, F, op_name);

            const int useSlab = (slots <= ROTO_OG_MAX_SLAB_FLOATS) ? 1 : 0;

            roto_occ_grad_hlsl::constants c{
                static_cast<int>(F),
                static_cast<int>(S),
                static_cast<int>(P),
                m_ext,
                static_cast<int>(blocksX),
                useSlab};

            m_shader->run(
                commandList,
                {gridX, F},
                {curve, ptsAll, own, present, tgtAll, wgtAll, log1mAll, dlduAll, posIdx,
                 sharpness, neighborWt, ownWt, ownTotal, partials},
                &c);
        }

        // The reduce runs UNCONDITIONALLY, including the F == 0 / S == 0 case where blocksX is 0.
        // Both outputs are WRITTEN rather than accumulated, and ORT hands back dirty buffers - so
        // skipping this would leave own_loss holding whatever the last object put in that
        // allocation. With blocksX == 0 the sums are empty and it writes the zeros directly, which
        // is what roto_occ_grad_cu's "zero ownLoss ahead of every early-out" comment is about.
        roto_occ_grad_reduce_hlsl::constants rc{
            static_cast<int>(F),
            static_cast<int>(S),
            static_cast<int>(blocksX)};

        const uint32_t reduceOutputs = F * slots + 1;   // + own_loss
        const uint32_t reduceGrid = roto::gridFor(reduceOutputs, ROTO_OGR_BLOCK);
        roto::checkDispatchGrid(reduceGrid, 1, op_name);

        m_reduce->run(
            commandList,
            {reduceGrid},
            {partials, dCurve, ownLoss, ownWt, ownTotal},
            &rc);
    }

private:
    // WHICH FIT TERM, as an attribute rather than an input: the two modes are exported as different
    // files anyway (own mode never runs pass 1), so it is known at export time, and knowing it here
    // lets the shader skip the other mode's loads rather than multiply by a zero coefficient.
    int32_t m_ext = 0;

    std::shared_ptr<ComputeShader> m_shader;
    std::shared_ptr<ComputeShader> m_reduce;
};

} // namespace bfx_ops
