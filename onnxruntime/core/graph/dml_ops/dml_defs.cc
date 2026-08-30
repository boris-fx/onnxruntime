// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/graph/dml_ops/dml_defs.h"
#include "core/graph/op.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "core/providers/dml/OperatorAuthorHelper/Attributes.h"

namespace ONNX_NAMESPACE {
void convPoolShapeInference(
    ONNX_NAMESPACE::InferenceContext& ctx,
    bool use_dilation, bool require_kernel_shape,
    int input1Idx,
    int input2Idx);
void convTransposeShapeInference(InferenceContext& ctx);
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
namespace dml {
using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;
using ONNX_NAMESPACE::OPTIONAL_VALUE;

void RegisterDmlSchemas() {
  MS_DML_OPERATOR_SCHEMA(DmlFusedConv)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused Conv+Activation)DOC")
      .Input(0, "X", "", "T")
      .Input(1, "W", "", "T")
      .Input(2, "B", "", "T", OpSchema::Optional)
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        ONNX_NAMESPACE::convPoolShapeInference(ctx, true, false, 0, 1);
      });

  MS_DML_OPERATOR_SCHEMA(DmlFusedConvTranspose)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused ConvTranspose+Activation)DOC")
      .Input(0, "X", "", "T")
      .Input(1, "W", "", "T")
      .Input(2, "B", "", "T", OpSchema::Optional)
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr("kernel_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("output_shape", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("output_padding", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("dilations", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("strides", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("auto_pad", "", AttributeProto::STRING, std::string("NOTSET"))
      .Attr("pads", "", AttributeProto::INTS, OPTIONAL_VALUE)
      .Attr("group", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction(
          [](ONNX_NAMESPACE::InferenceContext& ctx) { ONNX_NAMESPACE::convTransposeShapeInference(ctx); });

  MS_DML_OPERATOR_SCHEMA(DmlFusedInstanceNormalization)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused InstanceNormalization+Activation)DOC")
      .Attr("epsilon", "", AttributeProto::FLOAT, 1e-5f)
      .Input(0, "input", "", "T")
      .Input(1, "scale", "", "T")
      .Input(2, "B", "", "T")
      .Output(0, "output", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
      });

  MS_DML_OPERATOR_SCHEMA(DmlFusedBatchNormalization)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused BatchNormalization+Activation)DOC")
      .NumOutputs({1, 5})
      .Attr("spatial", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Attr("epsilon", "", AttributeProto::FLOAT, 1e-5f)
      .Attr("momentum", "", AttributeProto::FLOAT, 0.9f)
      .Attr("training_mode", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Input(0, "X", "", "T")
      .Input(1, "scale", "", "T")
      .Input(2, "B", "", "T")
      .Input(3, "mean", "", "T")
      .Input(4, "var", "", "T")
      .Output(0, "Y", "", "T")
      .Output(1, "mean", "", "T", OpSchema::Optional)
      .Output(2, "var", "", "T", OpSchema::Optional)
      .Output(3, "saved_mean", "", "T", OpSchema::Optional)
      .Output(4, "saved_var", "", "T", OpSchema::Optional)
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
        // TODO in training mode, it may be possible to infer some of
        // the other outputs as well.
      });

  MS_DML_OPERATOR_SCHEMA(DmlFusedMeanVarianceNormalization)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused MeanVarianceNormalization+Activation)DOC")
      .Attr("across_channels", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("normalize_variance", "", AttributeProto::INT, static_cast<int64_t>(1))
      .Input(0, "input", "", "T")
      .Output(0, "output", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput);

  MS_DML_OPERATOR_SCHEMA(DmlFusedGemm)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused Gemm+Activation)DOC")
      .Input(0, "A", "", "T")
      .Input(1, "B", "", "T")
      .Input(2, "C", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr("transA", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("transB", "", AttributeProto::INT, static_cast<int64_t>(0))
      .Attr("alpha", "", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "", AttributeProto::FLOAT, 1.0f)
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (hasNInputShapes(ctx, 2)) {
          auto transAAttr = ctx.getAttribute("transA");
          bool transA =
              transAAttr ? static_cast<int>(transAAttr->i()) != 0 : false;
          auto transBAttr = ctx.getAttribute("transB");
          bool transB =
              transBAttr ? static_cast<int>(transBAttr->i()) != 0 : false;
          auto& first_input_shape = getInputShape(ctx, 0);
          auto& second_input_shape = getInputShape(ctx, 1);
          if (first_input_shape.dim_size() != 2)
            fail_shape_inference("First input does not have rank 2");
          if (second_input_shape.dim_size() != 2)
            fail_shape_inference("Second input does not have rank 2");
          updateOutputShape(
              ctx,
              0,
              {first_input_shape.dim(transA ? 1 : 0),
               second_input_shape.dim(transB ? 0 : 1)});
        }
      });

  MS_DML_OPERATOR_SCHEMA(DmlFusedMatMul)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused MatMul+Activation)DOC")
      .Input(0, "A", "", "T")
      .Input(1, "B", "", "T")
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (!hasNInputShapes(ctx, 2)) {
          return;
        }

        const auto shape0 = ctx.getInputType(0)->tensor_type().shape();
        const auto shape1 = ctx.getInputType(1)->tensor_type().shape();

        if (shape0.dim_size() == 0 || shape1.dim_size() == 0) {
          fail_shape_inference("Input tensors of wrong rank (0).");
        }

        ONNX_NAMESPACE::TensorShapeProto shapeL, shapeR;

        // First promote each shape to at least rank-2. This logic is
        // specific to matmul, not generic broadcasting.
        {
          if (shape0.dim_size() == 1) {
            shapeL.add_dim()->set_dim_value(1);
            *shapeL.add_dim() = shape0.dim(0);
          } else {
            *shapeL.mutable_dim() = shape0.dim();
          }
          if (shape1.dim_size() == 1) {
            *shapeR.add_dim() = shape1.dim(0);
            shapeR.add_dim()->set_dim_value(1);
          } else {
            *shapeR.mutable_dim() = shape1.dim();
          }
        }

        // Check for compatible matrix multiply dimensions
        {
          auto dimL = shapeL.dim(shapeL.dim_size() - 1);
          auto dimR = shapeR.dim(shapeR.dim_size() - 2);
          if (dimL.has_dim_value() && dimR.has_dim_value() &&
              dimL.dim_value() != dimR.dim_value()) {
            fail_shape_inference(
                "Incompatible dimensions for matrix multiplication");
            ;
          }
        }

        ONNX_NAMESPACE::TensorShapeProto resultShape;

        // Now call out to generic multidimensional broadcasting for
        // the broadcastable prefixes.
        {
          ONNX_NAMESPACE::TensorShapeProto prefixShapeL, prefixShapeR;
          for (int i = 0; i < shapeL.dim_size() - 2; ++i) {
            *prefixShapeL.add_dim() = shapeL.dim(i);
          }
          for (int i = 0; i < shapeR.dim_size() - 2; ++i) {
            *prefixShapeR.add_dim() = shapeR.dim(i);
          }
          bidirectionalBroadcastShapeInference(
              prefixShapeL, prefixShapeR, resultShape);
        }

        // Back to matmul-specific. Add the trailing dimensions back in.
        {
          if (shape0.dim_size() != 1) {
            *resultShape.add_dim() = shapeL.dim(shapeL.dim_size() - 2);
          }
          if (shape1.dim_size() != 1) {
            *resultShape.add_dim() = shapeR.dim(shapeR.dim_size() - 1);
          }
        }

        *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape() =
            resultShape;
      });

  MS_DML_OPERATOR_SCHEMA(DmlFusedAdd)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused Add+Activation)DOC")
      .Input(0, "A", "", "T")
      .Input(1, "B", "", "T")
      .Output(0, "C", "", "T")
      .TypeConstraint("T", OpSchema::numeric_types_for_math_reduction(), "")
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        if (hasNInputShapes(ctx, 2))
          bidirectionalBroadcastShapeInference(
              ctx.getInputType(0)->tensor_type().shape(),
              ctx.getInputType(1)->tensor_type().shape(),
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
      });

  MS_DML_OPERATOR_SCHEMA(DmlFusedSum)
      .SetDomain(kMSDmlDomain)
      .SinceVersion(1)
      .SetDoc(R"DOC(DirectML fused Sum+Activation)DOC")
      .Input(0, "data_0", "", "T", OpSchema::Variadic)
      .Output(0, "sum", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)", "tensor(double)"}, "")
      .Attr(AttrName::FusedActivation, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationDomain, "", onnx::AttributeProto::STRING)
      .Attr(AttrName::FusedActivationSinceVersion, "", onnx::AttributeProto::INT)
      .Attr(AttrName::FusedAlpha, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedBeta, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedGamma, "", onnx::AttributeProto::FLOAT, false)
      .Attr(AttrName::FusedRatio, "", onnx::AttributeProto::FLOAT, false)
      .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
        ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
        int num_inputs = static_cast<int>(ctx.getNumInputs());
        std::vector<const ONNX_NAMESPACE::TensorShapeProto*> shapes;
        for (int i = 0; i < num_inputs; ++i) {
          auto input_type = ctx.getInputType(i);
          if (nullptr == input_type || !input_type->has_tensor_type() ||
              !input_type->tensor_type().has_shape()) {
            return;
          }
          shapes.push_back(&input_type->tensor_type().shape());
        }

        multidirectionalBroadcastShapeInference(
            shapes,
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
      });


  // BFX CUSTOM OPS

  MS_DML_OPERATOR_SCHEMA(deform_conv2d_im2cols)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "input", "", "T")
      .Input(1, "offset", "", "T")
      .Input(2, "mask", "", "T")
      .Output(0, "output", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "")
      .Attr("dil_h", "", onnx::AttributeProto::INT)
      .Attr("dil_w", "", onnx::AttributeProto::INT)
      .Attr("kernel_h", "", onnx::AttributeProto::INT)
      .Attr("kernel_w", "", onnx::AttributeProto::INT)
      .Attr("pad_h", "", onnx::AttributeProto::INT)
      .Attr("pad_w", "", onnx::AttributeProto::INT)
      .Attr("stride_h", "", onnx::AttributeProto::INT)
      .Attr("stride_w", "", onnx::AttributeProto::INT)
      .Attr("n_offset_grps", "", onnx::AttributeProto::INT)
      .Attr("use_mask", "", onnx::AttributeProto::INT)
      .Attr("plugin_namespace", "", onnx::AttributeProto::STRING)
      .Attr("plugin_version", "", onnx::AttributeProto::STRING);

  MS_DML_OPERATOR_SCHEMA(warp_flow)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "input", "", "T")
      .Input(1, "flow", "", "T")
      .Output(0, "output", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "")
      .Attr("interpolation_mode", "", onnx::AttributeProto::INT)
      .Attr("padding_mode", "", onnx::AttributeProto::INT)
      .Attr("align_corners", "", onnx::AttributeProto::INT)
      .Attr("plugin_namespace", "", onnx::AttributeProto::STRING)
      .Attr("plugin_version", "", onnx::AttributeProto::STRING);

  MS_DML_OPERATOR_SCHEMA(second_order_deform_alignment_make_offset_and_mask)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "feats", "", "T")
      .Input(1, "flow_1", "", "T")
      .Input(2, "flow_2", "", "T")
      .Output(0, "out_offset", "", "T")
      .Output(1, "out_mask", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "")
      .Attr("max_residue_magnitude", "", onnx::AttributeProto::FLOAT)
      .Attr("plugin_namespace", "", onnx::AttributeProto::STRING)
      .Attr("plugin_version", "", onnx::AttributeProto::STRING);

  MS_DML_OPERATOR_SCHEMA(grid_sample)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "image", "", "T")
      .Input(1, "grid", "", "tensor(float)") // grid must be full float!
      .Output(0, "out", "", "T")
      .TypeConstraint("T", {"tensor(float16)", "tensor(float)"}, "")
      .Attr("interpolation_mode", "", onnx::AttributeProto::INT)
      .Attr("padding_mode", "", onnx::AttributeProto::INT)
      .Attr("align_corners", "", onnx::AttributeProto::INT)
      .Attr("plugin_namespace", "", onnx::AttributeProto::STRING)
      .Attr("plugin_version", "", onnx::AttributeProto::STRING);

  MS_DML_OPERATOR_SCHEMA(make_multiscale_upres_sample_grid)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "exec_config", "", "tensor(float)") // config and output must be full float
      .Output(0, "out", "", "tensor(float)")
      .Attr("n", "", onnx::AttributeProto::INT)
      .Attr("tile_height", "", onnx::AttributeProto::INT)
      .Attr("tile_width", "", onnx::AttributeProto::INT)
      .Attr("plugin_namespace", "", onnx::AttributeProto::STRING)
      .Attr("plugin_version", "", onnx::AttributeProto::STRING);

  MS_DML_OPERATOR_SCHEMA(rle_encode)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "x", "", "tensor(int32)")
      .Output(0, "enc_n", "", "tensor(int32)")
      .Output(1, "enc_d", "", "tensor(int32)")
      .Output(2, "enc_i", "", "tensor(int32)");

  MS_DML_OPERATOR_SCHEMA(rle_decode)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "enc_n", "", "tensor(int32)")
      .Input(1, "enc_d", "", "tensor(int32)")
      .Input(2, "enc_i", "", "tensor(int32)")
      .Output(0, "x", "", "tensor(int32)");

  // The roto refine ops. These MUST match python/roto/refine_graph.py::onnx_translation_table
  // exactly - same input order, same names, same types, same required attribute - since that is
  // what the exported nodes were type-checked against.
  //
  // No plugin_namespace / plugin_version, unlike the older bfx schemas above: those exist for the
  // TensorRT path, and torch.onnx.export emits neither on these nodes.

  MS_DML_OPERATOR_SCHEMA(roto_UnionForward)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0, "log1m_in", "", "T")
      .Input(1, "curve", "", "T")
      .Input(2, "pts_all", "", "T")
      .Input(3, "pos_idx", "", "I")
      .Input(4, "sharpness", "", "T")
      .Output(0, "log1m_out", "", "T")
      .TypeConstraint("T", {"tensor(float)"},
                      "accumulator (Q,P), curve (F,S,2), points (Q,P,2), rank-0 sharpness")
      .TypeConstraint("I", {"tensor(int64)"}, "this object's row into the range, (F)");

  MS_DML_OPERATOR_SCHEMA(roto_OccGrad)
      .SetDomain("bfx")
      .SinceVersion(1)
      .Input(0,  "curve", "", "T")
      .Input(1,  "pts_all", "", "T")
      .Input(2,  "own", "", "U")
      .Input(3,  "present", "", "U")
      .Input(4,  "tgt_all", "", "U")
      .Input(5,  "wgt_all", "", "H")
      .Input(6,  "log1m_all", "", "T")
      .Input(7,  "dldu_all", "", "T")
      .Input(8,  "pos_idx", "", "I")
      .Input(9,  "sharpness", "", "T")
      .Input(10, "neighbor_weight", "", "T")
      .Input(11, "own_weight", "", "T")
      .Input(12, "own_total", "", "T")
      .Output(0, "d_curve", "", "T")
      .Output(1, "own_loss", "", "T")
      .TypeConstraint("T", {"tensor(float)"},
                      "curve (F,S,2), the (Q,P) stacks, and the rank-0 weights")
      .TypeConstraint("I", {"tensor(int64)"}, "pos_idx, (F)")
      .TypeConstraint("U", {"tensor(uint8)"}, "own (F,P), present (F), tgt_all (Q,P)")
      .TypeConstraint("H", {"tensor(float16)"}, "wgt_all (Q,P) - only ever tested against zero")
      // required, and ONNX's 3-argument Attr() defaults to required - which is what we want: a
      // call that names neither mode would otherwise silently fit the wrong thing.
      .Attr("ext", "1 to fit the union against the external matte, 0 to fit each object against "
                   "its own mask. Never both.", onnx::AttributeProto::INT);
}
}  // namespace dml
}  // namespace onnxruntime
