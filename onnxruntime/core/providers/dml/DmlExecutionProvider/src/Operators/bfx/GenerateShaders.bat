@echo off

REM  generated shader bytecode is checked into source control. to update bytecode, run:
REM   cd .\onnxruntime\core\providers\dml\DmlExecutionProvider\src\Operators\bfx
REM   .\GenerateShaders.bat
REM
REM  NOTES ON THE ROTO SHADERS
REM  - cs_6_2 like everything else here, so the wave intrinsics in roto_occ_grad.hlsl
REM    (WaveActiveSum / WaveActiveMin / WaveActiveAnyTrue / WavePrefixCountBits, all SM 6.0) are
REM    already inside the profile. What they DO add is a device requirement -
REM    D3D12_FEATURE_DATA_D3D12_OPTIONS1::WaveOps. Compile with -DROTO_OG_WAVE=0 to drop it.
REM  - NO -enable-16bit-types and no fp16 variant: the roto graphs are float32 throughout (the
REM    ONNX schema pins every input's type), and the one float16 tensor is read as raw bits and
REM    only ever tested against zero.
REM  - IF THE NUMBERS DRIFT past the test-case tolerances, add -Gis (force IEEE strictness) before
REM    touching the arithmetic: DXC lowers exp(x) to exp2(x * 1.44269504) and may contract divides
REM    into reciprocal-multiplies, which is the most likely source of a mismatch against CUDA.

if "%1" == "DEBUG" (
    echo "WARNING: Compiling shaders for DEBUG configuration; do not check generated header files into the repo!"

    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float                         -Zi -Od -Qembed_debug -Fh GeneratedShaders/deform_conv2d_im2cols_fp32.h
    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh GeneratedShaders/deform_conv2d_im2cols_fp16.h

    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float                         -Zi -Od -Qembed_debug -Fh GeneratedShaders/warp_flow_fp32.h
    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh GeneratedShaders/warp_flow_fp16.h

    dxc.exe grid_sample.hlsl                     -E grid_sample                     -T cs_6_2 -Dscalar_t=float                         -Zi -Od -Qembed_debug -Fh GeneratedShaders/grid_sample_fp32.h
    dxc.exe grid_sample.hlsl                     -E grid_sample                     -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh GeneratedShaders/grid_sample_fp16.h

    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float                         -Zi -Od -Qembed_debug -Fh GeneratedShaders/second_order_deform_offset_mask_fp32.h
    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh GeneratedShaders/second_order_deform_offset_mask_fp16.h

    dxc.exe make_multiscale_upres_sample_grid.hlsl -E make_multiscale_upres_sample_grid -T cs_6_2                                      -Zi -Od -Qembed_debug -Fh GeneratedShaders/make_multiscale_upres_sample_grid.h

    dxc.exe rle_encode_get_diffs.hlsl            -E rle_encode_get_diffs            -T cs_6_2 -DT_diffs=int32_t -DT_idxs=int32_t      -Zi -Od -Qembed_debug -Fh GeneratedShaders/rle_encode_get_diffs_int32_int32.h
    dxc.exe rle_encode_write_out.hlsl            -E rle_encode_write_out            -T cs_6_2 -DT_diffs=int32_t -DT_idxs=int32_t      -Zi -Od -Qembed_debug -Fh GeneratedShaders/rle_encode_write_out_int32_int32.h
    dxc.exe rle_decode_scatter.hlsl              -E rle_decode_scatter              -T cs_6_2 -DT_vals=int32_t -DT_idxs=int32_t       -Zi -Od -Qembed_debug -Fh GeneratedShaders/rle_decode_scatter_int32_int32.h
    dxc.exe scan/scan_clear_buffer.hlsl          -E scan_clear_buffer               -T cs_6_2 -DT=int32_t                             -Zi -Od -Qembed_debug -Fh GeneratedShaders/scan_clear_buffer_int32.h
    dxc.exe scan/scan_prescan.hlsl               -E scan_prescan                    -T cs_6_2 -DT=int32_t                             -Zi -Od -Qembed_debug -Fh GeneratedShaders/scan_prescan_int32.h
    dxc.exe scan/scan_add_block_sums.hlsl        -E scan_add_block_sums             -T cs_6_2 -DT=int32_t                             -Zi -Od -Qembed_debug -Fh GeneratedShaders/scan_add_block_sums_int32.h

    dxc.exe roto_union_forward.hlsl              -E roto_union_forward              -T cs_6_2                                          -Zi -Od -Qembed_debug -Fh GeneratedShaders/roto_union_forward.h
    dxc.exe roto_occ_grad.hlsl                   -E roto_occ_grad                   -T cs_6_2                                          -Zi -Od -Qembed_debug -Fh GeneratedShaders/roto_occ_grad.h
    dxc.exe roto_occ_grad_reduce.hlsl            -E roto_occ_grad_reduce            -T cs_6_2                                          -Zi -Od -Qembed_debug -Fh GeneratedShaders/roto_occ_grad_reduce.h

) else (
    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float                         -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/deform_conv2d_im2cols_fp32.h
    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/deform_conv2d_im2cols_fp16.h

    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float                         -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/warp_flow_fp32.h
    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/warp_flow_fp16.h

    dxc.exe grid_sample.hlsl                     -E grid_sample                     -T cs_6_2 -Dscalar_t=float                         -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/grid_sample_fp32.h
    dxc.exe grid_sample.hlsl                     -E grid_sample                     -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/grid_sample_fp16.h

    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float                         -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/second_order_deform_offset_mask_fp32.h
    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/second_order_deform_offset_mask_fp16.h

    dxc.exe make_multiscale_upres_sample_grid.hlsl -E make_multiscale_upres_sample_grid -T cs_6_2                                      -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/make_multiscale_upres_sample_grid.h

    dxc.exe rle_encode_get_diffs.hlsl            -E rle_encode_get_diffs            -T cs_6_2 -DT_diffs=int32_t -DT_idxs=int32_t       -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/rle_encode_get_diffs_int32_int32.h
    dxc.exe rle_encode_write_out.hlsl            -E rle_encode_write_out            -T cs_6_2 -DT_diffs=int32_t -DT_idxs=int32_t       -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/rle_encode_write_out_int32_int32.h
    dxc.exe rle_decode_scatter.hlsl              -E rle_decode_scatter              -T cs_6_2 -DT_vals=int32_t -DT_idxs=int32_t       -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/rle_decode_scatter_int32_int32.h
    dxc.exe scan/scan_clear_buffer.hlsl          -E scan_clear_buffer               -T cs_6_2 -DT=int32_t                              -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/scan_clear_buffer_int32.h
    dxc.exe scan/scan_prescan.hlsl               -E scan_prescan                    -T cs_6_2 -DT=int32_t                              -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/scan_prescan_int32.h
    dxc.exe scan/scan_add_block_sums.hlsl        -E scan_add_block_sums             -T cs_6_2 -DT=int32_t                              -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/scan_add_block_sums_int32.h

    dxc.exe roto_union_forward.hlsl              -E roto_union_forward              -T cs_6_2                                           -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/roto_union_forward.h
    dxc.exe roto_occ_grad.hlsl                   -E roto_occ_grad                   -T cs_6_2                                           -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/roto_occ_grad.h
    dxc.exe roto_occ_grad_reduce.hlsl            -E roto_occ_grad_reduce            -T cs_6_2                                           -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/roto_occ_grad_reduce.h
)
