@echo off

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
)
