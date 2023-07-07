@echo off

if "%1" == "DEBUG" (
    echo "WARNING: Compiling shaders for DEBUG configuration; do not check generated header files into the repo!"

    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float                         -Zi -Od -Qembed_debug -Fh GeneratedShaders/deform_conv2d_im2cols_fp32.h
    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh GeneratedShaders/deform_conv2d_im2cols_fp16.h

    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float                         -Zi -Od -Qembed_debug -Fh GeneratedShaders/warp_flow_fp32.h
    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh GeneratedShaders/warp_flow_fp16.h

    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float                         -Zi -Od -Qembed_debug -Fh GeneratedShaders/second_order_deform_offset_mask_fp32.h
    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -Zi -Od -Qembed_debug -Fh GeneratedShaders/second_order_deform_offset_mask_fp16.h
) else (
    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float                         -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/deform_conv2d_im2cols_fp32.h
    dxc.exe deform_conv2d_im2cols.hlsl           -E deform_conv2d_im2cols           -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/deform_conv2d_im2cols_fp16.h

    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float                         -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/warp_flow_fp32.h
    dxc.exe warp_flow.hlsl                       -E warp_flow                       -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/warp_flow_fp16.h

    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float                         -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/second_order_deform_offset_mask_fp32.h
    dxc.exe second_order_deform_offset_mask.hlsl -E second_order_deform_offset_mask -T cs_6_2 -Dscalar_t=float16_t -enable-16bit-types -O3 -Qstrip_reflect -Qstrip_debug -Qstrip_rootsignature -Fh GeneratedShaders/second_order_deform_offset_mask_fp16.h
)
