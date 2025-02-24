#if 0
;
; Note: shader requires additional functionality:
;       Use native low precision
;
;
; Input signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
;
; Output signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
; shader hash: 2fe430065d7a39bcbaa18be082ac65a0
;
; Pipeline Runtime Information: 
;
;
;
; Buffer Definitions:
;
; cbuffer 
; {
;
;   [20 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [2 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [2 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [2 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [2 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [2 x i8] (type annotation not present)
;
; }
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
;                                   cbuffer      NA          NA     CB0            cb0     1
;                                       UAV  struct         r/w      U0             u0     1
;                                       UAV  struct         r/w      U1             u1     1
;                                       UAV  struct         r/w      U2             u2     1
;                                       UAV  struct         r/w      U3             u3     1
;                                       UAV  struct         r/w      U4             u4     1
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.ResRet.f16 = type { half, half, half, half, i32 }
%dx.types.CBufRet.f32 = type { float, float, float, float }
%"class.RWStructuredBuffer<half>" = type { half }
%Constants = type { float, i32, i32, i32, i32 }

define void @second_order_deform_offset_mask() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 4, i32 4, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 3, i32 3, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 2, i32 2, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %4 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 1, i32 1, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %5 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %6 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %7 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %8 = call i32 @dx.op.threadId.i32(i32 93, i32 1)  ; ThreadId(component)
  %9 = call i32 @dx.op.threadId.i32(i32 93, i32 2)  ; ThreadId(component)
  %10 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %6, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %11 = extractvalue %dx.types.CBufRet.i32 %10, 2
  %12 = sdiv i32 %9, %11
  %13 = srem i32 %9, %11
  %14 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %6, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %15 = extractvalue %dx.types.CBufRet.i32 %14, 0
  %16 = icmp sge i32 %7, %15
  %17 = extractvalue %dx.types.CBufRet.i32 %10, 3
  %18 = icmp sge i32 %8, %17
  %19 = or i1 %16, %18
  %20 = icmp slt i32 %11, 0
  %21 = or i1 %19, %20
  %22 = extractvalue %dx.types.CBufRet.i32 %10, 1
  %23 = icmp sge i32 %12, %22
  %24 = or i1 %21, %23
  br i1 %24, label %88, label %25

; <label>:25                                      ; preds = %0
  %26 = sdiv i32 %11, 3
  %27 = shl nsw i32 %26, 1
  %28 = icmp slt i32 %13, %27
  %29 = mul nsw i32 %11, %12
  %30 = mul nsw i32 %29, %17
  %31 = mul nsw i32 %30, %15
  %32 = mul nsw i32 %17, %13
  %33 = mul nsw i32 %32, %15
  %34 = mul nsw i32 %15, %8
  %35 = add i32 %31, %7
  %36 = add i32 %35, %33
  %37 = add i32 %36, %34
  br i1 %28, label %38, label %70

; <label>:38                                      ; preds = %25
  %39 = shl i32 %12, 1
  %40 = mul nsw i32 %39, %17
  %41 = mul nsw i32 %40, %15
  %42 = add nsw i32 %13, 1
  %43 = srem i32 %42, 2
  %44 = mul nsw i32 %17, %43
  %45 = mul nsw i32 %44, %15
  %46 = add i32 %41, %7
  %47 = add i32 %46, %45
  %48 = add i32 %47, %34
  %49 = icmp slt i32 %13, %26
  %50 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %4, i32 %48, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %51 = extractvalue %dx.types.ResRet.f16 %50, 0
  %52 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %3, i32 %48, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %53 = extractvalue %dx.types.ResRet.f16 %52, 0
  %54 = select i1 %49, half %51, half %53
  %55 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %5, i32 %37, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %56 = extractvalue %dx.types.ResRet.f16 %55, 0
  %57 = fpext half %56 to float
  %58 = call %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32(i32 59, %dx.types.Handle %6, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %59 = extractvalue %dx.types.CBufRet.f32 %58, 0
  %60 = call float @dx.op.unary.f32(i32 20, float %57)  ; Htan(value)
  %61 = fmul fast float %59, %60
  %62 = fptrunc float %61 to half
  %63 = fadd fast half %62, %54
  %64 = mul nsw i32 %27, %12
  %65 = mul nsw i32 %64, %17
  %66 = mul nsw i32 %65, %15
  %67 = add i32 %66, %7
  %68 = add i32 %67, %33
  %69 = add i32 %68, %34
  call void @dx.op.rawBufferStore.f16(i32 140, %dx.types.Handle %2, i32 %69, i32 0, half %63, half undef, half undef, half undef, i8 1, i32 2)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %88

; <label>:70                                      ; preds = %25
  %71 = sub nsw i32 %13, %27
  %72 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %5, i32 %37, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %73 = extractvalue %dx.types.ResRet.f16 %72, 0
  %74 = fpext half %73 to float
  %75 = fmul fast float %74, 0xBFF7154760000000
  %76 = call float @dx.op.unary.f32(i32 21, float %75)  ; Exp(value)
  %77 = fadd fast float %76, 1.000000e+00
  %78 = fdiv fast float 1.000000e+00, %77
  %79 = fptrunc float %78 to half
  %80 = mul nsw i32 %26, %12
  %81 = mul nsw i32 %80, %17
  %82 = mul nsw i32 %81, %15
  %83 = mul nsw i32 %17, %71
  %84 = mul nsw i32 %83, %15
  %85 = add i32 %82, %7
  %86 = add i32 %85, %84
  %87 = add i32 %86, %34
  call void @dx.op.rawBufferStore.f16(i32 140, %dx.types.Handle %1, i32 %87, i32 0, half %79, half undef, half undef, half undef, i8 1, i32 2)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %88

; <label>:88                                      ; preds = %70, %38, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind readonly
declare %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32, %dx.types.Handle, i32, i32, i8, i32) #1

; Function Attrs: nounwind
declare void @dx.op.rawBufferStore.f16(i32, %dx.types.Handle, i32, i32, half, half, half, half, i8, i32) #2

; Function Attrs: nounwind readnone
declare float @dx.op.unary.f32(i32, float) #0

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32(i32, %dx.types.Handle, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind }

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!13}

!0 = !{!"clang version 3.7 (tags/RELEASE_370/final)"}
!1 = !{i32 1, i32 2}
!2 = !{i32 1, i32 6}
!3 = !{!"cs", i32 6, i32 2}
!4 = !{null, !5, !11, null}
!5 = !{!6, !7, !8, !9, !10}
!6 = !{i32 0, %"class.RWStructuredBuffer<half>"* undef, !"", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !1}
!7 = !{i32 1, %"class.RWStructuredBuffer<half>"* undef, !"", i32 0, i32 1, i32 1, i32 12, i1 false, i1 false, i1 false, !1}
!8 = !{i32 2, %"class.RWStructuredBuffer<half>"* undef, !"", i32 0, i32 2, i32 1, i32 12, i1 false, i1 false, i1 false, !1}
!9 = !{i32 3, %"class.RWStructuredBuffer<half>"* undef, !"", i32 0, i32 3, i32 1, i32 12, i1 false, i1 false, i1 false, !1}
!10 = !{i32 4, %"class.RWStructuredBuffer<half>"* undef, !"", i32 0, i32 4, i32 1, i32 12, i1 false, i1 false, i1 false, !1}
!11 = !{!12}
!12 = !{i32 0, %Constants* undef, !"", i32 0, i32 0, i32 1, i32 20, null}
!13 = !{void ()* @second_order_deform_offset_mask, !"second_order_deform_offset_mask", null, !4, !14}
!14 = !{i32 0, i64 8388656, i32 4, !15}
!15 = !{i32 16, i32 16, i32 1}

#endif

const unsigned char g_second_order_deform_offset_mask[] = {
  0x44, 0x58, 0x42, 0x43, 0x38, 0x0f, 0x15, 0x6a, 0xf4, 0x8c, 0xdd, 0x23,
  0x7a, 0xcb, 0x2c, 0x7e, 0x7d, 0x82, 0x45, 0xca, 0x01, 0x00, 0x00, 0x00,
  0xdc, 0x09, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x48, 0x01, 0x00, 0x00, 0x64, 0x01, 0x00, 0x00, 0x53, 0x46, 0x49, 0x30,
  0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x49, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x4f, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x50, 0x53, 0x56, 0x30,
  0xd8, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x48, 0x41, 0x53, 0x48, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x2f, 0xe4, 0x30, 0x06, 0x5d, 0x7a, 0x39, 0xbc,
  0xba, 0xa1, 0x8b, 0xe0, 0x82, 0xac, 0x65, 0xa0, 0x44, 0x58, 0x49, 0x4c,
  0x70, 0x08, 0x00, 0x00, 0x62, 0x00, 0x05, 0x00, 0x1c, 0x02, 0x00, 0x00,
  0x44, 0x58, 0x49, 0x4c, 0x02, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x58, 0x08, 0x00, 0x00, 0x42, 0x43, 0xc0, 0xde, 0x21, 0x0c, 0x00, 0x00,
  0x13, 0x02, 0x00, 0x00, 0x0b, 0x82, 0x20, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x07, 0x81, 0x23, 0x91, 0x41, 0xc8, 0x04, 0x49,
  0x06, 0x10, 0x32, 0x39, 0x92, 0x01, 0x84, 0x0c, 0x25, 0x05, 0x08, 0x19,
  0x1e, 0x04, 0x8b, 0x62, 0x80, 0x18, 0x45, 0x02, 0x42, 0x92, 0x0b, 0x42,
  0xc4, 0x10, 0x32, 0x14, 0x38, 0x08, 0x18, 0x4b, 0x0a, 0x32, 0x62, 0x88,
  0x48, 0x90, 0x14, 0x20, 0x43, 0x46, 0x88, 0xa5, 0x00, 0x19, 0x32, 0x42,
  0xe4, 0x48, 0x0e, 0x90, 0x11, 0x23, 0xc4, 0x50, 0x41, 0x51, 0x81, 0x8c,
  0xe1, 0x83, 0xe5, 0x8a, 0x04, 0x31, 0x46, 0x06, 0x51, 0x18, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x1b, 0x8c, 0xe0, 0xff, 0xff, 0xff, 0xff, 0x07,
  0x40, 0x02, 0xa8, 0x0d, 0x86, 0xf0, 0xff, 0xff, 0xff, 0xff, 0x03, 0x20,
  0x01, 0xd5, 0x06, 0x62, 0xf8, 0xff, 0xff, 0xff, 0xff, 0x01, 0x90, 0x00,
  0x49, 0x18, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x13, 0x82, 0x60, 0x42,
  0x20, 0x4c, 0x08, 0x06, 0x00, 0x00, 0x00, 0x00, 0x89, 0x20, 0x00, 0x00,
  0x3e, 0x00, 0x00, 0x00, 0x32, 0x22, 0x88, 0x09, 0x20, 0x64, 0x85, 0x04,
  0x13, 0x23, 0xa4, 0x84, 0x04, 0x13, 0x23, 0xe3, 0x84, 0xa1, 0x90, 0x14,
  0x12, 0x4c, 0x8c, 0x8c, 0x0b, 0x84, 0xc4, 0x4c, 0x10, 0x88, 0xc1, 0x08,
  0x40, 0x09, 0x00, 0x0a, 0xe6, 0x08, 0xc0, 0xa0, 0x0c, 0xc3, 0x30, 0x10,
  0x31, 0x0a, 0x70, 0xd3, 0x70, 0xf9, 0x13, 0xf6, 0x10, 0x92, 0xbf, 0x12,
  0xd2, 0x4a, 0x4c, 0x7e, 0x51, 0xeb, 0xa8, 0x30, 0x0c, 0xc3, 0x18, 0xe6,
  0x08, 0x10, 0x42, 0xee, 0x19, 0x2e, 0x7f, 0xc2, 0x1e, 0x42, 0xf2, 0x43,
  0xa0, 0x19, 0x16, 0x02, 0x05, 0x49, 0x39, 0x8e, 0x41, 0x19, 0x06, 0x64,
  0xa0, 0xa5, 0x2c, 0xc0, 0xa0, 0x0c, 0x83, 0x61, 0x18, 0x06, 0x32, 0x50,
  0x33, 0x03, 0x50, 0x86, 0x67, 0x78, 0x08, 0x3a, 0x6a, 0xb8, 0xfc, 0x09,
  0x7b, 0x08, 0xc9, 0xe7, 0x36, 0xaa, 0x58, 0x89, 0xc9, 0x47, 0x6e, 0x1b,
  0x11, 0xc3, 0x30, 0x0c, 0x85, 0x90, 0x06, 0x65, 0xa0, 0xe9, 0xa8, 0xe1,
  0xf2, 0x27, 0xec, 0x21, 0x24, 0x9f, 0xdb, 0xa8, 0x62, 0x25, 0x26, 0xbf,
  0xb8, 0x6d, 0x44, 0x3c, 0xcf, 0xf3, 0x14, 0xa2, 0x1a, 0x94, 0x81, 0xac,
  0x39, 0x82, 0xa0, 0x18, 0xca, 0x80, 0x0c, 0x03, 0x46, 0xd9, 0x40, 0xc0,
  0x4c, 0xde, 0x38, 0xb0, 0x43, 0x38, 0xcc, 0xc3, 0x3c, 0xb8, 0x81, 0x2c,
  0xdc, 0xc2, 0x2c, 0xd0, 0x83, 0x3c, 0xd4, 0xc3, 0x38, 0xd0, 0x43, 0x3d,
  0xc8, 0x43, 0x39, 0x90, 0x83, 0x28, 0xd4, 0x83, 0x39, 0x98, 0x43, 0x39,
  0xc8, 0x03, 0x1f, 0xa0, 0x43, 0x38, 0xb0, 0x83, 0x39, 0xf8, 0x01, 0x0a,
  0x0c, 0xe2, 0x2e, 0xe1, 0x9c, 0x46, 0x9a, 0x80, 0x66, 0x92, 0x50, 0xf1,
  0x0c, 0xc3, 0x30, 0x90, 0x37, 0x47, 0x00, 0x0a, 0x53, 0x00, 0x00, 0x00,
  0x13, 0x14, 0x72, 0xc0, 0x87, 0x74, 0x60, 0x87, 0x36, 0x68, 0x87, 0x79,
  0x68, 0x03, 0x72, 0xc0, 0x87, 0x0d, 0xae, 0x50, 0x0e, 0x6d, 0xd0, 0x0e,
  0x7a, 0x50, 0x0e, 0x6d, 0x00, 0x0f, 0x7a, 0x30, 0x07, 0x72, 0xa0, 0x07,
  0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x71, 0xa0, 0x07, 0x73, 0x20, 0x07,
  0x6d, 0x90, 0x0e, 0x78, 0xa0, 0x07, 0x78, 0xd0, 0x06, 0xe9, 0x10, 0x07,
  0x76, 0xa0, 0x07, 0x71, 0x60, 0x07, 0x6d, 0x90, 0x0e, 0x73, 0x20, 0x07,
  0x7a, 0x30, 0x07, 0x72, 0xd0, 0x06, 0xe9, 0x60, 0x07, 0x74, 0xa0, 0x07,
  0x76, 0x40, 0x07, 0x6d, 0x60, 0x0e, 0x71, 0x60, 0x07, 0x7a, 0x10, 0x07,
  0x76, 0xd0, 0x06, 0xe6, 0x30, 0x07, 0x72, 0xa0, 0x07, 0x73, 0x20, 0x07,
  0x6d, 0x60, 0x0e, 0x76, 0x40, 0x07, 0x7a, 0x60, 0x07, 0x74, 0xd0, 0x06,
  0xee, 0x80, 0x07, 0x7a, 0x10, 0x07, 0x76, 0xa0, 0x07, 0x73, 0x20, 0x07,
  0x7a, 0x60, 0x07, 0x74, 0x30, 0xe4, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0xc8, 0x43, 0x00, 0x01, 0x10, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x90, 0x67, 0x01, 0x02, 0x40,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x21, 0x4f, 0x03, 0x04,
  0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x43, 0x1e, 0x08,
  0x08, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x86, 0x3c,
  0x13, 0x10, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c,
  0x79, 0x2c, 0x20, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x18, 0xf2, 0x64, 0x40, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x90, 0x05, 0x02, 0x0c, 0x00, 0x00, 0x00, 0x32, 0x1e, 0x98, 0x14,
  0x19, 0x11, 0x4c, 0x90, 0x8c, 0x09, 0x26, 0x47, 0xc6, 0x04, 0x43, 0x1a,
  0x4a, 0xa0, 0x08, 0x8a, 0x61, 0x04, 0xa0, 0x30, 0xca, 0xa0, 0x10, 0x0a,
  0xa5, 0x40, 0x08, 0x1b, 0x01, 0x20, 0xb0, 0xc0, 0x01, 0x01, 0x11, 0xe8,
  0x9b, 0x01, 0xa0, 0x6e, 0x06, 0x00, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00,
  0x53, 0x00, 0x00, 0x00, 0x1a, 0x03, 0x4c, 0x90, 0x46, 0x02, 0x13, 0x44,
  0x35, 0x18, 0x63, 0x0b, 0x73, 0x3b, 0x03, 0xb1, 0x2b, 0x93, 0x9b, 0x4b,
  0x7b, 0x73, 0x03, 0x99, 0x71, 0xb9, 0x01, 0x41, 0xa1, 0x0b, 0x3b, 0x9b,
  0x7b, 0x91, 0x2a, 0x62, 0x2a, 0x0a, 0x9a, 0x2a, 0xfa, 0x9a, 0xb9, 0x81,
  0x79, 0x31, 0x4b, 0x73, 0x0b, 0x63, 0x4b, 0xd9, 0x10, 0x04, 0x13, 0x84,
  0x01, 0x99, 0x20, 0x0c, 0xc9, 0x06, 0x61, 0x20, 0x26, 0x08, 0x83, 0xb2,
  0x41, 0x18, 0x0c, 0x0a, 0x63, 0x73, 0x1b, 0x06, 0xc4, 0x20, 0x26, 0x08,
  0xc3, 0x32, 0x41, 0xe8, 0x28, 0x02, 0x13, 0x84, 0x81, 0x99, 0x20, 0x60,
  0xd1, 0x86, 0x45, 0x59, 0x18, 0x45, 0x19, 0x1a, 0xc7, 0x71, 0x8a, 0x0d,
  0xcb, 0xb0, 0x30, 0xca, 0x30, 0x34, 0x8e, 0xe3, 0x14, 0x1b, 0x16, 0x62,
  0x61, 0x14, 0x62, 0x68, 0x1c, 0xc7, 0x29, 0x26, 0x08, 0x43, 0xb3, 0x61,
  0x91, 0x16, 0x46, 0x91, 0x86, 0xc6, 0x71, 0x9c, 0x62, 0x82, 0x30, 0x38,
  0x1b, 0x16, 0x6a, 0x61, 0x14, 0x6a, 0x68, 0x1c, 0xc7, 0x29, 0x36, 0x14,
  0x0f, 0x14, 0x4d, 0xd5, 0x04, 0xe1, 0x9b, 0x26, 0x08, 0xc3, 0xb3, 0x01,
  0x51, 0x2e, 0x46, 0x51, 0x06, 0x0c, 0xd8, 0x10, 0x64, 0x1b, 0x08, 0xc0,
  0xd2, 0x80, 0x09, 0x82, 0x00, 0xf0, 0x9b, 0x2b, 0x1b, 0x7b, 0x73, 0x23,
  0xfb, 0x7a, 0x93, 0x23, 0x2b, 0x93, 0xfb, 0x22, 0x2b, 0x33, 0x7b, 0x93,
  0x6b, 0xfb, 0x7a, 0x33, 0x33, 0x9b, 0x2b, 0xa3, 0xfb, 0x6a, 0x0b, 0x9b,
  0x5b, 0x9b, 0x20, 0x80, 0x81, 0x34, 0x41, 0x18, 0xa0, 0x0d, 0xc3, 0xf7,
  0x0d, 0x1b, 0x08, 0xc5, 0xa3, 0xc0, 0x60, 0x43, 0xc1, 0x75, 0xc0, 0x16,
  0x06, 0x55, 0xd8, 0xd8, 0xec, 0xda, 0x5c, 0xd2, 0xc8, 0xca, 0xdc, 0xe8,
  0xa6, 0x04, 0x41, 0x15, 0x32, 0x3c, 0x17, 0xbb, 0x32, 0xb9, 0xb9, 0xb4,
  0x37, 0xb7, 0x29, 0x01, 0xd1, 0x84, 0x0c, 0xcf, 0xc5, 0x2e, 0x8c, 0xcd,
  0xae, 0x4c, 0x6e, 0x4a, 0x60, 0xd4, 0x21, 0xc3, 0x73, 0x99, 0x43, 0x0b,
  0x23, 0x2b, 0x93, 0x6b, 0x7a, 0x23, 0x2b, 0x63, 0x9b, 0x12, 0x20, 0x65,
  0xc8, 0xf0, 0x5c, 0xe4, 0xca, 0xe6, 0xde, 0xea, 0xe4, 0xc6, 0xca, 0xe6,
  0xa6, 0x04, 0x5a, 0x1d, 0x32, 0x3c, 0x97, 0x32, 0x37, 0x3a, 0xb9, 0x3c,
  0xa8, 0xb7, 0x34, 0x37, 0xba, 0xb9, 0x29, 0x41, 0x18, 0x00, 0x00, 0x00,
  0x79, 0x18, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00, 0x33, 0x08, 0x80, 0x1c,
  0xc4, 0xe1, 0x1c, 0x66, 0x14, 0x01, 0x3d, 0x88, 0x43, 0x38, 0x84, 0xc3,
  0x8c, 0x42, 0x80, 0x07, 0x79, 0x78, 0x07, 0x73, 0x98, 0x71, 0x0c, 0xe6,
  0x00, 0x0f, 0xed, 0x10, 0x0e, 0xf4, 0x80, 0x0e, 0x33, 0x0c, 0x42, 0x1e,
  0xc2, 0xc1, 0x1d, 0xce, 0xa1, 0x1c, 0x66, 0x30, 0x05, 0x3d, 0x88, 0x43,
  0x38, 0x84, 0x83, 0x1b, 0xcc, 0x03, 0x3d, 0xc8, 0x43, 0x3d, 0x8c, 0x03,
  0x3d, 0xcc, 0x78, 0x8c, 0x74, 0x70, 0x07, 0x7b, 0x08, 0x07, 0x79, 0x48,
  0x87, 0x70, 0x70, 0x07, 0x7a, 0x70, 0x03, 0x76, 0x78, 0x87, 0x70, 0x20,
  0x87, 0x19, 0xcc, 0x11, 0x0e, 0xec, 0x90, 0x0e, 0xe1, 0x30, 0x0f, 0x6e,
  0x30, 0x0f, 0xe3, 0xf0, 0x0e, 0xf0, 0x50, 0x0e, 0x33, 0x10, 0xc4, 0x1d,
  0xde, 0x21, 0x1c, 0xd8, 0x21, 0x1d, 0xc2, 0x61, 0x1e, 0x66, 0x30, 0x89,
  0x3b, 0xbc, 0x83, 0x3b, 0xd0, 0x43, 0x39, 0xb4, 0x03, 0x3c, 0xbc, 0x83,
  0x3c, 0x84, 0x03, 0x3b, 0xcc, 0xf0, 0x14, 0x76, 0x60, 0x07, 0x7b, 0x68,
  0x07, 0x37, 0x68, 0x87, 0x72, 0x68, 0x07, 0x37, 0x80, 0x87, 0x70, 0x90,
  0x87, 0x70, 0x60, 0x07, 0x76, 0x28, 0x07, 0x76, 0xf8, 0x05, 0x76, 0x78,
  0x87, 0x77, 0x80, 0x87, 0x5f, 0x08, 0x87, 0x71, 0x18, 0x87, 0x72, 0x98,
  0x87, 0x79, 0x98, 0x81, 0x2c, 0xee, 0xf0, 0x0e, 0xee, 0xe0, 0x0e, 0xf5,
  0xc0, 0x0e, 0xec, 0x30, 0x03, 0x62, 0xc8, 0xa1, 0x1c, 0xe4, 0xa1, 0x1c,
  0xcc, 0xa1, 0x1c, 0xe4, 0xa1, 0x1c, 0xdc, 0x61, 0x1c, 0xca, 0x21, 0x1c,
  0xc4, 0x81, 0x1d, 0xca, 0x61, 0x06, 0xd6, 0x90, 0x43, 0x39, 0xc8, 0x43,
  0x39, 0x98, 0x43, 0x39, 0xc8, 0x43, 0x39, 0xb8, 0xc3, 0x38, 0x94, 0x43,
  0x38, 0x88, 0x03, 0x3b, 0x94, 0xc3, 0x2f, 0xbc, 0x83, 0x3c, 0xfc, 0x82,
  0x3b, 0xd4, 0x03, 0x3b, 0xb0, 0xc3, 0x0c, 0xc4, 0x21, 0x07, 0x7c, 0x70,
  0x03, 0x7a, 0x28, 0x87, 0x76, 0x80, 0x87, 0x19, 0xd1, 0x43, 0x0e, 0xf8,
  0xe0, 0x06, 0xe4, 0x20, 0x0e, 0xe7, 0xe0, 0x06, 0xf6, 0x10, 0x0e, 0xf2,
  0xc0, 0x0e, 0xe1, 0x90, 0x0f, 0xef, 0x50, 0x0f, 0xf4, 0x30, 0x83, 0x81,
  0xc8, 0x01, 0x1f, 0xdc, 0x40, 0x1c, 0xe4, 0xa1, 0x1c, 0xc2, 0x61, 0x1d,
  0xdc, 0x40, 0x1c, 0xe4, 0x01, 0x00, 0x00, 0x00, 0x71, 0x20, 0x00, 0x00,
  0x27, 0x00, 0x00, 0x00, 0x66, 0xb0, 0x0d, 0x97, 0xef, 0x3c, 0xbe, 0x10,
  0x50, 0x45, 0x41, 0x44, 0xa5, 0x03, 0x0c, 0x25, 0x61, 0x00, 0x02, 0xe6,
  0x17, 0xb7, 0x6d, 0x05, 0xdb, 0x70, 0xf9, 0xce, 0xe3, 0x0b, 0x01, 0x55,
  0x14, 0x44, 0x54, 0x3a, 0xc0, 0x50, 0x12, 0x06, 0x20, 0x60, 0x3e, 0x72,
  0xdb, 0x76, 0x20, 0x0d, 0x97, 0xef, 0x3c, 0xbe, 0x10, 0x11, 0xc0, 0x44,
  0x84, 0x40, 0x33, 0x2c, 0x84, 0x09, 0x5c, 0xc3, 0xe5, 0x3b, 0x8f, 0x1f,
  0x01, 0xd6, 0x46, 0x15, 0x05, 0x11, 0x95, 0x0e, 0x30, 0xf8, 0x45, 0xad,
  0xdb, 0x00, 0x36, 0x5c, 0xbe, 0xf3, 0xf8, 0x11, 0x60, 0x6d, 0x54, 0x51,
  0x10, 0x11, 0x3b, 0x39, 0x11, 0xe1, 0x17, 0xb5, 0x6e, 0x01, 0xd2, 0x70,
  0xf9, 0xce, 0xe3, 0x4f, 0x47, 0x44, 0x00, 0x83, 0x38, 0xf8, 0xc8, 0x6d,
  0x1b, 0xc1, 0x33, 0x5c, 0xbe, 0xf3, 0xf8, 0x54, 0x03, 0x44, 0x98, 0x5f,
  0xdc, 0xb6, 0x01, 0x7c, 0x12, 0x21, 0x38, 0xcd, 0xf0, 0x3b, 0xd1, 0x40,
  0x44, 0xff, 0x40, 0x14, 0x4e, 0xc4, 0xfc, 0x4e, 0x51, 0x48, 0xc4, 0xf4,
  0x33, 0x80, 0xa4, 0x00, 0x61, 0x20, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00,
  0x13, 0x04, 0x45, 0x2c, 0x10, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x34, 0x94, 0x5c, 0x59, 0x0a, 0x94, 0x6e, 0x40, 0xd9, 0x15, 0xa6, 0x40,
  0xa9, 0x10, 0x52, 0x04, 0x25, 0x40, 0xc6, 0x0c, 0x00, 0x3d, 0x63, 0x04,
  0x20, 0x08, 0x82, 0xf8, 0x37, 0x46, 0xb0, 0xc7, 0x6a, 0xbc, 0xff, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x23, 0x06, 0x09, 0x00, 0x82, 0x60, 0x90, 0x65,
  0x0b, 0x21, 0x49, 0xcf, 0x88, 0x41, 0x02, 0x80, 0x20, 0x18, 0x64, 0x1a,
  0x53, 0x50, 0x14, 0x34, 0x62, 0x90, 0x00, 0x20, 0x08, 0x06, 0xd9, 0xd6,
  0x18, 0x59, 0x16, 0x8d, 0x18, 0x24, 0x00, 0x08, 0x82, 0x41, 0xc6, 0x39,
  0xc7, 0xb6, 0x49, 0x23, 0x06, 0x09, 0x00, 0x82, 0x60, 0x90, 0x75, 0x0f,
  0x92, 0x65, 0xd3, 0x88, 0x41, 0x02, 0x80, 0x20, 0x18, 0x64, 0x1e, 0xa4,
  0x68, 0x1a, 0x35, 0x62, 0x60, 0x00, 0x20, 0x08, 0x06, 0x44, 0x19, 0x3c,
  0xdb, 0x88, 0x81, 0x01, 0x80, 0x20, 0x18, 0x10, 0x66, 0x00, 0x7d, 0x23,
  0x06, 0x06, 0x00, 0x82, 0x60, 0x40, 0x9c, 0x41, 0xf4, 0x8d, 0x18, 0x1c,
  0x00, 0x08, 0x82, 0xc1, 0x44, 0x06, 0x11, 0xe1, 0x8d, 0x26, 0x04, 0x42,
  0x0d, 0x01, 0x15, 0x21, 0xd8, 0x88, 0xc1, 0x01, 0x80, 0x20, 0x18, 0x4c,
  0x68, 0x50, 0x21, 0x65, 0x30, 0x9a, 0x10, 0x00, 0xc3, 0x0d, 0x49, 0x70,
  0x06, 0xa3, 0x09, 0xc7, 0x30, 0xdc, 0xa0, 0x04, 0x67, 0x50, 0x43, 0xb0,
  0xc3, 0x0d, 0x09, 0x1a, 0xa0, 0x41, 0x09, 0xc1, 0x8e, 0x26, 0x30, 0xc1,
  0x70, 0xc3, 0x12, 0x9c, 0x41, 0x0d, 0xc1, 0xce, 0x32, 0x10, 0x41, 0x50,
  0xce, 0x1a, 0x90, 0x05, 0x71, 0x70, 0x82, 0xe1, 0x06, 0x27, 0x40, 0x03,
  0x8b, 0x20, 0x11, 0x58, 0xc0, 0x88, 0xc0, 0x82, 0x47, 0x04, 0xe6, 0x48,
  0x22, 0xb0, 0x20, 0x12, 0x81, 0x49, 0x99, 0x08, 0x8a, 0xd8, 0xa0, 0x82,
  0x01, 0x2a, 0x18, 0x60, 0x96, 0x41, 0x18, 0x94, 0xca, 0xf8, 0xe0, 0x2c,
  0xa8, 0x44, 0x60, 0x01, 0x26, 0x02, 0xdb, 0xfe, 0x00, 0x04, 0x15, 0xfc,
  0x81, 0x59, 0x16, 0x88, 0xc0, 0x02, 0x4e, 0x04, 0x55, 0x94, 0x01, 0x54,
  0x20, 0x40, 0x05, 0x0d, 0x0c, 0x37, 0x88, 0x81, 0x85, 0x06, 0x23, 0x06,
  0x0a, 0x00, 0x82, 0x60, 0xb0, 0xb4, 0x42, 0x1e, 0xb0, 0x81, 0x40, 0x0a,
  0x73, 0x60, 0x0a, 0xa3, 0x09, 0x01, 0x30, 0x62, 0xa0, 0x00, 0x20, 0x08,
  0x06, 0xcb, 0x2b, 0xec, 0xc1, 0x1b, 0x10, 0xa6, 0x50, 0x07, 0xa8, 0x30,
  0x9a, 0x10, 0x00, 0xd3, 0x0d, 0x43, 0x50, 0x8c, 0x18, 0x28, 0x00, 0x08,
  0x82, 0xc1, 0x22, 0x0b, 0x7e, 0x00, 0x07, 0x51, 0x2a, 0xe0, 0xc1, 0x2a,
  0x8c, 0x26, 0x04, 0xc0, 0x05, 0x0f, 0x8e, 0x18, 0x1c, 0x00, 0x08, 0x82,
  0x81, 0x15, 0x0b, 0x7f, 0x20, 0x07, 0xac, 0x30, 0x9a, 0x10, 0x00, 0x23,
  0x06, 0x06, 0x00, 0x82, 0x60, 0x00, 0xd5, 0x82, 0x2a, 0x0c, 0x26, 0x04,
  0xf2, 0xb9, 0xc0, 0x38, 0x0b, 0x12, 0xf8, 0x18, 0x19, 0xc8, 0x81, 0x08,
  0x2c, 0x70, 0x03, 0x11, 0x58, 0x10, 0x07, 0x22, 0xa8, 0x40, 0x0f, 0xa0,
  0x02, 0x31, 0x80, 0x0a, 0xc4, 0x00, 0x46, 0x0c, 0x1c, 0x00, 0x04, 0xc1,
  0xa0, 0x01, 0x07, 0x55, 0x10, 0x85, 0x00, 0x17, 0x0e, 0x53, 0x30, 0x05,
  0x53, 0x38, 0x05, 0x5d, 0x98, 0x25, 0x20, 0xec, 0x0e, 0xd4, 0x20, 0x04,
  0x23, 0x06, 0x0a, 0x00, 0x82, 0x60, 0xb0, 0x88, 0x83, 0x2b, 0x80, 0x42,
  0x18, 0xe4, 0x02, 0x2a, 0xec, 0xc2, 0x68, 0x42, 0x00, 0x5c, 0xf0, 0xa0,
  0x05, 0xa8, 0x20, 0x9f, 0x11, 0x03, 0x03, 0x00, 0x41, 0x30, 0x80, 0xc8,
  0xc1, 0x15, 0x02, 0x0b, 0x56, 0x01, 0x3e, 0xc6, 0x0a, 0x01, 0x7d, 0x2e,
  0x30, 0xce, 0xe8, 0x20, 0x14, 0x44, 0x60, 0x41, 0x1f, 0x88, 0xc0, 0x02,
  0x50, 0x10, 0x81, 0xfd, 0x01, 0x23, 0x02, 0x0b, 0x44, 0x41, 0x04, 0x35,
  0xac, 0x02, 0x54, 0x20, 0x40, 0x05, 0x73, 0x00, 0x23, 0x06, 0x0e, 0x00,
  0x82, 0x60, 0xd0, 0xc4, 0xc3, 0x2e, 0xd0, 0x42, 0x90, 0x0e, 0xc9, 0x2d,
  0xdc, 0xc2, 0x2d, 0xe0, 0xc2, 0x3a, 0xcc, 0x12, 0x10, 0x08, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00
};
