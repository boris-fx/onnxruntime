#if 0
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
; shader hash: 7d443420b4893534ab8e35b0ee90af30
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
;   [12 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [4 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [4 x i8] (type annotation not present)
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
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.ResRet.f32 = type { float, float, float, float, i32 }
%"class.RWStructuredBuffer<float>" = type { float }
%Constants = type { i32, i32, i32 }

define void @make_multiscale_upres_sample_grid() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 1, i32 1, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %4 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %5 = call i32 @dx.op.threadId.i32(i32 93, i32 1)  ; ThreadId(component)
  %6 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %7 = extractvalue %dx.types.CBufRet.i32 %6, 1
  %8 = icmp sge i32 %4, %7
  %9 = extractvalue %dx.types.CBufRet.i32 %6, 2
  %10 = icmp sge i32 %5, %9
  %11 = or i1 %8, %10
  br i1 %11, label %70, label %12

; <label>:12                                      ; preds = %0
  %13 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %2, i32 0, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %14 = extractvalue %dx.types.ResRet.f32 %13, 0
  %15 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %2, i32 1, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %16 = extractvalue %dx.types.ResRet.f32 %15, 0
  %17 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %2, i32 2, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %18 = extractvalue %dx.types.ResRet.f32 %17, 0
  %19 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %2, i32 3, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %20 = extractvalue %dx.types.ResRet.f32 %19, 0
  %21 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %2, i32 4, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %22 = extractvalue %dx.types.ResRet.f32 %21, 0
  %23 = sitofp i32 %4 to float
  %24 = fadd fast float %23, 5.000000e-01
  %25 = fadd fast float %24, %14
  %26 = fdiv fast float %25, %22
  %27 = fsub fast float %26, %18
  %28 = sitofp i32 %7 to float
  %29 = fdiv fast float %27, %28
  %30 = sitofp i32 %5 to float
  %31 = fadd fast float %30, 5.000000e-01
  %32 = fadd fast float %31, %16
  %33 = fdiv fast float %32, %22
  %34 = fsub fast float %33, %20
  %35 = sitofp i32 %9 to float
  %36 = fdiv fast float %34, %35
  %37 = fmul fast float %29, 2.000000e+00
  %38 = fadd fast float %37, -1.000000e+00
  %39 = fmul fast float %36, 2.000000e+00
  %40 = fadd fast float %39, -1.000000e+00
  %41 = extractvalue %dx.types.CBufRet.i32 %6, 0
  %42 = icmp sgt i32 %41, 0
  br i1 %42, label %43, label %70

; <label>:43                                      ; preds = %12
  br label %44

; <label>:44                                      ; preds = %44, %43
  %45 = phi i32 [ %65, %44 ], [ 0, %43 ]
  %46 = shl i32 %45, 1
  %47 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %48 = extractvalue %dx.types.CBufRet.i32 %47, 1
  %49 = mul nsw i32 %46, %48
  %50 = extractvalue %dx.types.CBufRet.i32 %47, 2
  %51 = mul nsw i32 %49, %50
  %52 = mul nsw i32 %48, %5
  %53 = add i32 %51, %4
  %54 = add i32 %53, %52
  call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %1, i32 %54, i32 0, float %38, float undef, float undef, float undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %55 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %56 = extractvalue %dx.types.CBufRet.i32 %55, 1
  %57 = mul nsw i32 %46, %56
  %58 = extractvalue %dx.types.CBufRet.i32 %55, 2
  %59 = mul nsw i32 %57, %58
  %60 = mul nsw i32 %58, %56
  %61 = mul nsw i32 %56, %5
  %62 = add i32 %59, %4
  %63 = add i32 %62, %60
  %64 = add i32 %63, %61
  call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %1, i32 %64, i32 0, float %40, float undef, float undef, float undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %65 = add nuw nsw i32 %45, 1
  %66 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %67 = extractvalue %dx.types.CBufRet.i32 %66, 0
  %68 = icmp slt i32 %65, %67
  br i1 %68, label %44, label %69

; <label>:69                                      ; preds = %44
  br label %70

; <label>:70                                      ; preds = %69, %12, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind readonly
declare %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32, %dx.types.Handle, i32, i32, i8, i32) #1

; Function Attrs: nounwind
declare void @dx.op.rawBufferStore.f32(i32, %dx.types.Handle, i32, i32, float, float, float, float, i8, i32) #2

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #1

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
!dx.entryPoints = !{!11}

!0 = !{!"clang version 3.7 (tags/RELEASE_370/final)"}
!1 = !{i32 1, i32 2}
!2 = !{i32 1, i32 6}
!3 = !{!"cs", i32 6, i32 2}
!4 = !{null, !5, !9, null}
!5 = !{!6, !8}
!6 = !{i32 0, %"class.RWStructuredBuffer<float>"* undef, !"", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!7 = !{i32 1, i32 4}
!8 = !{i32 1, %"class.RWStructuredBuffer<float>"* undef, !"", i32 0, i32 1, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!9 = !{!10}
!10 = !{i32 0, %Constants* undef, !"", i32 0, i32 0, i32 1, i32 12, null}
!11 = !{void ()* @make_multiscale_upres_sample_grid, !"make_multiscale_upres_sample_grid", null, !4, !12}
!12 = !{i32 0, i64 16, i32 4, !13}
!13 = !{i32 16, i32 16, i32 1}

#endif

const unsigned char g_make_multiscale_upres_sample_grid[] = {
  0x44, 0x58, 0x42, 0x43, 0x89, 0x68, 0x0b, 0x8a, 0xf1, 0x0e, 0xf6, 0x9c,
  0xe2, 0xa2, 0x86, 0x2b, 0xe1, 0xe7, 0x67, 0x44, 0x01, 0x00, 0x00, 0x00,
  0xac, 0x08, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x1c, 0x01, 0x00, 0x00, 0x53, 0x46, 0x49, 0x30,
  0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x49, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x4f, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x50, 0x53, 0x56, 0x30,
  0x90, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x48, 0x41, 0x53, 0x48, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x7d, 0x44, 0x34, 0x20, 0xb4, 0x89, 0x35, 0x34,
  0xab, 0x8e, 0x35, 0xb0, 0xee, 0x90, 0xaf, 0x30, 0x44, 0x58, 0x49, 0x4c,
  0x88, 0x07, 0x00, 0x00, 0x62, 0x00, 0x05, 0x00, 0xe2, 0x01, 0x00, 0x00,
  0x44, 0x58, 0x49, 0x4c, 0x02, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x70, 0x07, 0x00, 0x00, 0x42, 0x43, 0xc0, 0xde, 0x21, 0x0c, 0x00, 0x00,
  0xd9, 0x01, 0x00, 0x00, 0x0b, 0x82, 0x20, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x07, 0x81, 0x23, 0x91, 0x41, 0xc8, 0x04, 0x49,
  0x06, 0x10, 0x32, 0x39, 0x92, 0x01, 0x84, 0x0c, 0x25, 0x05, 0x08, 0x19,
  0x1e, 0x04, 0x8b, 0x62, 0x80, 0x14, 0x45, 0x02, 0x42, 0x92, 0x0b, 0x42,
  0xa4, 0x10, 0x32, 0x14, 0x38, 0x08, 0x18, 0x4b, 0x0a, 0x32, 0x52, 0x88,
  0x48, 0x90, 0x14, 0x20, 0x43, 0x46, 0x88, 0xa5, 0x00, 0x19, 0x32, 0x42,
  0xe4, 0x48, 0x0e, 0x90, 0x91, 0x22, 0xc4, 0x50, 0x41, 0x51, 0x81, 0x8c,
  0xe1, 0x83, 0xe5, 0x8a, 0x04, 0x29, 0x46, 0x06, 0x51, 0x18, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x1b, 0x8c, 0xe0, 0xff, 0xff, 0xff, 0xff, 0x07,
  0x40, 0x02, 0xa8, 0x0d, 0x86, 0xf0, 0xff, 0xff, 0xff, 0xff, 0x03, 0x20,
  0x01, 0xd5, 0x06, 0x62, 0xf8, 0xff, 0xff, 0xff, 0xff, 0x01, 0x90, 0x00,
  0x49, 0x18, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x13, 0x82, 0x60, 0x42,
  0x20, 0x4c, 0x08, 0x06, 0x00, 0x00, 0x00, 0x00, 0x89, 0x20, 0x00, 0x00,
  0x34, 0x00, 0x00, 0x00, 0x32, 0x22, 0x48, 0x09, 0x20, 0x64, 0x85, 0x04,
  0x93, 0x22, 0xa4, 0x84, 0x04, 0x93, 0x22, 0xe3, 0x84, 0xa1, 0x90, 0x14,
  0x12, 0x4c, 0x8a, 0x8c, 0x0b, 0x84, 0xa4, 0x4c, 0x10, 0x70, 0x23, 0x00,
  0x25, 0x00, 0x14, 0xe6, 0x08, 0xc0, 0xa0, 0x0c, 0x63, 0x0c, 0x22, 0x33,
  0x00, 0x37, 0x0d, 0x97, 0x3f, 0x61, 0x0f, 0x21, 0xf9, 0x2b, 0x21, 0xad,
  0xc4, 0xe4, 0x17, 0xb7, 0x8d, 0x0a, 0x63, 0x8c, 0x19, 0x73, 0x04, 0x08,
  0xa1, 0x7b, 0x86, 0xcb, 0x9f, 0xb0, 0x87, 0x90, 0xfc, 0x10, 0x68, 0x86,
  0x85, 0x40, 0x41, 0x2a, 0xc7, 0x19, 0x6a, 0x0c, 0x34, 0x68, 0x95, 0x05,
  0x0c, 0x35, 0x86, 0x31, 0xc6, 0xa0, 0x41, 0xed, 0xa8, 0xe1, 0xf2, 0x27,
  0xec, 0x21, 0x24, 0x9f, 0xdb, 0xa8, 0x62, 0x25, 0x26, 0x1f, 0xb9, 0x6d,
  0x44, 0x8c, 0x31, 0x46, 0x21, 0xde, 0x50, 0x83, 0xe0, 0x1c, 0x41, 0x50,
  0x0c, 0x35, 0xd0, 0x18, 0x92, 0xe6, 0x40, 0xc0, 0x4c, 0xdf, 0x38, 0xb0,
  0x43, 0x38, 0xcc, 0xc3, 0x3c, 0xb8, 0x81, 0x2c, 0xdc, 0xc2, 0x2c, 0xd0,
  0x83, 0x3c, 0xd4, 0xc3, 0x38, 0xd0, 0x43, 0x3d, 0xc8, 0x43, 0x39, 0x90,
  0x83, 0x28, 0xd4, 0x83, 0x39, 0x98, 0x43, 0x39, 0xc8, 0x03, 0x1f, 0x98,
  0x03, 0x3b, 0xbc, 0x43, 0x38, 0xd0, 0x83, 0x1f, 0xa0, 0xc0, 0x90, 0xbd,
  0x84, 0x73, 0x1a, 0x69, 0x02, 0x9a, 0x49, 0x42, 0xc3, 0x18, 0x83, 0xf0,
  0x1c, 0x01, 0x28, 0x4c, 0x01, 0x00, 0x00, 0x00, 0x13, 0x14, 0x72, 0xc0,
  0x87, 0x74, 0x60, 0x87, 0x36, 0x68, 0x87, 0x79, 0x68, 0x03, 0x72, 0xc0,
  0x87, 0x0d, 0xaf, 0x50, 0x0e, 0x6d, 0xd0, 0x0e, 0x7a, 0x50, 0x0e, 0x6d,
  0x00, 0x0f, 0x7a, 0x30, 0x07, 0x72, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d,
  0x90, 0x0e, 0x71, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x78,
  0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x71, 0x60, 0x07, 0x7a,
  0x30, 0x07, 0x72, 0xd0, 0x06, 0xe9, 0x30, 0x07, 0x72, 0xa0, 0x07, 0x73,
  0x20, 0x07, 0x6d, 0x90, 0x0e, 0x76, 0x40, 0x07, 0x7a, 0x60, 0x07, 0x74,
  0xd0, 0x06, 0xe6, 0x10, 0x07, 0x76, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d,
  0x60, 0x0e, 0x73, 0x20, 0x07, 0x7a, 0x30, 0x07, 0x72, 0xd0, 0x06, 0xe6,
  0x60, 0x07, 0x74, 0xa0, 0x07, 0x76, 0x40, 0x07, 0x6d, 0xe0, 0x0e, 0x78,
  0xa0, 0x07, 0x71, 0x60, 0x07, 0x7a, 0x30, 0x07, 0x72, 0xa0, 0x07, 0x76,
  0x40, 0x07, 0x43, 0x9e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x86, 0x3c, 0x04, 0x10, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x0c, 0x79, 0x16, 0x20, 0x00, 0x04, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0xf2, 0x34, 0x40, 0x00, 0x0c, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0xe4, 0x81, 0x80, 0x00, 0x10,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0xc8, 0x33, 0x01, 0x01,
  0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x16, 0x08, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x32, 0x1e, 0x98, 0x14, 0x19, 0x11, 0x4c, 0x90,
  0x8c, 0x09, 0x26, 0x47, 0xc6, 0x04, 0x43, 0x1a, 0x25, 0x50, 0x04, 0xc5,
  0x30, 0x02, 0x50, 0x18, 0x85, 0x50, 0x20, 0x24, 0x47, 0x00, 0x48, 0x17,
  0x08, 0xe5, 0x19, 0x00, 0xba, 0x33, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00,
  0x4a, 0x00, 0x00, 0x00, 0x1a, 0x03, 0x4c, 0x90, 0x46, 0x02, 0x13, 0x44,
  0x35, 0x18, 0x63, 0x0b, 0x73, 0x3b, 0x03, 0xb1, 0x2b, 0x93, 0x9b, 0x4b,
  0x7b, 0x73, 0x03, 0x99, 0x71, 0xb9, 0x01, 0x41, 0xa1, 0x0b, 0x3b, 0x9b,
  0x7b, 0x91, 0x2a, 0x62, 0x2a, 0x0a, 0x9a, 0x2a, 0xfa, 0x9a, 0xb9, 0x81,
  0x79, 0x31, 0x4b, 0x73, 0x0b, 0x63, 0x4b, 0xd9, 0x10, 0x04, 0x13, 0x84,
  0xc1, 0x98, 0x20, 0x0c, 0xc7, 0x06, 0x61, 0x20, 0x26, 0x08, 0x03, 0xb2,
  0x41, 0x18, 0x0c, 0x0a, 0x63, 0x73, 0x1b, 0x06, 0xc4, 0x20, 0x26, 0x08,
  0x43, 0x32, 0x41, 0xb8, 0x20, 0x02, 0x13, 0x84, 0x41, 0x99, 0x20, 0x48,
  0xcd, 0x04, 0x61, 0x58, 0x36, 0x08, 0xc3, 0xb3, 0x61, 0x51, 0x16, 0x46,
  0x51, 0x86, 0xc6, 0x71, 0x1c, 0x68, 0xc3, 0x32, 0x2c, 0x8c, 0x32, 0x0c,
  0x8d, 0xe3, 0x38, 0xd0, 0x06, 0x21, 0x92, 0x26, 0x08, 0xd9, 0xb3, 0x01,
  0x51, 0x28, 0x46, 0x51, 0x86, 0x06, 0xd8, 0x10, 0x54, 0x1b, 0x08, 0x60,
  0xb2, 0x80, 0x09, 0x82, 0x00, 0x30, 0x0c, 0xda, 0xc2, 0xd6, 0xca, 0xbe,
  0xda, 0xea, 0xd8, 0xe8, 0xd2, 0xe6, 0xc6, 0xc2, 0xd8, 0xca, 0xbe, 0xea,
  0xe0, 0xe4, 0xca, 0xe6, 0xbe, 0xe6, 0xc2, 0xda, 0xe0, 0xd8, 0xca, 0xbe,
  0xce, 0xe4, 0xd2, 0xc8, 0x26, 0x08, 0x9a, 0x33, 0x41, 0x18, 0x98, 0x0d,
  0xc3, 0xb6, 0x0d, 0x1b, 0x08, 0x45, 0x7b, 0xb8, 0x0d, 0x05, 0x96, 0x01,
  0x57, 0x57, 0x85, 0x8d, 0xcd, 0xae, 0xcd, 0x25, 0x8d, 0xac, 0xcc, 0x8d,
  0x6e, 0x4a, 0x10, 0x54, 0x21, 0xc3, 0x73, 0xb1, 0x2b, 0x93, 0x9b, 0x4b,
  0x7b, 0x73, 0x9b, 0x12, 0x10, 0x4d, 0xc8, 0xf0, 0x5c, 0xec, 0xc2, 0xd8,
  0xec, 0xca, 0xe4, 0xa6, 0x04, 0x46, 0x1d, 0x32, 0x3c, 0x97, 0x39, 0xb4,
  0x30, 0xb2, 0x32, 0xb9, 0xa6, 0x37, 0xb2, 0x32, 0xb6, 0x29, 0x01, 0x52,
  0x86, 0x0c, 0xcf, 0x45, 0xae, 0x6c, 0xee, 0xad, 0x4e, 0x6e, 0xac, 0x6c,
  0x6e, 0x4a, 0x60, 0xd5, 0x21, 0xc3, 0x73, 0x29, 0x73, 0xa3, 0x93, 0xcb,
  0x83, 0x7a, 0x4b, 0x73, 0xa3, 0x9b, 0x9b, 0x12, 0x74, 0x00, 0x00, 0x00,
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
  0x1f, 0x00, 0x00, 0x00, 0x46, 0xb0, 0x0d, 0x97, 0xef, 0x3c, 0xbe, 0x10,
  0x50, 0x45, 0x41, 0x44, 0xa5, 0x03, 0x0c, 0x25, 0x61, 0x00, 0x02, 0xe6,
  0x23, 0xb7, 0x6d, 0x05, 0xd2, 0x70, 0xf9, 0xce, 0xe3, 0x0b, 0x11, 0x01,
  0x4c, 0x44, 0x08, 0x34, 0xc3, 0x42, 0x98, 0xc0, 0x35, 0x5c, 0xbe, 0xf3,
  0xf8, 0x11, 0x60, 0x6d, 0x54, 0x51, 0x10, 0x51, 0xe9, 0x00, 0x83, 0x5f,
  0xdc, 0xb6, 0x0d, 0x60, 0xc3, 0xe5, 0x3b, 0x8f, 0x1f, 0x01, 0xd6, 0x46,
  0x15, 0x05, 0x11, 0xb1, 0x93, 0x13, 0x11, 0x7e, 0x71, 0xdb, 0x16, 0x20,
  0x0d, 0x97, 0xef, 0x3c, 0xfe, 0x74, 0x44, 0x04, 0x30, 0x88, 0x83, 0x8f,
  0xdc, 0xb6, 0x01, 0x84, 0x01, 0x03, 0x28, 0xc4, 0xcf, 0x50, 0xcb, 0x84,
  0x48, 0x02, 0xb0, 0x10, 0x3f, 0xf5, 0x44, 0x84, 0xf4, 0x4b, 0x00, 0xf3,
  0x2c, 0xc4, 0x6f, 0x44, 0xc8, 0x00, 0x00, 0x00, 0x61, 0x20, 0x00, 0x00,
  0x79, 0x00, 0x00, 0x00, 0x13, 0x04, 0x46, 0x2c, 0x10, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x34, 0xca, 0x52, 0xa0, 0xec, 0x4a, 0xae, 0x74,
  0x03, 0x0a, 0x53, 0xa0, 0x0c, 0x08, 0x15, 0x41, 0x09, 0x90, 0x19, 0x23,
  0x00, 0x41, 0x10, 0x04, 0xc1, 0x60, 0x8c, 0x00, 0x04, 0x41, 0x10, 0xff,
  0x85, 0x31, 0x02, 0x10, 0x04, 0x41, 0xf8, 0x9b, 0x01, 0x00, 0x00, 0x00,
  0x23, 0x06, 0x09, 0x00, 0x82, 0x60, 0x30, 0x61, 0x4a, 0x71, 0x5d, 0xd0,
  0x88, 0x41, 0x02, 0x80, 0x20, 0x18, 0x4c, 0xd9, 0x62, 0x54, 0x55, 0x34,
  0x62, 0x90, 0x00, 0x20, 0x08, 0x06, 0x93, 0xc6, 0x20, 0x96, 0x25, 0x8d,
  0x18, 0x18, 0x00, 0x08, 0x82, 0x01, 0xf1, 0x31, 0xd7, 0x88, 0x81, 0x01,
  0x80, 0x20, 0x18, 0x10, 0x60, 0xd0, 0x6c, 0x23, 0x06, 0x07, 0x00, 0x82,
  0x60, 0x00, 0x79, 0xd0, 0x90, 0x8d, 0x26, 0x04, 0xc1, 0x70, 0x03, 0x11,
  0x9c, 0xc1, 0x68, 0xc2, 0x20, 0x0c, 0x37, 0x14, 0xc1, 0x19, 0xd4, 0x10,
  0xec, 0x2c, 0x43, 0x11, 0x04, 0x23, 0x06, 0x0a, 0x00, 0x82, 0x60, 0xb0,
  0x98, 0xc1, 0xa5, 0x7c, 0x1f, 0xd4, 0x8d, 0x26, 0x04, 0xc0, 0x88, 0x81,
  0x02, 0x80, 0x20, 0x18, 0x2c, 0x68, 0x90, 0x31, 0x64, 0x10, 0x06, 0xd2,
  0x37, 0x9a, 0x10, 0x00, 0x23, 0x06, 0x0a, 0x00, 0x82, 0x60, 0xb0, 0xa8,
  0xc1, 0xe6, 0x94, 0xc1, 0x18, 0x50, 0x61, 0x30, 0x9a, 0x10, 0x00, 0x23,
  0x06, 0x0a, 0x00, 0x82, 0x60, 0xb0, 0xb0, 0x41, 0x07, 0x61, 0x65, 0x60,
  0x8d, 0xc1, 0x68, 0x42, 0x00, 0x8c, 0x18, 0x28, 0x00, 0x08, 0x82, 0xc1,
  0xe2, 0x06, 0x9f, 0x54, 0x06, 0x67, 0x80, 0x95, 0xc1, 0x68, 0x42, 0x00,
  0x9c, 0x64, 0xcc, 0x02, 0x0c, 0x3e, 0x16, 0x2c, 0xf0, 0xb1, 0x80, 0xa0,
  0x8f, 0x05, 0x49, 0x7c, 0x8e, 0x32, 0x66, 0x42, 0x40, 0x9f, 0xc3, 0x8c,
  0x59, 0xf0, 0xc1, 0xc7, 0x02, 0x08, 0x3e, 0x16, 0x2c, 0xf4, 0xb1, 0xc0,
  0x89, 0xcf, 0x65, 0xc6, 0x4c, 0x08, 0xe8, 0x63, 0xc8, 0x19, 0xc8, 0xc7,
  0x82, 0x33, 0x80, 0x8f, 0x0d, 0x69, 0x20, 0x1f, 0x0b, 0xd2, 0x00, 0x3e,
  0xa3, 0x09, 0x62, 0x00, 0x0c, 0x37, 0x04, 0x7c, 0x60, 0x06, 0xb3, 0x0c,
  0x42, 0x11, 0xcc, 0x12, 0x0c, 0x03, 0x15, 0x43, 0x1a, 0x0c, 0xfa, 0x20,
  0x54, 0x10, 0x0a, 0x37, 0x62, 0x70, 0x00, 0x20, 0x08, 0x06, 0x10, 0x29,
  0xd8, 0x41, 0x1a, 0xfc, 0xc1, 0x68, 0x42, 0x10, 0xd8, 0x10, 0x88, 0x60,
  0x34, 0x61, 0x10, 0x4c, 0x08, 0x44, 0x60, 0x04, 0x1b, 0x88, 0xa0, 0x04,
  0x37, 0x80, 0x0a, 0x04, 0x18, 0x31, 0x70, 0x00, 0x10, 0x04, 0x83, 0xa6,
  0x15, 0xf6, 0x60, 0x0e, 0x82, 0x53, 0x78, 0xe8, 0x80, 0x0e, 0xe8, 0x00,
  0x0f, 0x4a, 0x61, 0xc4, 0xe0, 0x00, 0x40, 0x10, 0x0c, 0x20, 0x56, 0xf0,
  0x83, 0x38, 0x38, 0x85, 0xd1, 0x84, 0x20, 0xb0, 0x25, 0x10, 0xc1, 0x68,
  0xc2, 0x20, 0x98, 0x10, 0x88, 0xc0, 0x04, 0x42, 0x04, 0x56, 0xd4, 0x81,
  0x08, 0x6a, 0xb8, 0x03, 0xa8, 0x60, 0x80, 0x0a, 0x06, 0x18, 0x31, 0x70,
  0x00, 0x10, 0x04, 0x83, 0xe6, 0x16, 0x4a, 0xa1, 0x0f, 0x82, 0x58, 0xb8,
  0xfc, 0xc0, 0x0f, 0xfc, 0x40, 0x14, 0x5e, 0xc1, 0x28, 0x5a, 0x80, 0xc1,
  0x88, 0xc1, 0x01, 0x80, 0x20, 0x18, 0x40, 0xb7, 0x90, 0x0a, 0x7c, 0x20,
  0x0b, 0xa3, 0x09, 0x01, 0x30, 0xdc, 0x30, 0x04, 0x68, 0x30, 0xcb, 0x30,
  0x10, 0xc1, 0x2c, 0x41, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};