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
; shader hash: e4a7fbc27c516f7f35e0557ef716c35b
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
;   [28 x i8] (type annotation not present)
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
;                                       UAV  struct         r/w      U2             u2     1
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.ResRet.f32 = type { float, float, float, float, i32 }
%"class.RWStructuredBuffer<float>" = type { float }
%Constants = type { i32, i32, i32, i32, i32, i32, i32 }

define void @warp_flow() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 2, i32 2, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 1, i32 1, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %4 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %5 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %6 = call i32 @dx.op.threadId.i32(i32 93, i32 1)  ; ThreadId(component)
  %7 = call i32 @dx.op.threadId.i32(i32 93, i32 2)  ; ThreadId(component)
  %8 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %4, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %9 = extractvalue %dx.types.CBufRet.i32 %8, 0
  %10 = sdiv i32 %7, %9
  %11 = srem i32 %7, %9
  %12 = extractvalue %dx.types.CBufRet.i32 %8, 2
  %13 = icmp sge i32 %5, %12
  %14 = extractvalue %dx.types.CBufRet.i32 %8, 1
  %15 = icmp sge i32 %6, %14
  %16 = or i1 %13, %15
  %17 = icmp slt i32 %9, 0
  %18 = or i1 %16, %17
  %19 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %4, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %20 = extractvalue %dx.types.CBufRet.i32 %19, 3
  %21 = icmp sge i32 %10, %20
  %22 = or i1 %18, %21
  br i1 %22, label %201, label %23

; <label>:23                                      ; preds = %0
  %24 = shl i32 %10, 1
  %25 = mul nsw i32 %24, %14
  %26 = mul nsw i32 %25, %12
  %27 = mul nsw i32 %12, %14
  %28 = mul nsw i32 %12, %6
  %29 = add i32 %26, %5
  %30 = add i32 %29, %27
  %31 = add i32 %30, %28
  %32 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %2, i32 %31, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %33 = extractvalue %dx.types.ResRet.f32 %32, 0
  %34 = add i32 %29, %28
  %35 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %2, i32 %34, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %36 = extractvalue %dx.types.ResRet.f32 %35, 0
  %37 = sitofp i32 %6 to float
  %38 = fadd fast float %33, %37
  %39 = sitofp i32 %5 to float
  %40 = fadd fast float %36, %39
  %41 = extractvalue %dx.types.CBufRet.i32 %19, 2
  %42 = icmp eq i32 %41, 0
  br i1 %42, label %43, label %56

; <label>:43                                      ; preds = %23
  %44 = sitofp i32 %14 to float
  %45 = fmul fast float %44, %38
  %46 = add nsw i32 %14, -1
  %47 = sitofp i32 %46 to float
  %48 = fdiv fast float %45, %47
  %49 = fadd fast float %48, -5.000000e-01
  %50 = sitofp i32 %12 to float
  %51 = fmul fast float %50, %40
  %52 = add nsw i32 %12, -1
  %53 = sitofp i32 %52 to float
  %54 = fdiv fast float %51, %53
  %55 = fadd fast float %54, -5.000000e-01
  br label %56

; <label>:56                                      ; preds = %43, %23
  %57 = phi float [ %38, %23 ], [ %49, %43 ]
  %58 = phi float [ %40, %23 ], [ %55, %43 ]
  %59 = mul nsw i32 %9, %10
  %60 = add i32 %11, %59
  %61 = mul i32 %14, %60
  %62 = mul i32 %61, %12
  %63 = extractvalue %dx.types.CBufRet.i32 %19, 1
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %65, label %140

; <label>:65                                      ; preds = %56
  %66 = fcmp fast ole float %57, -1.000000e+00
  %67 = sitofp i32 %14 to float
  %68 = fcmp fast ole float %67, %57
  %69 = or i1 %66, %68
  %70 = fcmp fast ole float %58, -1.000000e+00
  %71 = or i1 %70, %69
  %72 = sitofp i32 %12 to float
  %73 = fcmp fast ole float %72, %58
  %74 = or i1 %73, %71
  br i1 %74, label %197, label %75

; <label>:75                                      ; preds = %65
  %76 = call float @dx.op.unary.f32(i32 27, float %57)  ; Round_ni(value)
  %77 = fptosi float %76 to i32
  %78 = call float @dx.op.unary.f32(i32 27, float %58)  ; Round_ni(value)
  %79 = fptosi float %78 to i32
  %80 = add nsw i32 %77, 1
  %81 = add nsw i32 %79, 1
  %82 = sitofp i32 %77 to float
  %83 = fsub fast float %57, %82
  %84 = sitofp i32 %79 to float
  %85 = fsub fast float %58, %84
  %86 = fsub fast float 1.000000e+00, %83
  %87 = fsub fast float 1.000000e+00, %85
  %88 = or i32 %79, %77
  %89 = icmp sgt i32 %88, -1
  br i1 %89, label %90, label %96

; <label>:90                                      ; preds = %75
  %91 = mul nsw i32 %77, %12
  %92 = add i32 %79, %62
  %93 = add i32 %92, %91
  %94 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %93, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %95 = extractvalue %dx.types.ResRet.f32 %94, 0
  br label %96

; <label>:96                                      ; preds = %90, %75
  %97 = phi float [ %95, %90 ], [ 0.000000e+00, %75 ]
  %98 = icmp sgt i32 %77, -1
  %99 = add nsw i32 %12, -1
  %100 = icmp slt i32 %79, %99
  %101 = and i1 %98, %100
  br i1 %101, label %102, label %108

; <label>:102                                     ; preds = %96
  %103 = mul nsw i32 %77, %12
  %104 = add i32 %103, %62
  %105 = add i32 %104, %81
  %106 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %105, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %107 = extractvalue %dx.types.ResRet.f32 %106, 0
  br label %108

; <label>:108                                     ; preds = %102, %96
  %109 = phi float [ %107, %102 ], [ 0.000000e+00, %96 ]
  %110 = add nsw i32 %14, -1
  %111 = icmp slt i32 %77, %110
  %112 = icmp sgt i32 %79, -1
  %113 = and i1 %111, %112
  br i1 %113, label %114, label %120

; <label>:114                                     ; preds = %108
  %115 = mul nsw i32 %80, %12
  %116 = add i32 %79, %62
  %117 = add i32 %116, %115
  %118 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %117, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %119 = extractvalue %dx.types.ResRet.f32 %118, 0
  br label %120

; <label>:120                                     ; preds = %114, %108
  %121 = phi float [ %119, %114 ], [ 0.000000e+00, %108 ]
  %122 = and i1 %111, %100
  br i1 %122, label %123, label %129

; <label>:123                                     ; preds = %120
  %124 = mul nsw i32 %80, %12
  %125 = add i32 %81, %62
  %126 = add i32 %125, %124
  %127 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %126, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %128 = extractvalue %dx.types.ResRet.f32 %127, 0
  br label %129

; <label>:129                                     ; preds = %123, %120
  %130 = phi float [ %128, %123 ], [ 0.000000e+00, %120 ]
  %131 = fmul fast float %97, %87
  %132 = fmul fast float %109, %85
  %133 = fmul fast float %121, %87
  %134 = fmul fast float %130, %85
  %135 = fadd fast float %134, %133
  %136 = fmul fast float %135, %83
  %137 = fadd fast float %132, %131
  %138 = fmul fast float %137, %86
  %139 = fadd fast float %138, %136
  br label %197

; <label>:140                                     ; preds = %56
  %141 = call float @dx.op.unary.f32(i32 27, float %57)  ; Round_ni(value)
  %142 = fptosi float %141 to i32
  %143 = call float @dx.op.unary.f32(i32 27, float %58)  ; Round_ni(value)
  %144 = fptosi float %143 to i32
  %145 = add nsw i32 %142, 1
  %146 = add nsw i32 %144, 1
  %147 = sitofp i32 %142 to float
  %148 = fsub fast float %57, %147
  %149 = sitofp i32 %144 to float
  %150 = fsub fast float %58, %149
  %151 = fsub fast float 1.000000e+00, %148
  %152 = fsub fast float 1.000000e+00, %150
  %153 = icmp slt i32 %142, 0
  %154 = select i1 %153, i32 0, i32 %142
  %155 = icmp sge i32 %154, %14
  %156 = add nsw i32 %14, -1
  %157 = select i1 %155, i32 %156, i32 %154
  %158 = icmp slt i32 %144, 0
  %159 = select i1 %158, i32 0, i32 %144
  %160 = icmp sge i32 %159, %12
  %161 = add nsw i32 %12, -1
  %162 = select i1 %160, i32 %161, i32 %159
  %163 = icmp slt i32 %145, 0
  %164 = select i1 %163, i32 0, i32 %145
  %165 = icmp sge i32 %164, %14
  %166 = select i1 %165, i32 %156, i32 %164
  %167 = icmp slt i32 %146, 0
  %168 = select i1 %167, i32 0, i32 %146
  %169 = icmp sge i32 %168, %12
  %170 = select i1 %169, i32 %161, i32 %168
  %171 = mul nsw i32 %157, %12
  %172 = add nsw i32 %171, %62
  %173 = add nsw i32 %172, %162
  %174 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %173, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %175 = extractvalue %dx.types.ResRet.f32 %174, 0
  %176 = add nsw i32 %172, %170
  %177 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %176, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %178 = extractvalue %dx.types.ResRet.f32 %177, 0
  %179 = mul nsw i32 %166, %12
  %180 = add i32 %162, %62
  %181 = add i32 %180, %179
  %182 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %181, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %183 = extractvalue %dx.types.ResRet.f32 %182, 0
  %184 = add nsw i32 %179, %62
  %185 = add nsw i32 %184, %170
  %186 = call %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32 139, %dx.types.Handle %3, i32 %185, i32 0, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %187 = extractvalue %dx.types.ResRet.f32 %186, 0
  %188 = fmul fast float %175, %152
  %189 = fmul fast float %178, %150
  %190 = fmul fast float %183, %152
  %191 = fmul fast float %187, %150
  %192 = fadd fast float %191, %190
  %193 = fmul fast float %192, %148
  %194 = fadd fast float %189, %188
  %195 = fmul fast float %194, %151
  %196 = fadd fast float %195, %193
  br label %197

; <label>:197                                     ; preds = %140, %129, %65
  %198 = phi float [ %196, %140 ], [ %139, %129 ], [ 0.000000e+00, %65 ]
  %199 = add i32 %62, %5
  %200 = add i32 %199, %28
  call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %1, i32 %200, i32 0, float %198, float undef, float undef, float undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %201

; <label>:201                                     ; preds = %197, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind readonly
declare %dx.types.ResRet.f32 @dx.op.rawBufferLoad.f32(i32, %dx.types.Handle, i32, i32, i8, i32) #1

; Function Attrs: nounwind
declare void @dx.op.rawBufferStore.f32(i32, %dx.types.Handle, i32, i32, float, float, float, float, i8, i32) #2

; Function Attrs: nounwind readnone
declare float @dx.op.unary.f32(i32, float) #0

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
!dx.entryPoints = !{!12}

!0 = !{!"clang version 3.7 (tags/RELEASE_370/final)"}
!1 = !{i32 1, i32 2}
!2 = !{i32 1, i32 6}
!3 = !{!"cs", i32 6, i32 2}
!4 = !{null, !5, !10, null}
!5 = !{!6, !8, !9}
!6 = !{i32 0, %"class.RWStructuredBuffer<float>"* undef, !"", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!7 = !{i32 1, i32 4}
!8 = !{i32 1, %"class.RWStructuredBuffer<float>"* undef, !"", i32 0, i32 1, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!9 = !{i32 2, %"class.RWStructuredBuffer<float>"* undef, !"", i32 0, i32 2, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!10 = !{!11}
!11 = !{i32 0, %Constants* undef, !"", i32 0, i32 0, i32 1, i32 28, null}
!12 = !{void ()* @warp_flow, !"warp_flow", null, !4, !13}
!13 = !{i32 0, i64 16, i32 4, !14}
!14 = !{i32 16, i32 16, i32 1}

#endif

const unsigned char g_warp_flow[] = {
  0x44, 0x58, 0x42, 0x43, 0x4f, 0x6b, 0xc2, 0x3a, 0x08, 0x9f, 0xc7, 0xaa,
  0xf5, 0x53, 0xd7, 0x6a, 0x57, 0x22, 0x48, 0xd8, 0x01, 0x00, 0x00, 0x00,
  0x2c, 0x0b, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x18, 0x01, 0x00, 0x00, 0x34, 0x01, 0x00, 0x00, 0x53, 0x46, 0x49, 0x30,
  0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x49, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x4f, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x50, 0x53, 0x56, 0x30,
  0xa8, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x48, 0x41, 0x53, 0x48, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xe4, 0xa7, 0xfb, 0xc2, 0x7c, 0x51, 0x6f, 0x7f,
  0x35, 0xe0, 0x55, 0x7e, 0xf7, 0x16, 0xc3, 0x5b, 0x44, 0x58, 0x49, 0x4c,
  0xf0, 0x09, 0x00, 0x00, 0x62, 0x00, 0x05, 0x00, 0x7c, 0x02, 0x00, 0x00,
  0x44, 0x58, 0x49, 0x4c, 0x02, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0xd8, 0x09, 0x00, 0x00, 0x42, 0x43, 0xc0, 0xde, 0x21, 0x0c, 0x00, 0x00,
  0x73, 0x02, 0x00, 0x00, 0x0b, 0x82, 0x20, 0x00, 0x02, 0x00, 0x00, 0x00,
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
  0x36, 0x00, 0x00, 0x00, 0x32, 0x22, 0x48, 0x09, 0x20, 0x64, 0x85, 0x04,
  0x93, 0x22, 0xa4, 0x84, 0x04, 0x93, 0x22, 0xe3, 0x84, 0xa1, 0x90, 0x14,
  0x12, 0x4c, 0x8a, 0x8c, 0x0b, 0x84, 0xa4, 0x4c, 0x10, 0x78, 0x23, 0x00,
  0x25, 0x00, 0x14, 0xe6, 0x08, 0xc0, 0xa0, 0x0c, 0x63, 0x0c, 0x22, 0x33,
  0x00, 0x37, 0x0d, 0x97, 0x3f, 0x61, 0x0f, 0x21, 0xf9, 0x2b, 0x21, 0xad,
  0xc4, 0xe4, 0x17, 0xb7, 0x8d, 0x0a, 0x63, 0x8c, 0x19, 0x73, 0x04, 0x08,
  0xa1, 0x7b, 0x86, 0xcb, 0x9f, 0xb0, 0x87, 0x90, 0xfc, 0x10, 0x68, 0x86,
  0x85, 0x40, 0x41, 0x2a, 0xc7, 0x19, 0x6a, 0x0c, 0x34, 0x68, 0x95, 0x05,
  0x0c, 0x35, 0x86, 0x31, 0xc6, 0xa0, 0x41, 0xad, 0x0c, 0x66, 0x18, 0x7a,
  0x47, 0x0d, 0x97, 0x3f, 0x61, 0x0f, 0x21, 0xf9, 0xdc, 0x46, 0x15, 0x2b,
  0x31, 0xf9, 0xc8, 0x6d, 0x23, 0x62, 0x8c, 0x31, 0x0a, 0x11, 0x87, 0x1a,
  0x24, 0xe7, 0x08, 0x82, 0x62, 0xa8, 0x81, 0xc6, 0xa0, 0x54, 0x07, 0x02,
  0x66, 0xfa, 0xc6, 0x81, 0x1d, 0xc2, 0x61, 0x1e, 0xe6, 0xc1, 0x0d, 0x64,
  0xe1, 0x16, 0x66, 0x81, 0x1e, 0xe4, 0xa1, 0x1e, 0xc6, 0x81, 0x1e, 0xea,
  0x41, 0x1e, 0xca, 0x81, 0x1c, 0x44, 0xa1, 0x1e, 0xcc, 0xc1, 0x1c, 0xca,
  0x41, 0x1e, 0xf8, 0xc0, 0x1c, 0xd8, 0xe1, 0x1d, 0xc2, 0x81, 0x1e, 0xfc,
  0x00, 0x05, 0x86, 0xf0, 0x25, 0x9c, 0xd3, 0x48, 0x13, 0xd0, 0x4c, 0x12,
  0x3a, 0xc6, 0x18, 0x63, 0x8c, 0x41, 0x7a, 0x8e, 0x00, 0x14, 0xa6, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x13, 0x14, 0x72, 0xc0, 0x87, 0x74, 0x60, 0x87,
  0x36, 0x68, 0x87, 0x79, 0x68, 0x03, 0x72, 0xc0, 0x87, 0x0d, 0xaf, 0x50,
  0x0e, 0x6d, 0xd0, 0x0e, 0x7a, 0x50, 0x0e, 0x6d, 0x00, 0x0f, 0x7a, 0x30,
  0x07, 0x72, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x71, 0xa0,
  0x07, 0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x78, 0xa0, 0x07, 0x73, 0x20,
  0x07, 0x6d, 0x90, 0x0e, 0x71, 0x60, 0x07, 0x7a, 0x30, 0x07, 0x72, 0xd0,
  0x06, 0xe9, 0x30, 0x07, 0x72, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x90,
  0x0e, 0x76, 0x40, 0x07, 0x7a, 0x60, 0x07, 0x74, 0xd0, 0x06, 0xe6, 0x10,
  0x07, 0x76, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x60, 0x0e, 0x73, 0x20,
  0x07, 0x7a, 0x30, 0x07, 0x72, 0xd0, 0x06, 0xe6, 0x60, 0x07, 0x74, 0xa0,
  0x07, 0x76, 0x40, 0x07, 0x6d, 0xe0, 0x0e, 0x78, 0xa0, 0x07, 0x71, 0x60,
  0x07, 0x7a, 0x30, 0x07, 0x72, 0xa0, 0x07, 0x76, 0x40, 0x07, 0x43, 0x9e,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x86,
  0x3c, 0x04, 0x10, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x0c, 0x79, 0x16, 0x20, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x18, 0xf2, 0x34, 0x40, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x30, 0xe4, 0x79, 0x80, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x60, 0xc8, 0x23, 0x01, 0x01, 0x20, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xc0, 0x90, 0xa7, 0x02, 0x02, 0x40, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x2c, 0x10, 0x0b, 0x00, 0x00, 0x00,
  0x32, 0x1e, 0x98, 0x14, 0x19, 0x11, 0x4c, 0x90, 0x8c, 0x09, 0x26, 0x47,
  0xc6, 0x04, 0x43, 0x1a, 0x25, 0x50, 0x04, 0xc5, 0x30, 0x02, 0x50, 0x18,
  0x85, 0x50, 0x38, 0x05, 0x42, 0x74, 0x04, 0x80, 0x78, 0x81, 0xd0, 0x9e,
  0x01, 0xa0, 0x3c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00,
  0x47, 0x00, 0x00, 0x00, 0x1a, 0x03, 0x4c, 0x90, 0x46, 0x02, 0x13, 0x44,
  0x35, 0x18, 0x63, 0x0b, 0x73, 0x3b, 0x03, 0xb1, 0x2b, 0x93, 0x9b, 0x4b,
  0x7b, 0x73, 0x03, 0x99, 0x71, 0xb9, 0x01, 0x41, 0xa1, 0x0b, 0x3b, 0x9b,
  0x7b, 0x91, 0x2a, 0x62, 0x2a, 0x0a, 0x9a, 0x2a, 0xfa, 0x9a, 0xb9, 0x81,
  0x79, 0x31, 0x4b, 0x73, 0x0b, 0x63, 0x4b, 0xd9, 0x10, 0x04, 0x13, 0x84,
  0xe1, 0x98, 0x20, 0x0c, 0xc8, 0x06, 0x61, 0x20, 0x26, 0x08, 0x43, 0xb2,
  0x41, 0x18, 0x0c, 0x0a, 0x63, 0x73, 0x1b, 0x06, 0xc4, 0x20, 0x26, 0x08,
  0x83, 0x32, 0x41, 0xc8, 0x24, 0x02, 0x13, 0x84, 0x61, 0x99, 0x20, 0x50,
  0xcf, 0x04, 0x61, 0x60, 0x36, 0x08, 0xc3, 0xb3, 0x61, 0x51, 0x16, 0x46,
  0x51, 0x86, 0xc6, 0x71, 0x1c, 0x68, 0xc3, 0x32, 0x2c, 0x8c, 0x32, 0x0c,
  0x8d, 0xe3, 0x38, 0xd0, 0x86, 0x85, 0x58, 0x18, 0x85, 0x18, 0x1a, 0xc7,
  0x71, 0xa0, 0x0d, 0x43, 0x24, 0x4d, 0x13, 0x84, 0x2d, 0x9a, 0x20, 0x0c,
  0xcd, 0x06, 0x44, 0xa9, 0x18, 0x45, 0x19, 0x2c, 0x60, 0x43, 0x70, 0x6d,
  0x20, 0x00, 0x0a, 0x03, 0x26, 0x08, 0x02, 0xc0, 0xe4, 0x2e, 0x4c, 0x0e,
  0xee, 0xcb, 0x8c, 0xed, 0xed, 0x6e, 0x82, 0xc0, 0x41, 0x13, 0x84, 0xc1,
  0xd9, 0x30, 0x74, 0xdd, 0xb0, 0x81, 0x50, 0xb8, 0xc7, 0xdb, 0x50, 0x68,
  0x1b, 0x90, 0x7d, 0x55, 0xd8, 0xd8, 0xec, 0xda, 0x5c, 0xd2, 0xc8, 0xca,
  0xdc, 0xe8, 0xa6, 0x04, 0x41, 0x15, 0x32, 0x3c, 0x17, 0xbb, 0x32, 0xb9,
  0xb9, 0xb4, 0x37, 0xb7, 0x29, 0x01, 0xd1, 0x84, 0x0c, 0xcf, 0xc5, 0x2e,
  0x8c, 0xcd, 0xae, 0x4c, 0x6e, 0x4a, 0x60, 0xd4, 0x21, 0xc3, 0x73, 0x99,
  0x43, 0x0b, 0x23, 0x2b, 0x93, 0x6b, 0x7a, 0x23, 0x2b, 0x63, 0x9b, 0x12,
  0x20, 0x65, 0xc8, 0xf0, 0x5c, 0xe4, 0xca, 0xe6, 0xde, 0xea, 0xe4, 0xc6,
  0xca, 0xe6, 0xa6, 0x04, 0x58, 0x1d, 0x32, 0x3c, 0x97, 0x32, 0x37, 0x3a,
  0xb9, 0x3c, 0xa8, 0xb7, 0x34, 0x37, 0xba, 0xb9, 0x29, 0xc1, 0x07, 0x00,
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
  0x1e, 0x00, 0x00, 0x00, 0x56, 0xb0, 0x0d, 0x97, 0xef, 0x3c, 0xbe, 0x10,
  0x50, 0x45, 0x41, 0x44, 0xa5, 0x03, 0x0c, 0x25, 0x61, 0x00, 0x02, 0xe6,
  0x23, 0xb7, 0x6d, 0x06, 0xd2, 0x70, 0xf9, 0xce, 0xe3, 0x0b, 0x11, 0x01,
  0x4c, 0x44, 0x08, 0x34, 0xc3, 0x42, 0x98, 0xc0, 0x35, 0x5c, 0xbe, 0xf3,
  0xf8, 0x11, 0x60, 0x6d, 0x54, 0x51, 0x10, 0x51, 0xe9, 0x00, 0x83, 0x5f,
  0xdc, 0xb6, 0x0d, 0x60, 0xc3, 0xe5, 0x3b, 0x8f, 0x1f, 0x01, 0xd6, 0x46,
  0x15, 0x05, 0x11, 0xb1, 0x93, 0x13, 0x11, 0x7e, 0x71, 0xdb, 0x16, 0x20,
  0x0d, 0x97, 0xef, 0x3c, 0xfe, 0x74, 0x44, 0x04, 0x30, 0x88, 0x83, 0x8f,
  0xdc, 0xb6, 0x11, 0x3c, 0xc3, 0xe5, 0x3b, 0x8f, 0x4f, 0x35, 0x40, 0x84,
  0xf9, 0xc5, 0x6d, 0x1b, 0x40, 0x62, 0x01, 0xd1, 0xf3, 0x17, 0x8b, 0x63,
  0x01, 0x00, 0x00, 0x00, 0x61, 0x20, 0x00, 0x00, 0x11, 0x01, 0x00, 0x00,
  0x13, 0x04, 0x51, 0x2c, 0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x34, 0xca, 0x52, 0xa0, 0x06, 0x4a, 0xae, 0x6c, 0x4a, 0x37, 0xa0, 0xec,
  0x0a, 0x53, 0x80, 0x50, 0x11, 0x94, 0x00, 0x99, 0x19, 0x80, 0x31, 0x02,
  0x10, 0x04, 0x41, 0xf8, 0x17, 0xc6, 0x08, 0x40, 0x10, 0x04, 0xf1, 0x5f,
  0x18, 0x23, 0x00, 0x41, 0x10, 0xc4, 0xbf, 0x11, 0x00, 0x00, 0x00, 0x00,
  0x23, 0x06, 0x09, 0x00, 0x82, 0x60, 0x50, 0x6d, 0x8c, 0x91, 0x65, 0xd2,
  0x88, 0x41, 0x02, 0x80, 0x20, 0x18, 0x54, 0x5c, 0x73, 0x6c, 0xdb, 0x34,
  0x62, 0x90, 0x00, 0x20, 0x08, 0x06, 0x55, 0xe7, 0x20, 0x59, 0x46, 0x8d,
  0x18, 0x24, 0x00, 0x08, 0x82, 0x41, 0xe5, 0x3d, 0x8a, 0xa6, 0x55, 0x23,
  0x06, 0x06, 0x00, 0x82, 0x60, 0x40, 0x90, 0x81, 0xb3, 0x8d, 0x18, 0x18,
  0x00, 0x08, 0x82, 0x01, 0x51, 0x06, 0xcf, 0x37, 0x62, 0x60, 0x00, 0x20,
  0x08, 0x06, 0x84, 0x19, 0x40, 0xdf, 0x88, 0xc1, 0x01, 0x80, 0x20, 0x18,
  0x48, 0x63, 0x00, 0x11, 0x61, 0x30, 0x9a, 0x10, 0x00, 0x35, 0x04, 0x54,
  0x84, 0x60, 0xa3, 0x09, 0x84, 0x30, 0xdc, 0x80, 0x04, 0x67, 0x30, 0x9a,
  0x60, 0x04, 0xc3, 0x0d, 0x49, 0x70, 0x06, 0x35, 0x04, 0x3b, 0xdc, 0x80,
  0x9c, 0x01, 0x1a, 0x94, 0x10, 0xec, 0x88, 0xc1, 0x01, 0x80, 0x20, 0x18,
  0x48, 0x6e, 0xb0, 0x3d, 0x69, 0x30, 0x9a, 0x10, 0x0c, 0xc3, 0x0d, 0x4b,
  0x70, 0x06, 0x45, 0x04, 0x3b, 0xcb, 0x00, 0x05, 0x41, 0x35, 0x70, 0x70,
  0x16, 0x28, 0x22, 0xb0, 0xa0, 0x11, 0x81, 0x39, 0x8c, 0x08, 0xec, 0xa9,
  0x44, 0x50, 0xc3, 0x05, 0x15, 0x0c, 0x50, 0xc1, 0x00, 0x23, 0x06, 0x0a,
  0x00, 0x82, 0x60, 0xb0, 0xf4, 0x01, 0x1b, 0x74, 0x41, 0x1d, 0x90, 0xc1,
  0x1c, 0x8c, 0x26, 0x04, 0x40, 0x15, 0x06, 0x8c, 0x18, 0x28, 0x00, 0x08,
  0x82, 0xc1, 0x02, 0x0a, 0x6f, 0x00, 0x06, 0x01, 0x1e, 0x9c, 0x81, 0x1d,
  0x8c, 0x26, 0x04, 0xc0, 0x79, 0xc6, 0xac, 0x08, 0xe0, 0x73, 0x61, 0x60,
  0xcc, 0x88, 0x00, 0x3e, 0xa3, 0x09, 0x95, 0x30, 0xdc, 0x10, 0xfc, 0x01,
  0x18, 0xcc, 0x32, 0x08, 0x43, 0x70, 0x9c, 0x31, 0x0b, 0x0c, 0xf9, 0x98,
  0x87, 0x07, 0x20, 0xb8, 0xc0, 0x98, 0x0d, 0x01, 0x7d, 0x2c, 0x90, 0x03,
  0xf8, 0x1c, 0x19, 0x18, 0xb3, 0x40, 0x91, 0x8f, 0x99, 0x81, 0x1f, 0x80,
  0xe0, 0x02, 0x63, 0x36, 0x04, 0xf4, 0xb1, 0x00, 0x0f, 0xe0, 0x33, 0x4b,
  0x30, 0x0c, 0x54, 0x18, 0x62, 0x10, 0x38, 0xc2, 0x40, 0x85, 0x01, 0x06,
  0x01, 0x21, 0xd8, 0x1b, 0xb8, 0x81, 0x08, 0xca, 0x0d, 0x02, 0x28, 0x36,
  0x08, 0xa4, 0x82, 0x37, 0x90, 0xd1, 0x84, 0x34, 0x08, 0x86, 0x1b, 0x82,
  0x59, 0x00, 0x83, 0x59, 0x06, 0xc2, 0x09, 0x86, 0x23, 0x10, 0x50, 0x28,
  0xbe, 0x8b, 0x03, 0x63, 0xc3, 0x11, 0x81, 0x52, 0x7c, 0x35, 0x04, 0x3b,
  0x1c, 0xb1, 0x90, 0x42, 0xf1, 0x55, 0x20, 0xec, 0xe1, 0x81, 0xb1, 0xe1,
  0x88, 0xc0, 0x29, 0xbe, 0x0a, 0x86, 0x9d, 0x65, 0x78, 0x8a, 0x60, 0xc4,
  0xc0, 0x00, 0x40, 0x10, 0x0c, 0x9e, 0x71, 0x88, 0x85, 0xe8, 0x82, 0x41,
  0x23, 0x06, 0x06, 0x00, 0x82, 0x60, 0xf0, 0x94, 0xc3, 0x2c, 0x48, 0x17,
  0x0c, 0xb2, 0x81, 0x1c, 0x40, 0x60, 0x42, 0x39, 0x80, 0xe0, 0x0a, 0x63,
  0x86, 0x05, 0xf1, 0xb9, 0xc2, 0x98, 0x65, 0x41, 0x7c, 0x4c, 0x16, 0x86,
  0xf8, 0xd8, 0x2c, 0x08, 0xf1, 0xa9, 0x64, 0xd9, 0xe1, 0x86, 0x00, 0x1c,
  0xcc, 0x60, 0x96, 0xc1, 0x38, 0x02, 0x6b, 0x52, 0x41, 0x04, 0xc5, 0x6c,
  0x50, 0x81, 0x00, 0x23, 0x06, 0x0a, 0x00, 0x82, 0x60, 0xb0, 0xd8, 0x43,
  0x39, 0xd4, 0x42, 0xe0, 0x0e, 0xbd, 0xc0, 0x0e, 0xa3, 0x09, 0x01, 0x30,
  0x4b, 0x70, 0x0c, 0x54, 0x18, 0x82, 0x41, 0x17, 0xc5, 0x70, 0xc3, 0x74,
  0x0e, 0x66, 0x60, 0xb0, 0x80, 0x0e, 0x20, 0x18, 0x6e, 0x98, 0x02, 0x34,
  0xa8, 0x21, 0xd0, 0x59, 0x06, 0x24, 0x09, 0xec, 0x9a, 0x05, 0x11, 0x54,
  0x50, 0x06, 0x50, 0x41, 0x05, 0x23, 0x06, 0x0a, 0x00, 0x82, 0x60, 0xb0,
  0x80, 0xc4, 0x3b, 0xfc, 0x42, 0x80, 0x0f, 0xe7, 0x60, 0x0f, 0xa3, 0x09,
  0x01, 0x30, 0x4b, 0x90, 0x0c, 0x54, 0x18, 0x02, 0x82, 0x1a, 0x87, 0xdd,
  0x42, 0x3c, 0x80, 0x60, 0xb8, 0xc1, 0x0b, 0xd0, 0x60, 0xb8, 0xa1, 0x9b,
  0x07, 0x33, 0x28, 0x21, 0xd0, 0x59, 0x06, 0x65, 0x09, 0xcc, 0xeb, 0x05,
  0x11, 0x14, 0x18, 0xbc, 0x01, 0x54, 0x20, 0xc0, 0x88, 0x81, 0x02, 0x80,
  0x20, 0x18, 0x2c, 0x2a, 0x91, 0x0f, 0xe9, 0x10, 0x88, 0x44, 0x3c, 0x80,
  0xc4, 0x68, 0x42, 0x00, 0xcc, 0x12, 0x2c, 0x03, 0x15, 0x86, 0xa0, 0xf0,
  0x46, 0x52, 0x89, 0xa4, 0xb3, 0x0c, 0x4c, 0x13, 0x58, 0x19, 0x90, 0x83,
  0x08, 0xaa, 0x0c, 0xec, 0x00, 0x2a, 0x10, 0x60, 0xc4, 0x40, 0x01, 0x40,
  0x10, 0x0c, 0x96, 0x98, 0x00, 0x09, 0x78, 0x08, 0x52, 0x02, 0x1f, 0x4e,
  0x62, 0x34, 0x21, 0x00, 0x66, 0x09, 0x9a, 0x81, 0x0a, 0x43, 0x60, 0xd4,
  0x63, 0x31, 0x8e, 0x0c, 0xe4, 0x63, 0xd3, 0x19, 0xc8, 0xc7, 0x14, 0x33,
  0x90, 0x8f, 0x11, 0x69, 0x20, 0x1f, 0x0b, 0x04, 0xf8, 0x58, 0xd0, 0x06,
  0xf2, 0xb1, 0xc2, 0x80, 0x8f, 0x05, 0x6c, 0x20, 0x1f, 0x0b, 0x06, 0xf8,
  0xcc, 0x12, 0x3c, 0x23, 0x06, 0x06, 0x00, 0x82, 0x60, 0xf0, 0xec, 0x44,
  0x4a, 0xa4, 0xc2, 0x05, 0x83, 0x46, 0x0c, 0x0c, 0x00, 0x04, 0xc1, 0xe0,
  0xe9, 0x89, 0x95, 0x50, 0x85, 0x0b, 0x06, 0xd9, 0xc0, 0x13, 0x20, 0x30,
  0xa1, 0x27, 0x40, 0x70, 0x85, 0x31, 0x83, 0x85, 0x20, 0x3e, 0x57, 0x18,
  0xb3, 0x58, 0x08, 0xe2, 0x63, 0x2a, 0x31, 0xc4, 0xc7, 0x56, 0x42, 0x88,
  0xcf, 0x70, 0xc3, 0x12, 0x16, 0x68, 0x30, 0xdd, 0x20, 0x16, 0x4c, 0x30,
  0xdc, 0x10, 0xfc, 0xc3, 0x19, 0x18, 0x48, 0xe8, 0x04, 0x08, 0xa6, 0x1b,
  0x82, 0x41, 0x18, 0x6e, 0x70, 0xcc, 0x02, 0x0d, 0xa6, 0x1b, 0xce, 0xe2,
  0x09, 0x86, 0x1b, 0x02, 0x93, 0x38, 0x03, 0x3b, 0x89, 0x9f, 0x00, 0xc1,
  0x74, 0x43, 0x30, 0x08, 0xc3, 0x0d, 0xd2, 0x5a, 0xa0, 0xc1, 0x74, 0x03,
  0x5b, 0x4c, 0xc1, 0x70, 0x43, 0x90, 0x12, 0x67, 0x30, 0xdd, 0xa0, 0x08,
  0xc1, 0x70, 0x43, 0xf5, 0x16, 0x68, 0x30, 0xdd, 0x00, 0x17, 0x56, 0x30,
  0xdc, 0x10, 0xbc, 0xc4, 0x19, 0x4c, 0x37, 0x24, 0x42, 0x60, 0x4e, 0x4c,
  0x88, 0xc0, 0x82, 0x71, 0x00, 0x81, 0x05, 0x0b, 0x08, 0x46, 0x0c, 0x14,
  0x00, 0x04, 0xc1, 0x60, 0xf1, 0x8b, 0xb6, 0xe8, 0x89, 0xc0, 0x2e, 0xca,
  0x82, 0x2e, 0x46, 0x13, 0x02, 0xc0, 0x08, 0x03, 0x04, 0x23, 0x06, 0x0a,
  0x00, 0x82, 0x60, 0xb0, 0x84, 0x06, 0x5c, 0x80, 0x45, 0x90, 0x17, 0x68,
  0x71, 0x17, 0xa3, 0x09, 0x01, 0x60, 0x4d, 0x4e, 0x88, 0xa0, 0xa4, 0x75,
  0x80, 0x0a, 0x04, 0x18, 0x31, 0x50, 0x00, 0x10, 0x04, 0x83, 0xc5, 0x34,
  0xea, 0xa2, 0x2c, 0x02, 0xbf, 0x68, 0x0b, 0xbe, 0x18, 0x4d, 0x08, 0x00,
  0x2b, 0xde, 0x01, 0x04, 0x16, 0x3c, 0x20, 0x18, 0x31, 0x50, 0x00, 0x10,
  0x04, 0x83, 0x45, 0x35, 0xf2, 0x22, 0x2d, 0x02, 0xd1, 0x88, 0x0b, 0xd0,
  0x18, 0x4d, 0x08, 0x00, 0x6b, 0xc8, 0x40, 0x3e, 0xb6, 0x9c, 0x81, 0x7c,
  0xec, 0x30, 0x03, 0xf9, 0x18, 0x91, 0x06, 0xf2, 0xb1, 0x40, 0x80, 0x8f,
  0x05, 0x6d, 0x20, 0x1f, 0x2b, 0x0c, 0xf8, 0x58, 0xc0, 0x06, 0xf2, 0xb1,
  0x60, 0x80, 0xcf, 0x2c, 0xc1, 0x33, 0xd0, 0x61, 0x08, 0x8e, 0x3c, 0x34,
  0xee, 0x42, 0x54, 0x3f, 0xcc, 0x05, 0x54, 0xe0, 0x13, 0x30, 0x62, 0xe0,
  0x00, 0x20, 0x08, 0x06, 0xcd, 0x6d, 0x84, 0x46, 0x5e, 0x04, 0xb0, 0x31,
  0xf8, 0x85, 0x5f, 0xf8, 0xc5, 0x5f, 0xb8, 0xc6, 0x2c, 0x01, 0x84, 0x00,
  0x00, 0x00, 0x00, 0x00
};