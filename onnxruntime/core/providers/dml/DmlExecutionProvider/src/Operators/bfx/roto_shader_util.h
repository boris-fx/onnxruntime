#ifndef BFX_HLSL_ROTO_SHADER_UTIL
#define BFX_HLSL_ROTO_SHADER_UTIL

// HLSL-side helpers shared by the roto shaders. Included from the .hlsl files only - the C++ side
// has no use for any of it.
//
// THE NUMERIC CONSTANTS ARE A CONTRACT, not tuning. Every one of them has to match
// python/roto/refine_graph.py (_SEG_LEN2_MIN / _DIST_EPS / _OCC_MAX and the crossing test's
// epsilon) and therefore the CUDA and CPU kernels, which match it already. The reference these
// are all checked against is refine_graph.occ_grad_native / union_forward_native, and the test
// cases in <Target>_tests were produced from it.
//
// CLAMPS, NEVER ADDED EPSILONS - except kCrossEps, which IS an add in the reference and must stay
// one. See the long comment in refine_graph.py about onnxscript's add_0_rule deleting small
// addends, and what one eliminated sqrt guard cost.

#define ROTO_SEG_LEN2_MIN 1e-8f
#define ROTO_DIST_EPS     1e-12f
#define ROTO_OCC_MAX      0.999999f
#define ROTO_CROSS_EPS    1e-12f

// FLT_MAX. Spelled as a bit pattern because HLSL has no <float.h>, and a decimal literal that
// close to the top of the range is at the mercy of the compiler's parser.
#define ROTO_FLT_MAX asfloat(0x7F7FFFFF)


// ---------------------------------------------------------------------------------------------
// Narrow tensor element reads.
//
// HLSL has no 8- or 16-bit buffer element type (RWStructuredBuffer<uint8_t> does not exist, and
// 16-bit types need -enable-16bit-types and still do not give you a byte). So every uint8 and
// float16 tensor is bound as RWStructuredBuffer<uint> and the element is extracted by shift.
//
// SAFE AT THE TAIL even when the element count is not a multiple of 4: each ORT tensor is its own
// ID3D12Resource starting at offset 0 (AllocationInfo::GetResource, and BucketizedBufferAllocator
// pools whole resources by size class rather than sub-allocating), and
// d3d_util.h::CreateD3D12ResourceOfByteSize rounds every allocation up to 4 bytes, so the dword
// containing the last byte is always inside the resource.
// ---------------------------------------------------------------------------------------------

// element i of a uint8 tensor
uint roto_load_u8(RWStructuredBuffer<uint> buf, uint i) {
    return (buf[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu;
}

// the raw 16 bits of element i of a float16 tensor. NOT converted to float, deliberately: the
// only question the loss ever asks of the weight is `> 0`, which is a sign-bit-and-mantissa test
// on the bit pattern. Keeping it integral is what lets this shader - like its CUDA and CPU twins -
// contain no float16 support at all.
uint roto_load_f16_bits(RWStructuredBuffer<uint> buf, uint i) {
    return (buf[i >> 1] >> ((i & 1u) * 16u)) & 0xFFFFu;
}

// `wgt > 0`, on the bit pattern: positive means the sign bit is clear and the rest is non-zero.
bool roto_f16_positive(uint bits) {
    return (bits & 0x8000u) == 0u && (bits & 0x7FFFu) != 0u;
}


// ---------------------------------------------------------------------------------------------
// NOTE ON FLOAT ATOMICS, which are NOT here.
//
// HLSL has none through SM 6.x - InterlockedAdd is integer only - so where the CUDA kernels say
// atomicAdd(float*) a compare-and-swap loop stands in. Those loops live in roto_occ_grad.hlsl
// rather than in this shared header, and deliberately: they need the concrete global (an
// RWByteAddressBuffer) and the concrete groupshared array as their destination.
//
// The RWStructuredBuffer parameters above are safe because shader_util.h already passes resources
// to functions the same way and it compiles; an RWByteAddressBuffer parameter is less well-trodden
// ground, and there is no reason to find out the hard way for a function with one caller.
//
// The cost of a CAS loop under contention - every loser re-reads and retries - is the whole reason
// roto_occ_grad.hlsl funnels through wave and groupshared aggregation, and writes its per-block
// results with plain stores rather than atomics. See roto_occ_grad_hlsl.h.
// ---------------------------------------------------------------------------------------------

#endif // BFX_HLSL_ROTO_SHADER_UTIL
