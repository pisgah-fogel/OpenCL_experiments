#include <immintrin.h>
#include <vector>
#include <assert.h>

//https://www.felixcloutier.com/x86/addps

//_aligned_malloc / _aligned_free on msvc, or aligned_alloc / free on gcc & clang. 
// For GCC replace _mm256_loadu_ps with _mm256_load_ps

// CPUs support RAM access like this: "ymmword ptr [rax+64]"
// Using templates with offset int argument to make easier for compiler to emit good code.

// Multiply 8 floats by another 8 floats.
template<int offsetRegs>
inline __m256 mul8( const float* p1, const float* p2 )
{
    constexpr int lanes = offsetRegs * 8;
    const __m256 a = _mm256_loadu_ps( p1 + lanes );
    const __m256 b = _mm256_loadu_ps( p2 + lanes );
    return _mm256_mul_ps( a, b );
}

// Returns acc + ( p1 * p2 ), for 8-wide float lanes.
template<int offsetRegs>
inline __m256 fma8( __m256 acc, const float* p1, const float* p2 )
{
    constexpr int lanes = offsetRegs * 8;
    const __m256 a = _mm256_loadu_ps( p1 + lanes );
    const __m256 b = _mm256_loadu_ps( p2 + lanes );
    return _mm256_fmadd_ps( a, b, acc );
}

// Compute dot product of float vectors, using 8-wide FMA instructions.
float dotProductFma( const std::vector<float>& a, const std::vector<float>& b )
{
    assert( a.size() == b.size() );
    assert( 0 == ( a.size() % 32 ) );
    if( a.empty() )
        return 0.0f;

    const float* p1 = a.data();
    const float* const p1End = p1 + a.size();
    const float* p2 = b.data();

    // Process initial 32 values. Nothing to add yet, just multiplying.
    __m256 dot0 = mul8<0>( p1, p2 );
    __m256 dot1 = mul8<1>( p1, p2 );
    __m256 dot2 = mul8<2>( p1, p2 );
    __m256 dot3 = mul8<3>( p1, p2 );
    p1 += 8 * 4;
    p2 += 8 * 4;

    // Process the rest of the data.
    // The code uses FMA instructions to multiply + accumulate, consuming 32 values per loop iteration.
    // Unrolling manually for 2 reasons:
    // 1. To reduce data dependencies. With a single register, every loop iteration would depend on the previous result.
    // 2. Unrolled code checks for exit condition 4x less often, therefore more CPU cycles spent computing useful stuff.
    while( p1 < p1End )
    {
        dot0 = fma8<0>( dot0, p1, p2 );
        dot1 = fma8<1>( dot1, p1, p2 );
        dot2 = fma8<2>( dot2, p1, p2 );
        dot3 = fma8<3>( dot3, p1, p2 );
        p1 += 8 * 4;
        p2 += 8 * 4;
    }

    // Add 32 values into 8
    const __m256 dot01 = _mm256_add_ps( dot0, dot1 );
    const __m256 dot23 = _mm256_add_ps( dot2, dot3 );
    const __m256 dot0123 = _mm256_add_ps( dot01, dot23 );
    // Add 8 values into 4
    const __m128 r4 = _mm_add_ps( _mm256_castps256_ps128( dot0123 ), _mm256_extractf128_ps( dot0123, 1 ) );
    // Add 4 values into 2
    const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps( r4, r4 ) );
    // Add 2 lower values into the final result
    const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );
    // Return the lowest lane of the result vector.
    // The intrinsic below compiles into noop, modern compilers return floats in the lowest lane of xmm0 register.
    return _mm_cvtss_f32( r1 );
}