// Copyright, 2024. See LICENSE for copyright holder information.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

/// @file

#include "compilers.hh"
#include <Random123/features/compilerfeatures.h>

// this is for sincosf
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include <cmath>

#if !defined(R123_NO_SINCOS) && defined(__APPLE__)
/* MacOS X 10.10.5 (2015) doesn't have sincosf */
// use "-D __APPLE__" as a compiler flag to make sure this is hit.
#define R123_NO_SINCOS 1
#endif

#if R123_NO_SINCOS /* enable this if sincos and sincosf are not in the math library */
R123_CUDA_DEVICE R123_STATIC_INLINE void sincosf(float x, float *s, float *c) {
    *s = std::sinf(x);
    *c = std::cosf(x);
}

R123_CUDA_DEVICE R123_STATIC_INLINE void sincos(double x, double *s, double *c) {
    *s = std::sin(x);
    *c = std::cos(x);
}
#endif /* sincos is not in the math library */

// this is for sincosf
#if !defined(__CUDACC__)
static inline void sincospif(float x, float *s, float *c) {
    const float PIf = 3.1415926535897932f;
    sincosf(PIf*x, s, c);
}

static inline void sincospi(double x, double *s, double *c) {
    const double PI = 3.1415926535897932;
    sincos(PI*x, s, c);
}
#endif


#include <Random123/array.h>
#include <Random123/philox.h>
#include <Random123/threefry.h>
// NOTE: we do not support Random123's AES or ARS generators.

RandBLAS_OPTIMIZE_OFF
#include <Random123/boxmuller.hpp>
// ^ We've run into correctness issues with that file when using clang and
//   compiling with optimization enabled. To err on the side of caution
//   we disable compiler optimizations for clang and three other compilers.
//
RandBLAS_OPTIMIZE_ON
#include <Random123/uniform.hpp>

/// our extensions to random123
namespace r123ext
{
/** Apply boxmuller transform to all elements of ri. The number of elements of r
 * must be evenly divisible by 2. See also r123::uneg11all.
 *
 * @tparam CTR a random123 CBRNG ctr_type
 * @tparam T the return element type. The default return type is dictated by
 *           the RNG's ctr_type's value_type : float for 32 bit counter elements
 *           and double for 64.
 *
 * @param[in] ri a sequence of N random values generated using random123 CBRNG
 *               type RNG. The transform is applied pair wise to the sequence.
 *
 * @returns a std::array<T,N> of transformed floating point values.
 */
template <typename CTR, typename T = typename std::conditional
    <sizeof(typename CTR::value_type) == sizeof(uint32_t), float, double>::type>
auto boxmulall(
    CTR const &ri
) {
    std::array<T, CTR::static_size> ro;
    int nit = CTR::static_size / 2;
    for (int i = 0; i < nit; ++i)
    {
        auto [v0, v1] = r123::boxmuller(ri[2*i], ri[2*i + 1]);
        ro[2*i    ] = v0;
        ro[2*i + 1] = v1;
    }
    return ro;
}

/** @defgroup generators
 * Generators take CBRNG, counter,and key instances and return a sequence of
 * random floating point numbers in a std::array. The length of the squence is
 * the length of the counter and the precision is float for 32 bit counters and
 * double for 64.
 */
/// @{

/// Generate a sequence of random values and apply a Box-Muller transform.
struct boxmul
{
    /** Generate a sequence of random values and apply a Box-Muller transform.
     *
     * @tparam RNG a random123 CBRNG type
     *
     * @param[in] a random123 CBRNG instance used to generate the sequence
     * @param[in] the CBRNG counter
     * @param[in] the CBRNG key
     *
     * @returns a std::array<N,T> where N is the CBRNG's ctr_type::static_size
     *          and T is deduced from the RNG's counter element type : float
     *          for 32 bit counter elements and double for 64. For example when
     *          RNG is Philox4x32 the return is a std::array<float,4>.
     */
    template <typename RNG>
    static
    auto generate(
        RNG &rng,
        typename RNG::ctr_type const &c,
        typename RNG::key_type const &k
    ) {
        return boxmulall(rng(c,k));
    }
};

/// Generate a sequence of random values and transform to -1.0 to 1.0.
struct uneg11
{
    /** Generate a sequence of random values and transform to -1.0 to 1.0.
     *
     * @tparam RNG a random123 CBRNG type
     *
     * @param[in] rng: a random123 CBRNG instance used to generate the sequence
     * @param[in] c: CBRNG counter
     * @param[in] k: CBRNG key
     *
     * @returns a std::array<N,T> where N is the CBRNG's ctr_type::static_size
     *          and T is deduced from the RNG's counter element type : float
     *          for 32 bit counter elements and double for 64. For example when
     *          RNG is Philox4x32 the return is a std::array<float,4>.
     */
    template <typename RNG, typename T = typename std::conditional
        <sizeof(typename RNG::ctr_type::value_type) == sizeof(uint32_t), float, double>::type>
    static
    auto generate(
        RNG &rng,
        typename RNG::ctr_type const &c,
        typename RNG::key_type const &k
    ) {
        return r123::uneg11all<T>(rng(c,k));
    }
};

/// @}

} // end of namespace r123ext

