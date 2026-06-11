/* TinyML soft backend – portable C primitives for integer‑only inference.
 *
 * Provides dot‑product and **integer** post‑processing (bias + activation +
 * requantisation).  The only floating‑point usage is during
 * ``Tinyml_Create`` when scales are converted to (mult, shift) pairs; the
 * ``TinyML_Run`` hot path is 100 % integer.
 */
#ifndef TINYML_ARCH_SOFT_H
#define TINYML_ARCH_SOFT_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "internal.h"

#ifndef TML_INLINE
#define TML_INLINE __attribute__((always_inline)) static inline
#endif

/* ------------------------------------------------------------------ */
/*  float scale → integer (mult, shift)  – used at Create time only   */
/* ------------------------------------------------------------------ */

/**
 * Convert a positive floating‑point scale into an integer
 * representation::
 *
 *     scale ≈ mult / 2^shift
 *
 * *mult* is a signed Q31 fixed‑point number.  *shift* is a right‑shift
 * amount.  Together they encode the same value as *scale*:
 *
 *     output = ((int64_t)accumulator * mult) >> shift
 *
 * is equivalent to::
 *
 *     output = accumulator * scale   (up to 31‑bit precision)
 *
 * Adapted from TensorFlow Lite's ``QuantizeMultiplier``.
 */
TML_INLINE void tml_scale_to_int(double scale, int32_t *mult,
                                  int32_t *shift)
{
    if (scale <= 0.0) { *mult = 0; *shift = 0; return; }

    /* decompose: scale = mantissa * 2^exp, mantissa ∈ [0.5, 1.0) */
    int exponent;
    double mantissa = frexp(scale, &exponent);

    /* fixed‑point: mult = round(mantissa * 2^31), in [2^30, 2^31) */
    int64_t m = (int64_t)(mantissa * 2147483648.0 + 0.5);
    if (m >= 0x80000000LL) { m >>= 1; exponent++; }

    *mult  = (int32_t)m;
    /* recovery:  scale = mult / 2^(31 - exponent)        */
    /* so:        x * scale = (x * mult) >> (31 - exponent)  */
    *shift = 31 - exponent;
}

/** Rounding right‑shift: ``(x + (1 << (n-1))) >> n`` for x ≥ 0. */
TML_INLINE int64_t tml_rshift_round(int64_t x, int n)
{
    if (n <= 0) return x << (-n);
    return x >= 0 ? (x + ((int64_t)1 << (n - 1))) >> n
                  : -(((-x) + ((int64_t)1 << (n - 1))) >> n);
}

/* ------------------------------------------------------------------ */
/*  dot‑product helpers  (pure integer)                                */
/* ------------------------------------------------------------------ */

TML_INLINE void tml_dot_prod(const int8_t *src, const int8_t *ker,
                              uint32_t size, int32_t *result)
{
    int32_t sum = 0;
    uint32_t i = 0;
    uint32_t n8 = (size >> 3) << 3;
    for (; i < n8; ) {
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
        sum += (int32_t)src[i] * (int32_t)ker[i]; i++;
    }
    for (; i < size; i++)
        sum += (int32_t)src[i] * (int32_t)ker[i];
    *result = sum;
}

TML_INLINE void tml_dot_prod_pack2(const int8_t *src,
                                    const int8_t *ker0,
                                    const int8_t *ker1,
                                    uint32_t size, int32_t *result)
{
    int32_t sum0 = 0, sum1 = 0;
    uint32_t i = 0;
    uint32_t n8 = (size >> 3) << 3;
    for (; i < n8; ) {
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i]; i++;
    }
    for (; i < size; i++) {
        sum0 += (int32_t)src[i] * (int32_t)ker0[i];
        sum1 += (int32_t)src[i] * (int32_t)ker1[i];
    }
    result[0] = sum0; result[1] = sum1;
}

/* ------------------------------------------------------------------ */
/*  integer post‑processing  (no floats!)                              */
/* ------------------------------------------------------------------ */

/** Clamp int32 to int8 range. */
TML_INLINE int8_t tml_clamp_i8(int32_t x) {
    if      ((int32_t)((int8_t)(x)) != x)
        return x > 127 ? (int8_t)127 : (int8_t)-128;
    return (int8_t)x;
}

/**
 * Single‑channel::
 *     sum  = dot_product + bias
 *     real = sum * scale  (via mult/shift)
 *     out  = clamp(real + out_zp)
 */
TML_INLINE void tml_post_i32(const int32_t *sum, int32_t bias,
                              const tml_qp_t *qp, int32_t out_zp,
                              int8_t *outp)
{
    int64_t acc = (int64_t)(*sum + bias) * (int64_t)qp->mult;
    acc = tml_rshift_round(acc, qp->shift);
    int32_t v = (int32_t)acc + out_zp;
    *outp = tml_clamp_i8(v);
}

/** Batched version for n output channels. */
TML_INLINE void tml_post_i32_n(int n, const int32_t *sums,
                                const int32_t *bias,
                                const tml_qp_t *qp, int32_t out_zp,
                                int8_t *outp)
{
    for (int i = 0; i < n; i++) {
        int64_t acc = (int64_t)(sums[i] + bias[i]) * (int64_t)qp[i].mult;
        acc = tml_rshift_round(acc, qp[i].shift);
        int32_t v = (int32_t)acc + out_zp;
        outp[i] = tml_clamp_i8(v);
    }
}

/**
 * Batched version **with ReLU activation**.
 * Equivalent to tml_post_i32_n with max(0, ...) applied before clamping.
 */
TML_INLINE void tml_post_i32_n_relu(int n, const int32_t *sums,
                                     const int32_t *bias,
                                     const tml_qp_t *qp, int32_t out_zp,
                                     int8_t *outp)
{
    for (int i = 0; i < n; i++) {
        int64_t acc = (int64_t)(sums[i] + bias[i]) * (int64_t)qp[i].mult;
        acc = tml_rshift_round(acc, qp[i].shift);
        if (acc < 0) acc = 0;  /* ReLU */
        int32_t v = (int32_t)acc + out_zp;
        outp[i] = tml_clamp_i8(v);
    }
}

#endif /* TINYML_ARCH_SOFT_H */
