/* TinyML soft backend – integer‑only primitives.
 *
 * Contains zero `float` or `double` types.  All requantisation
 * parameters are pre‑computed by the compiler and stored in the
 * TMDL binary as ``tml_qp_t`` pairs.
 */
#ifndef TINYML_ARCH_SOFT_H
#define TINYML_ARCH_SOFT_H

#include <stdint.h>
#include <string.h>
#include "internal.h"

#ifndef TML_INLINE
#define TML_INLINE __attribute__((always_inline)) static inline
#endif

/* ================================================================== */
/*  rounding right‑shift                                               */
/* ================================================================== */

TML_INLINE int64_t tml_rshift_round(int64_t x, int n)
{
    if (n <= 0) return x << (-n);
    return x >= 0 ? (x + ((int64_t)1 << (n - 1))) >> n
                  : -(((-x) + ((int64_t)1 << (n - 1))) >> n);
}

/* ================================================================== */
/*  dot‑product                                                        */
/* ================================================================== */

TML_INLINE void tml_dot_prod(const int8_t *src, const int8_t *ker,
                              uint32_t size, int32_t *result)
{
    int32_t sum = 0;
    uint32_t i = 0, n8 = (size >> 3) << 3;
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
    for (; i < size; i++) sum += (int32_t)src[i] * (int32_t)ker[i];
    *result = sum;
}

TML_INLINE void tml_dot_prod_pack2(const int8_t *src,
                                    const int8_t *k0, const int8_t *k1,
                                    uint32_t size, int32_t *res)
{
    int32_t s0 = 0, s1 = 0;
    uint32_t i = 0, n8 = (size >> 3) << 3;
    for (; i < n8; ) {
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i]; i++;
    }
    for (; i < size; i++) {
        s0 += (int32_t)src[i] * (int32_t)k0[i];
        s1 += (int32_t)src[i] * (int32_t)k1[i];
    }
    res[0] = s0; res[1] = s1;
}

/* ================================================================== */
/*  integer post‑processing                                            */
/* ================================================================== */

TML_INLINE int8_t tml_clamp_i8(int32_t x) {
    if ((int32_t)((int8_t)x) != x)
        return x > 127 ? (int8_t)127 : (int8_t)-128;
    return (int8_t)x;
}

TML_INLINE void tml_post_i32(const int32_t *sum, int32_t bias,
                              const tml_qp_t *qp, int32_t out_zp,
                              int8_t *outp)
{
    int64_t acc = (int64_t)(*sum + bias) * (int64_t)qp->mult;
    acc = tml_rshift_round(acc, qp->shift);
    int32_t v = (int32_t)acc + out_zp;
    *outp = tml_clamp_i8(v);
}

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

TML_INLINE void tml_post_i32_n_relu(int n, const int32_t *sums,
                                     const int32_t *bias,
                                     const tml_qp_t *qp, int32_t out_zp,
                                     int8_t *outp)
{
    for (int i = 0; i < n; i++) {
        int64_t acc = (int64_t)(sums[i] + bias[i]) * (int64_t)qp[i].mult;
        acc = tml_rshift_round(acc, qp[i].shift);
        if (acc < 0) acc = 0;
        int32_t v = (int32_t)acc + out_zp;
        outp[i] = tml_clamp_i8(v);
    }
}

#endif /* TINYML_ARCH_SOFT_H */
