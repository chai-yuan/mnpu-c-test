/* TinyML arch – Zve32x SIMD‑accelerated operator implementations.
 *
 * All compute‑heavy operators (CONV2D, DWCONV2D, FC, GAP) use RISC‑V
 * "V" vector intrinsics to accelerate INT8 dot‑product / accumulation.
 * SOFTMAX, RESHAPE and ADD remain scalar as they see little benefit.
 *
 * Quantisation scheme is identical to soft.c (see there for details).
 *
 * Key design note:
 *   __riscv_vwadd_vx(…, 0, …) is optimised by GCC to vsext.vf2 which is
 *   ILLEGAL on Zve32x.  We add WIDEN_DELTA then subtract it back so
 *   the compiler cannot fold the two into a single vsext.
 */

#include "../internal.h"
#include <riscv_vector.h>
#include <string.h>

#define WIDEN_DELTA             \
    1 /* any value 1…127, must \
       * fit in int8 */

/* ------------------------------------------------------------------ */
/*  Static workspace (matches soft.c sizing)                           */
/* ------------------------------------------------------------------ */

#define TML_ZVE32X_MAX_KSIZE  (169) /* 13×13 max kernel */
#define TML_ZVE32X_MAX_KCSIZE (TML_ZVE32X_MAX_KSIZE * TML_MAX_CH)
#define TML_MAX_PARTIALS                                       \
    (32) /* int32 partials buffer, covers VLEN up to 256 bits \
          */

static int8_t sbuf[TML_ZVE32X_MAX_KCSIZE];

/* ------------------------------------------------------------------ */
/*  Widen helpers – avoid vsext.vf2 (illegal on Zve32x)               */
/* ------------------------------------------------------------------ */

/** Zero‑extend‑style widen: int8 → int16  (no zero‑point). */
static inline vint16m2_t _widen_i8_to_i16(vint8m1_t v, size_t vl) {
    vint16m2_t w = __riscv_vwadd_vx_i16m2(v, WIDEN_DELTA, vl);
    return __riscv_vsub_vx_i16m2(w, WIDEN_DELTA, vl);
}

/** Widen int8 → int16 and subtract zero‑point. */
static inline vint16m2_t _widen_sub_zp(vint8m1_t v, int16_t zp, size_t vl) {
    vint16m2_t w = __riscv_vwadd_vx_i16m2(v, WIDEN_DELTA, vl);
    return __riscv_vsub_vx_i16m2(w, (int16_t)(WIDEN_DELTA + zp), vl);
}

/** Widen int16 → int32 (used for GAP reduction). */
static inline vint32m4_t _widen_i16_to_i32(vint16m2_t v, size_t vl) {
    vint32m4_t w = __riscv_vwadd_vx_i32m4(v, WIDEN_DELTA, vl);
    return __riscv_vsub_vx_i32m4(w, WIDEN_DELTA, vl);
}

/* ------------------------------------------------------------------ */
/*  Scalar requantisation (identical to soft.c)                        */
/* ------------------------------------------------------------------ */

static inline int8_t requant(int32_t sum, int32_t mult, int32_t shift, int32_t zp, int act) {
    int64_t v = (int64_t)sum * mult;
    int32_t q = (int32_t)(v >> shift);
    if (act == TML_ACT_RELU && q < 0)
        q = 0;
    int32_t rc = q + zp;
    if (rc > 127)
        rc = 127;
    else if (rc < -128)
        rc = -128;
    return (int8_t)rc;
}

/* ------------------------------------------------------------------ */
/*  Scalar dot‑product  (8‑way unrolled, same as soft.c)               */
/* ------------------------------------------------------------------ */

/** Scalar fallback for short vectors where SIMD setup overhead dominates. */
static inline int32_t dot_prod_scalar(const int8_t *a, const int8_t *b, int len) {
    int32_t sum = 0;
    int     i;
    int     cnt = (len >> 3) << 3;
    for (i = 0; i < cnt; i += 8) {
        sum += (int32_t)a[i + 0] * b[i + 0];
        sum += (int32_t)a[i + 1] * b[i + 1];
        sum += (int32_t)a[i + 2] * b[i + 2];
        sum += (int32_t)a[i + 3] * b[i + 3];
        sum += (int32_t)a[i + 4] * b[i + 4];
        sum += (int32_t)a[i + 5] * b[i + 5];
        sum += (int32_t)a[i + 6] * b[i + 6];
        sum += (int32_t)a[i + 7] * b[i + 7];
    }
    for (; i < len; i++)
        sum += (int32_t)a[i] * b[i];
    return sum;
}

/* ------------------------------------------------------------------ */
/*  Vector dot‑product  (a, b are int8; bias already fused → no zp)   */
/* ------------------------------------------------------------------ */

/**
 * Compute Σ(a[i] * b[i]).
 * For len ≥ TML_VEC_THRESH uses RISC‑V vector widening MAC;
 * below threshold falls back to scalar (better for short kernels).
 * Returns the sum as int64_t for safety (no overflow on reduction).
 */
#define TML_VEC_THRESH 32

static inline int64_t vec_dot_nozp(const int8_t *a, const int8_t *b, int len) {
    if (len <= 0)
        return 0;

    /* Scalar fallback for short vectors — avoids SIMD setup overhead */
    if (len < TML_VEC_THRESH)
        return (int64_t)dot_prod_scalar(a, b, len);

    int    j  = 0;
    int    n  = len;
    size_t vl = __riscv_vsetvl_e16m2(n);
    if (vl == 0)
        return (int64_t)dot_prod_scalar(a, b, len);

    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);
    size_t     vl0  = vl;

    for (; n > 0; n -= (int)vl, j += (int)vl) {
        vl = __riscv_vsetvl_e16m2(n);

        vint8m1_t  va8  = __riscv_vle8_v_i8m1(a + j, vl);
        vint8m1_t  vb8  = __riscv_vle8_v_i8m1(b + j, vl);
        vint16m2_t va16 = _widen_i8_to_i16(va8, vl);
        vint16m2_t vb16 = _widen_i8_to_i16(vb8, vl);

        vacc = __riscv_vwmacc_vv_i32m4(vacc, va16, vb16, vl);
    }

    /* reduce partial-sums → scalar */
    int32_t partials[TML_MAX_PARTIALS];
    __riscv_vse32_v_i32m4(partials, vacc, vl0);

    int64_t total = 0;
    for (size_t k = 0; k < vl0; k++)
        total += (int64_t)partials[k];

    return total;
}

/* ------------------------------------------------------------------ */
/*  CONV2D / DWCONV2D                                                  */
/* ------------------------------------------------------------------ */

int tml_arch_conv2d(const tml_tensor_t *in, tml_tensor_t *out, const int8_t *w, const int32_t *b, const int32_t *qp,
                    int8_t kw, int8_t kh, int8_t sx, int8_t sy, int8_t dx, int8_t dy, uint16_t act, uint8_t pt,
                    uint8_t pb, uint8_t pl, uint8_t pr, uint32_t dmul, int32_t in_zp, int32_t out_zp) {
    int     maxk = (int)kw * kh;
    int     chi  = (int)in->c;
    int     cho  = (int)out->c;
    int     oh   = (int)out->h;
    int     ow   = (int)out->w;
    int     ih   = (int)in->h;
    int     iw   = (int)in->w;
    int8_t *outp = out->data;

    (void)dx;
    (void)dy;
    (void)pb;
    (void)pr;

    /* ── 1×1 kernel → point‑wise path (vectorised dot‑product) ──── */
    if (maxk == 1) {
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                const int8_t *inp = TML_MATP(in, oy * sy, ox * sx, 0);
                const int8_t *wp  = w;
                for (int cc = 0; cc < cho; cc++) {
                    int64_t s  = vec_dot_nozp(inp, wp, chi);
                    outp[cc]   = requant((int32_t)(s + (int64_t)b[cc]), qp[2 * cc], qp[2 * cc + 1], out_zp, act);
                    wp += chi;
                }
                outp += cho;
            }
        }
        return 0;
    }

    /* ── k×k convolution / depth‑wise ────────────────────────────── */
    for (int oy = 0; oy < oh; oy++) {
        int iy0 = oy * sy - pt;
        for (int ox = 0; ox < ow; ox++) {
            int ix0      = ox * sx - pl;
            int pad_flag = (iy0 < 0) || (ix0 < 0) || (iy0 + kh > ih) || (ix0 + kw > iw);

            int c_im2col = dmul ? cho : chi;

            if (!pad_flag) {
                /* valid region (no padding) */
                int sidx = 0;
                for (int cc = 0; cc < c_im2col; cc++) {
                    for (int ky = 0; ky < kh; ky++) {
                        const int8_t *row = TML_MATP(in, iy0 + ky, ix0, cc);
                        for (int kx = 0; kx < kw; kx++)
                            sbuf[sidx++] = row[kx * (int)in->c];
                    }
                }
            } else {
                /* padded region */
                memset(sbuf, (int)in_zp, (size_t)c_im2col * maxk);
                int sidx = 0;
                for (int cc = 0; cc < c_im2col; cc++) {
                    for (int ky = 0; ky < kh; ky++) {
                        int iy   = iy0 + ky;
                        int srow = sidx;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = ix0 + kx;
                            if (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                sbuf[srow + kx] = *TML_MATP(in, iy, ix, cc);
                        }
                        sidx += kw;
                    }
                }
            }

            /* vectorised dot‑products */
            if (dmul) {
                for (int oc = 0; oc < cho; oc++) {
                    int64_t s = vec_dot_nozp(sbuf + oc * maxk, w + oc * maxk, maxk);
                    outp[oc]  = requant((int32_t)(s + (int64_t)b[oc]), qp[2 * oc], qp[2 * oc + 1], out_zp, act);
                }
            } else {
                int stride_w = chi * maxk;
                for (int oc = 0; oc < cho; oc++) {
                    int64_t s = vec_dot_nozp(sbuf, w + oc * stride_w, stride_w);
                    outp[oc]  = requant((int32_t)(s + (int64_t)b[oc]), qp[2 * oc], qp[2 * oc + 1], out_zp, act);
                }
            }
            outp += cho;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  GAP – Global Average Pooling  (vectorised reduction)               */
/* ------------------------------------------------------------------ */

int tml_arch_gap(const tml_tensor_t *in, tml_tensor_t *out, int32_t qp_mult, int32_t qp_shift, int32_t in_zp,
                 int32_t out_zp) {
    int ih         = (int)in->h;
    int iw         = (int)in->w;
    int ic         = (int)in->c;
    int num_pixels = ih * iw;
    int c          = 0;

    /* Process channels in vector‑sized chunks.
     * For each chunk we maintain one int32 accumulator per channel,
     * summing over all spatial positions, then reduce + requantise. */
    for (; c < ic;) {
        int    chunk = ic - c;
        size_t vl    = __riscv_vsetvl_e8m1(chunk);
        if (vl == 0) {
            /* fallback: scalar for remaining channels */
            for (; c < ic; c++) {
                int32_t       sum = 0;
                const int8_t *col = TML_MATP(in, 0, 0, c);
                for (int i = 0; i < num_pixels; i++)
                    sum += (int32_t)col[i * ic] - in_zp;
                int64_t v = (int64_t)sum * qp_mult;
                int32_t q = (int32_t)(v >> qp_shift);
                int32_t o = q + out_zp;
                if (o > 127)
                    o = 127;
                else if (o < -128)
                    o = -128;
                out->data[c] = (int8_t)o;
            }
            continue;
        }

        vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

        for (int i = 0; i < num_pixels; i++) {
            const int8_t *row = in->data + (size_t)i * ic + c;
            vint8m1_t     v8  = __riscv_vle8_v_i8m1(row, vl);
            vint16m2_t    v16 = _widen_sub_zp(v8, (int16_t)in_zp, vl);
            vint32m4_t    v32 = _widen_i16_to_i32(v16, vl);
            vacc               = __riscv_vadd_vv_i32m4(vacc, v32, vl);
        }

        /* reduce partial sums and requantise */
        {
            int32_t partials[TML_MAX_PARTIALS];
            __riscv_vse32_v_i32m4(partials, vacc, vl);
            for (size_t k = 0; k < vl; k++, c++) {
                int64_t v = (int64_t)partials[k] * qp_mult;
                int32_t q = (int32_t)(v >> qp_shift);
                int32_t o = q + out_zp;
                if (o > 127)
                    o = 127;
                else if (o < -128)
                    o = -128;
                out->data[c] = (int8_t)o;
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  FC – Fully Connected  (vectorised dot‑product)                     */
/* ------------------------------------------------------------------ */

int tml_arch_fc(const tml_tensor_t *in, tml_tensor_t *out, const int8_t *w, const int32_t *b, int32_t qp_mult,
                int32_t qp_shift, int32_t out_zp) {
    int ic = (int)in->c;
    int oc = (int)out->c;

    for (int o = 0; o < oc; o++) {
        int64_t s  = vec_dot_nozp(in->data, w + o * ic, ic);
        out->data[o] = requant((int32_t)(s + (int64_t)b[o]), qp_mult, qp_shift, out_zp, 0);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  SOFTMAX – arg‑max style  (scalar – no SIMD benefit)                */
/* ------------------------------------------------------------------ */

int tml_arch_softmax(const tml_tensor_t *in, tml_tensor_t *out, int32_t in_zp, int32_t out_zp) {
    int     nc   = (int)in->c;
    int     best = 0;
    int32_t maxv = (int32_t)in->data[0] - in_zp;

    for (int i = 1; i < nc; i++) {
        int32_t v = (int32_t)in->data[i] - in_zp;
        if (v > maxv) {
            maxv = v;
            best = i;
        }
    }

    memset(out->data, 0, (size_t)nc);
    out->data[best] = 127;
    (void)out_zp;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  RESHAPE – copy verbatim  (scalar – no SIMD benefit)                */
/* ------------------------------------------------------------------ */

int tml_arch_reshape(const tml_tensor_t *in, tml_tensor_t *out) {
    int sz = (int)in->h * in->w * in->c;
    memcpy(out->data, in->data, (size_t)sz);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  ADD – element‑wise with per‑input requantisation (scalar)          */
/* ------------------------------------------------------------------ */

int tml_arch_add(const tml_tensor_t *in0, const tml_tensor_t *in1, tml_tensor_t *out, int32_t qp0_mult,
                 int32_t qp0_shift, int32_t in_zp0, int32_t qp1_mult, int32_t qp1_shift, int32_t in_zp1,
                 int32_t out_zp) {
    int n = (int)in0->h * in0->w * in0->c;

    for (int i = 0; i < n; i++) {
        int32_t v0 = (int32_t)in0->data[i] - in_zp0;
        int32_t v1 = (int32_t)in1->data[i] - in_zp1;

        int64_t r0 = (int64_t)v0 * qp0_mult;
        int64_t r1 = (int64_t)v1 * qp1_mult;
        int32_t r  = (int32_t)(r0 >> qp0_shift) + (int32_t)(r1 >> qp1_shift);

        int32_t o = r + out_zp;
        if (o > 127)
            o = 127;
        else if (o < -128)
            o = -128;
        out->data[i] = (int8_t)o;
    }
    return 0;
}
