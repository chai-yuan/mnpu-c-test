/* TinyML arch – pure‑software operator implementations (no SIMD).
 *
 * All operators use plain C loops.  These are the "soft reference"
 * implementations that can later be replaced by SIMD‑accelerated
 * versions (NEON, RISC‑V V, etc.) for specific architectures.
 *
 * Quantisation scheme (INT8 only):
 *
 *   real    = (qval - zp) * scale
 *   qval    = clamp(round(real / out_scale) + out_zp, −128, 127)
 *
 * Pre‑computed QP (Q31 fixed‑point):
 *
 *   mult / 2^shift  ≈  scale / out_scale         (per‑channel for conv)
 *
 *   output = clamp(((sum * mult) >> shift) + out_zp, −128, 127)
 *
 * Bias is already fused with −in_zp * Σw in the compiler.
 *
 * Per‑channel QP is stored as interleaved (mult, shift) pairs:
 *   qp[2*c + 0] = mult[c]
 *   qp[2*c + 1] = shift[c]
 */

#include "../internal.h"
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Shared workspace buffers (static – NOT reentrant)                  */
/* ------------------------------------------------------------------ */

#define TML_SOFT_MAX_KSIZE (169) /* 13×13 max kernel */
#define TML_SOFT_MAX_KCSIZE (TML_SOFT_MAX_KSIZE * TML_MAX_CH)

static int8_t sbuf[TML_SOFT_MAX_KCSIZE];

/* ------------------------------------------------------------------ */
/*  Low‑level arithmetic helpers (potentially SIMD targets)            */
/* ------------------------------------------------------------------ */

/** Dot product of two INT8 vectors → INT32 result. */
static inline int32_t dot_prod(const int8_t *a, const int8_t *b, int32_t len) {
    int32_t sum = 0;
    int32_t i;
    /* 8‑way unroll */
    int32_t cnt = (len >> 3) << 3;
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

/** Packed dot‑prod: compute two sums in one pass (better cache reuse). */
static inline void dot_prod_dual(const int8_t *a, const int8_t *b0, const int8_t *b1, int32_t len, int32_t *r0,
                                 int32_t *r1) {
    int32_t s0 = 0, s1 = 0;
    int32_t i;
    int32_t cnt = (len >> 2) << 2;
    for (i = 0; i < cnt; i += 4) {
        s0 += (int32_t)a[i + 0] * b0[i + 0];
        s1 += (int32_t)a[i + 0] * b1[i + 0];
        s0 += (int32_t)a[i + 1] * b0[i + 1];
        s1 += (int32_t)a[i + 1] * b1[i + 1];
        s0 += (int32_t)a[i + 2] * b0[i + 2];
        s1 += (int32_t)a[i + 2] * b1[i + 2];
        s0 += (int32_t)a[i + 3] * b0[i + 3];
        s1 += (int32_t)a[i + 3] * b1[i + 3];
    }
    for (; i < len; i++) {
        s0 += (int32_t)a[i] * b0[i];
        s1 += (int32_t)a[i] * b1[i];
    }
    *r0 = s0;
    *r1 = s1;
}

/** Requantize: int32 sum → int8, with optional RELU activation. */
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
/*  CONV2D / DWCONV2D  –  shared implementation                        */
/* ------------------------------------------------------------------ */

int tml_soft_conv2d(const tml_tensor_t *in, tml_tensor_t *out, const int8_t *w, const int32_t *b, const int32_t *qp,
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
    (void)pr; /* dilation, padding R/B unused yet */

    /* ── 1×1 kernel → point‑wise path ─────────────────────────── */
    if (maxk == 1) {
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                const int8_t *inp = TML_MATP(in, oy * sy, ox * sx, 0);
                const int8_t *wp  = w;
                int           cc  = 0;
                for (; cc + 2 <= cho; cc += 2) {
                    int32_t s0, s1;
                    dot_prod_dual(inp, wp + 0 * chi, wp + 1 * chi, chi, &s0, &s1);
                    outp[cc + 0] = requant(s0 + b[cc + 0], qp[2 * (cc + 0)], qp[2 * (cc + 0) + 1], out_zp, act);
                    outp[cc + 1] = requant(s1 + b[cc + 1], qp[2 * (cc + 1)], qp[2 * (cc + 1) + 1], out_zp, act);
                    wp += 2 * chi;
                }
                for (; cc < cho; cc++) {
                    int32_t s = dot_prod(inp, wp, chi);
                    outp[cc]  = requant(s + b[cc], qp[2 * cc], qp[2 * cc + 1], out_zp, act);
                    wp += chi;
                }
                outp += cho;
            }
        }
        return 0;
    }

    /* ── k×k convolution / depth‑wise ──────────────────────────── */
    for (int oy = 0; oy < oh; oy++) {
        int iy0 = oy * sy - pt;
        for (int ox = 0; ox < ow; ox++) {
            int ix0      = ox * sx - pl;
            int pad_flag = (iy0 < 0) || (ix0 < 0) || (iy0 + kh > ih) || (ix0 + kw > iw);

            int c_im2col = dmul ? cho : chi;

            if (!pad_flag) {
                /* ── valid region (no padding) ───────────────── */
                int sidx = 0;
                for (int cc = 0; cc < c_im2col; cc++) {
                    for (int ky = 0; ky < kh; ky++) {
                        const int8_t *row = TML_MATP(in, iy0 + ky, ix0, cc);
                        for (int kx = 0; kx < kw; kx++)
                            sbuf[sidx++] = row[kx * (int)in->c];
                    }
                }
            } else {
                /* ── padded region ────────────────────────────── */
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

            /* ── compute dot‑products ─────────────────────────── */
            if (dmul) {
                /* depth‑wise: each output ch = dot(slice[ch*maxk], w[ch*maxk]) */
                for (int oc = 0; oc < cho; oc++) {
                    int32_t s = dot_prod(sbuf + oc * maxk, w + oc * maxk, maxk);
                    outp[oc]  = requant(s + b[oc], qp[2 * oc], qp[2 * oc + 1], out_zp, act);
                }
            } else {
                /* regular conv: each output ch = dot(im2col, w[oc*chi*maxk]) */
                int stride_w = chi * maxk;
                for (int oc = 0; oc < cho; oc++) {
                    int32_t s = dot_prod(sbuf, w + oc * stride_w, stride_w);
                    outp[oc]  = requant(s + b[oc], qp[2 * oc], qp[2 * oc + 1], out_zp, act);
                }
            }
            outp += cho;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  GAP – Global Average Pooling                                        */
/* ------------------------------------------------------------------ */

int tml_soft_gap(const tml_tensor_t *in, tml_tensor_t *out, int32_t qp_mult, int32_t qp_shift, int32_t in_zp,
                 int32_t out_zp) {
    int ih         = (int)in->h;
    int iw         = (int)in->w;
    int ic         = (int)in->c;
    int num_pixels = ih * iw;

    /*
     *  output[c] = avg(input[c]) * in_s / out_s + out_zp
     *            = sum(input[c] − in_zp) / N * in_s / out_s + out_zp
     *            = sum(input[c] − in_zp) * qp_mult >> qp_shift + out_zp
     *
     * qp ≈ in_s / (N * out_s)  — pre‑computed by compiler.
     */
    for (int c = 0; c < ic; c++) {
        int32_t       sum = 0;
        const int8_t *col = TML_MATP(in, 0, 0, c);
        for (int i = 0; i < num_pixels; i++) {
            sum += (int32_t)col[i * ic] - in_zp;
        }
        int64_t v = (int64_t)sum * qp_mult;
        int32_t q = (int32_t)(v >> qp_shift);
        int32_t o = q + out_zp;
        if (o > 127)
            o = 127;
        else if (o < -128)
            o = -128;
        out->data[c] = (int8_t)o;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  FC – Fully Connected                                                */
/* ------------------------------------------------------------------ */

int tml_soft_fc(const tml_tensor_t *in, tml_tensor_t *out, const int8_t *w, const int32_t *b, int32_t qp_mult,
                int32_t qp_shift, int32_t out_zp) {
    int ic = (int)in->c;
    int oc = (int)out->c;

    /*
     *  output[o] = dot(input, w[o*ic ..]) * in_s * ws[0] / out_s + out_zp
     *            = dot(input, w[o*ic ..]) * qp_mult >> qp_shift + out_zp
     *
     * bias is fused with −in_zp * Σw.
     * FC uses a single (mult,shift) stored in the layer header.
     */
    for (int o = 0; o < oc; o++) {
        int32_t s    = dot_prod(in->data, w + o * ic, ic);
        out->data[o] = requant(s + b[o], qp_mult, qp_shift, out_zp, 0);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  SOFTMAX  –  arg‑max style (matches public API contract)             */
/* ------------------------------------------------------------------ */

int tml_soft_softmax(const tml_tensor_t *in, tml_tensor_t *out, int32_t in_zp, int32_t out_zp) {
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

    /* output[argmax] = 127, others = 0 — API contract */
    memset(out->data, 0, (size_t)nc);
    out->data[best] = 127;
    (void)out_zp;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  RESHAPE  –  copy data verbatim                                      */
/* ------------------------------------------------------------------ */

int tml_soft_reshape(const tml_tensor_t *in, tml_tensor_t *out) {
    int sz = (int)in->h * in->w * in->c;
    memcpy(out->data, in->data, (size_t)sz);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  ADD  –  element‑wise with per‑input re‑quantisation                 */
/* ------------------------------------------------------------------ */

int tml_soft_add(const tml_tensor_t *in0, const tml_tensor_t *in1, tml_tensor_t *out, int32_t qp0_mult,
                 int32_t qp0_shift, int32_t in_zp0, int32_t qp1_mult, int32_t qp1_shift, int32_t in_zp1,
                 int32_t out_zp) {
    /*
     *  real0 = (q0 − zp0) * in_s0
     *  real1 = (q1 − zp1) * in_s1
     *  out_q = (real0 + real1) / out_s + out_zp
     *        = (q0−zp0) * in_s0/out_s + (q1−zp1) * in_s1/out_s + out_zp
     *        ≈ ((q0−zp0) * qp0_mult >> qp0_shift) +
     *          ((q1−zp1) * qp1_mult >> qp1_shift) + out_zp
     */
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
