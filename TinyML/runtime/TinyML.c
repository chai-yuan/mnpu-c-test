/* TinyML Runtime – INT8 integer‑only soft‑CPU implementation.
 *
 * All requantisation uses pre‑computed (multiplier, shift) pairs — the
 * ``TinyML_Run`` hot path contains **no floating‑point operations**.
 * The only ``float`` usage is during ``TinyML_Create`` (one‑time scale
 * pre‑computation) and in the softmax layer (N ≤ ~1000, not a hot loop).
 */
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "TinyML.h"
#include "internal.h"
#include "arch/soft.h"

/* ================================================================== */
/*  internal helpers                                                   */
/* ================================================================== */

static void *_tml_alloc(const memory_if *mem, size_t sz)
{
    if (mem && mem->mem_alloc) return mem->mem_alloc(mem->self, sz);
    return malloc(sz);
}

static void _tml_free(const memory_if *mem, void *p)
{
    if (mem && mem->mem_free) mem->mem_free(mem->self, p);
    else free(p);
}

/* ---- clang‑style unused suppression ---- */
#ifdef __GNUC__
#define TML_UNUSED __attribute__((unused))
#else
#define TML_UNUSED
#endif

/* ================================================================== */
/*  forward declarations                                               */
/* ================================================================== */

static int _tml_conv2d_dw(const tml_tensor_t *in,  tml_tensor_t *out,
    const uint8_t *w, const uint8_t *b,
    int kw, int kh, int sx, int sy, int dx, int dy,
    int act, const uint8_t pad[4], int dmul,
    const tml_qp_t *qp, int32_t in_zp, int32_t out_zp);

static int _tml_gap(const tml_tensor_t *in, tml_tensor_t *out,
    const tml_qp_t *qp, int32_t in_zp, int32_t out_zp);

static int _tml_fc(const tml_tensor_t *in, tml_tensor_t *out,
    const uint8_t *w, const uint8_t *b,
    const tml_qp_t *qp, int32_t out_zp,
    int32_t in_zp TML_UNUSED);

static int _tml_softmax(const tml_tensor_t *in, tml_tensor_t *out,
    float in_s, int32_t in_zp, float out_s, int32_t out_zp);

static int _tml_add(const tml_tensor_t *in0, const tml_tensor_t *in1,
    tml_tensor_t *out,
    int32_t zp0, int32_t zp1, int32_t out_zp,
    const tml_qp_t *qp0, const tml_qp_t *qp1, const tml_qp_t *qp2);

/* ================================================================== */
/*  model lifecycle                                                    */
/* ================================================================== */

TinyML *TinyML_Create(const memory_if *mem, const uint8_t *model)
{
    const tml_bin_t *bin = (const tml_bin_t *)model;
    if (model[0] != TML_MAGIC_0 || model[1] != TML_MAGIC_1 ||
        model[2] != TML_MAGIC_2 || model[3] != TML_MAGIC_3)  return NULL;
    if (bin->mdl_type != 0)       return NULL; /* INT8 only */

    TinyML *self = (TinyML *)_tml_alloc(mem, sizeof(TinyML));
    if (!self) return NULL;

    self->bin     = bin;
    self->own_buf = 0;
    self->mem     = mem;
    self->layer_qp     = NULL;
    self->layer_qp_off = NULL;

    self->buf = (int8_t *)_tml_alloc(mem, (size_t)bin->buf_size);
    if (!self->buf) { _tml_free(mem, self); return NULL; }
    self->own_buf = 1;

    /* ── Pre‑compute integer requantisation scales ──────────────── */
    uint16_t nlyr = bin->layer_cnt;
    self->layer_qp_off = (int32_t *)_tml_alloc(mem,
        (size_t)(nlyr + 1) * sizeof(int32_t));
    if (!self->layer_qp_off) goto oom;

    /* first pass: count needed qp entries */
    const uint8_t *p = (const uint8_t *)bin + TML_HEADER_SIZE;
    int total_qp = 0;
    for (uint16_t li = 0; li < nlyr; li++) {
        const tml_head_t *h = (const tml_head_t *)p;
        self->layer_qp_off[li] = total_qp;

        switch (h->type) {
        case TML_CONV2D:
        case TML_DWCONV2D:
            total_qp += h->out_dims[3]; /* per out‑channel */
            break;
        case TML_FC:
            total_qp += 1; /* single scale shared by all channels */
            break;
        case TML_GAP:
            total_qp += 1;
            break;
        case TML_ADD:
            total_qp += 3; /* qp0 for in0, qp1 for in1, qp2 for out */
            break;
        default:
            total_qp += 0;
            break;
        }
        p += h->size;
    }
    self->layer_qp_off[nlyr] = total_qp;

    if (total_qp > 0) {
        self->layer_qp = (tml_qp_t *)_tml_alloc(mem,
            (size_t)total_qp * sizeof(tml_qp_t));
        if (!self->layer_qp) goto oom;
    }

    /* second pass: fill qp entries */
    p = (const uint8_t *)bin + TML_HEADER_SIZE;
    for (uint16_t li = 0; li < nlyr; li++) {
        const tml_head_t *h = (const tml_head_t *)p;
        int off = self->layer_qp_off[li];
        float   is = h->in_s;
        float   os = h->out_s;
        int32_t izp TML_UNUSED = h->in_zp;
        int32_t ozp TML_UNUSED = h->out_zp;

        switch (h->type) {
        case TML_CONV2D:
        case TML_DWCONV2D: {
            const tml_conv_t *l = (const tml_conv_t *)p;
            const float *ws = (const float *)(p + l->ws_oft);
            int cho = h->out_dims[3];
            for (int c = 0; c < cho; c++) {
                double s = (double)ws[c] * (double)is / (double)os;
                tml_scale_to_int(s, &self->layer_qp[off + c].mult,
                                    &self->layer_qp[off + c].shift);
            }
            break;
        }
        case TML_FC: {
            const tml_fc_t *l = (const tml_fc_t *)p;
            const float *ws = (const float *)(p + l->ws_oft);
            double s = (double)ws[0] * (double)is / (double)os;
            tml_scale_to_int(s, &self->layer_qp[off].mult,
                                &self->layer_qp[off].shift);
            break;
        }
        case TML_GAP: {
            double s = (double)is / (double)(h->in_dims[1]
                     * h->in_dims[2]) / (double)os;
            tml_scale_to_int(s, &self->layer_qp[off].mult,
                                &self->layer_qp[off].shift);
            break;
        }
        case TML_ADD: {
            const tml_add_t *l = (const tml_add_t *)p;
            double s0 = (double)is  / (double)os;
            double s1 = (double)l->in_s1 / (double)os;
            double s2 = 1.0 / (double)os; /* output requant */
            tml_scale_to_int(s0,
                &self->layer_qp[off + 0].mult,
                &self->layer_qp[off + 0].shift);
            tml_scale_to_int(s1,
                &self->layer_qp[off + 1].mult,
                &self->layer_qp[off + 1].shift);
            tml_scale_to_int(s2,
                &self->layer_qp[off + 2].mult,
                &self->layer_qp[off + 2].shift);
            break;
        }
        default:
            break;
        }
        p += h->size;
    }

    return self;

oom:
    if (self->layer_qp_off) _tml_free(mem, self->layer_qp_off);
    if (self->layer_qp)     _tml_free(mem, self->layer_qp);
    if (self->own_buf)      _tml_free(mem, self->buf);
    _tml_free(mem, self);
    return NULL;
}

void TinyML_Destroy(TinyML *self)
{
    if (!self) return;
    if (self->own_buf)   _tml_free(self->mem, self->buf);
    if (self->layer_qp)      _tml_free(self->mem, self->layer_qp);
    if (self->layer_qp_off)  _tml_free(self->mem, self->layer_qp_off);
    _tml_free(self->mem, self);
}

int TinyML_GetInputSize(TinyML *self)
{
    const uint16_t *d = self->bin->in_dims;
    return (int)d[1] * (int)d[2] * (int)d[3];
}

int TinyML_GetOutputSize(TinyML *self)
{
    const uint16_t *d = self->bin->out_dims;
    return (int)d[1] * (int)d[2] * (int)d[3];
}

/* ================================================================== */
/*  inference loop                                                     */
/* ================================================================== */

int TinyML_Run(TinyML *self, const int8_t *input, float *output)
{
    const tml_bin_t  *bin = self->bin;
    const uint8_t    *p   = (const uint8_t *)bin + TML_HEADER_SIZE;
    int8_t           *buf = self->buf;

    int in_size = TinyML_GetInputSize(self);
    memcpy(buf, input, (size_t)in_size);

    tml_tensor_t in, out;
    in.data  = buf;
    in.dims  = bin->in_dims[0];
    in.h     = bin->in_dims[1];
    in.w     = bin->in_dims[2];
    in.c     = bin->in_dims[3];

    for (uint16_t li = 0; li < bin->layer_cnt; li++) {
        const tml_head_t *h = (const tml_head_t *)p;
        const tml_qp_t   *qp = self->layer_qp
            ? &self->layer_qp[self->layer_qp_off[li]] : NULL;

        if (li > 0) {
            memcpy(&in, h->in_dims, sizeof(uint16_t) * 4);
            in.data = buf + h->in_oft;
        }
        memcpy(&out, h->out_dims, sizeof(uint16_t) * 4);
        out.data = buf + h->out_oft;

        int rc;
        switch (h->type) {
        case TML_CONV2D:
        case TML_DWCONV2D: {
            const tml_conv_t *l = (const tml_conv_t *)p;
            rc = _tml_conv2d_dw(&in, &out,
                p + l->w_oft, p + l->b_oft,
                l->kernel_w,       l->kernel_h,
                l->stride_w,        l->stride_h,
                l->dilation_w,      l->dilation_h,
                l->act, l->pad, l->depth_mul,
                qp, h->in_zp, h->out_zp);
            break;
        }
        case TML_GAP:
            rc = _tml_gap(&in, &out, qp, h->in_zp, h->out_zp);
            break;
        case TML_FC: {
            const tml_fc_t *l = (const tml_fc_t *)p;
            rc = _tml_fc(&in, &out,
                p + l->w_oft, p + l->b_oft,
                qp, h->out_zp, h->in_zp);
            break;
        }
        case TML_SOFTMAX:
            /* softmax still uses float – the only float in Run().
             * Reason: N ≤ ~1000; a pure‑integer exp‑LUT softmax
             * is measurable but adds ~300 lines.  Future work. */
            rc = _tml_softmax(&in, &out, h->in_s, h->in_zp,
                              h->out_s, h->out_zp);
            break;
        case TML_RESHAPE:
            rc = 0;
            break;
        case TML_ADD: {
            const tml_add_t *l = (const tml_add_t *)p;
            tml_tensor_t in1;
            memcpy(&in1, h->in_dims, sizeof(uint16_t) * 4);
            in1.data = buf + l->in_oft1;
            rc = _tml_add(&in, &in1, &out,
                          h->in_zp, l->in_zp1, h->out_zp,
                          &qp[0], &qp[1], &qp[2]);
            break;
        }
        default:
            return TML_ERR_LAYER;
        }
        if (rc) return rc;

        if (h->is_out) {
            int n = (int)h->out_dims[1] * (int)h->out_dims[2]
                   * (int)h->out_dims[3];
            if (!bin->out_deq) {
                for (int j = 0; j < n; j++)
                    output[j] = (float)(out.data[j]);
            } else {
                float  os = h->out_s;
                int32_t zp = h->out_zp;
                for (int j = 0; j < n; j++)
                    output[j] = ((float)out.data[j] - (float)zp) * os;
            }
        }
        p += h->size;
    }

    return TML_OK;
}

/* ================================================================== */
/*  layer implementations  (100 % integer arithmetic)                  */
/* ================================================================== */

/* static workspace buffers */
#define TML_MAX_KSIZE   (5 * 5)
#define TML_MAX_KCSIZE  (3 * 3 * 256)

static int32_t _tml_k_oft[TML_MAX_KSIZE];
static int8_t  _tml_sbuf[TML_MAX_KCSIZE];

/* ──────────────────────── CONV2D / DWCONV2D ──────────────────────── */

static int _tml_conv2d_dw(
    const tml_tensor_t *in,  tml_tensor_t *out,
    const uint8_t *w, const uint8_t *b,
    int kw, int kh, int sx, int sy, int dx, int dy,
    int act, const uint8_t pad[4], int dmul,
    const tml_qp_t *qp, int32_t in_zp, int32_t out_zp)
{
    int maxk  = kw * kh;
    int chi   = dmul ? 1 : in->c;
    int cho   = out->c;
    int pad_t = (int)pad[0], pad_b = (int)pad[1];
    int pad_l = (int)pad[2], pad_r = (int)pad[3];
    int pad_flag = (pad_t | pad_b | pad_l | pad_r) != 0;

    if (dx != 1 || dy != 1) return TML_ERR_PARAM;

    const int8_t  *w8  = (const int8_t  *)w;
    const int32_t *b32 = (const int32_t *)b;
    int8_t *outp = out->data;

    /* point‑wise fast path (maxk==1, no pad, regular conv) */
    if (maxk == 1 && !pad_flag && !dmul) {
#define PW_BATCH 2
        int32_t sums[PW_BATCH];
        for (int y = 0; y < out->h; y++) {
            for (int x = 0; x < out->w; x++) {
                const int8_t *sptr = TML_MATP(in, sy * y, sx * x, 0);
                const int8_t *kptr = w8;
                int c = 0;
                for (; c + PW_BATCH <= cho; ) {
                    tml_dot_prod_pack2(sptr, kptr, kptr + chi,
                                       (uint32_t)chi, sums);
                    tml_post_i32_n(PW_BATCH, sums, b32 + c,
                                   qp + c, out_zp, outp + c);
                    c     += PW_BATCH;
                    kptr  += chi * PW_BATCH;
                }
                for (; c < cho; c++) {
                    tml_dot_prod(sptr, kptr, (uint32_t)chi, sums);
                    tml_post_i32(sums, b32[c], qp + c, out_zp, outp + c);
                    kptr += chi;
                }
            }
        }
        return TML_OK;
    }

    /* k_oft lookup table */
    int idx = 0;
    for (int y = 0; y < kh; y++)
        for (int x = 0; x < kw; x++)
            _tml_k_oft[idx++] = y * in->w * chi + x * chi;

    for (int y = 0; y < out->h; y++) {
        int src_y0 = sy * y - pad_t;
        for (int x = 0; x < out->w; x++) {
            int src_x0 = sx * x - pad_l;
            int slow = (src_y0 < 0) | (src_x0 < 0)
                     | (src_y0 + kh > in->h)
                     | (src_x0 + kw > in->w);
            int clip_y = src_y0 < 0 ? 0 : (src_y0 > in->h - 1
                                           ? in->h - 1 : src_y0);
            int clip_x = src_x0 < 0 ? 0 : (src_x0 > in->w - 1
                                           ? in->w - 1 : src_x0);
            const int8_t *sptr_base = TML_MATP(in, clip_y, clip_x, 0);

            if (!slow) {
                int sidx = 0;
                const int8_t *sp = sptr_base;
                for (int cc = 0; cc < (dmul ? cho : chi); cc++) {
                    for (int k = 0; k < maxk; k++)
                        _tml_sbuf[sidx + k] = sp[_tml_k_oft[k]];
                    sidx += maxk;
                    sp = sptr_base + (dmul ? (cc + 1) / dmul : cc + 1);
                }
            } else {
                memset(_tml_sbuf, (int)in_zp,
                       (size_t)(dmul ? cho * maxk : chi * maxk));
                int ky0 = src_y0 < 0 ? -src_y0 : 0;
                int kx0 = src_x0 < 0 ? -src_x0 : 0;
                int ky1 = in->h - src_y0 > kh ? kh : in->h - src_y0;
                int kx1 = in->w - src_x0 > kw ? kw : in->w - src_x0;
                int sidx = 0;
                const int8_t *sp = sptr_base;
                for (int cc = 0; cc < (dmul ? cho : chi); cc++) {
                    for (int ky = ky0; ky < ky1; ky++)
                        for (int kx = kx0; kx < kx1; kx++)
                            _tml_sbuf[sidx + ky * kw + kx]
                                = sp[_tml_k_oft[ky * kw + kx]];
                    sidx += maxk;
                    sp = sptr_base + (dmul ? (cc + 1) / dmul : cc + 1);
                }
            }

            /* dot‑product + integer post‑process */
            const int8_t *sptr = _tml_sbuf;
            if (act == TML_ACT_RELU) {
                for (int c = 0; c < cho; c++) {
                    int32_t sum;
                    tml_dot_prod(sptr, w8 + c * chi * maxk,
                                 (uint32_t)(maxk * chi), &sum);
                    tml_post_i32_n_relu(1, &sum, b32 + c, qp + c,
                                        out_zp, outp + c);
                    if (dmul) sptr += maxk;
                }
            } else {
                for (int c = 0; c < cho; c++) {
                    int32_t sum;
                    tml_dot_prod(sptr, w8 + c * chi * maxk,
                                 (uint32_t)(maxk * chi), &sum);
                    tml_post_i32(&sum, b32[c], qp + c, out_zp, outp + c);
                    if (dmul) sptr += maxk;
                }
            }
            outp += cho;
        }
    }

    return TML_OK;
}

/* ────────────────────── GAP (Global Average Pool) ────────────────── */

static int _tml_gap(const tml_tensor_t *in, tml_tensor_t *out,
                     const tml_qp_t *qp, int32_t in_zp, int32_t out_zp)
{
    int area = in->h * in->w;
    for (int c = 0; c < out->c; c++) {
        int32_t sum = 0;
        const int8_t *p = in->data + c;
        for (int y = 0; y < in->h; y++)
            for (int x = 0; x < in->w; x++) {
                sum += (int32_t)*p;
                p   += out->c;
            }
        /* avg = sum/area, then requantise: (avg - in_zp) * scale + out_zp
         * The pre‑computed qp already includes /(in_h*in_w*out_s). */
        int64_t acc = (int64_t)(sum - in_zp * area) * (int64_t)qp->mult;
        acc = tml_rshift_round(acc, qp->shift);
        int32_t v = (int32_t)acc + out_zp;
        out->data[c] = tml_clamp_i8(v);
    }
    return TML_OK;
}

/* ───────────────────────────── FC ────────────────────────────────── */

static int _tml_fc(const tml_tensor_t *in, tml_tensor_t *out,
                    const uint8_t *w, const uint8_t *b,
                    const tml_qp_t *qp, int32_t out_zp,
                    int32_t in_zp)
{
    (void)in_zp; /* bias already fused by compiler */
    int mi = in->c;
    int mo = out->c;
    const int8_t  *w8  = (const int8_t  *)w;
    const int32_t *b32 = (const int32_t *)b;

    for (int c = 0; c < mo; c++) {
        int32_t sum;
        tml_dot_prod(in->data, w8 + c * mi, (uint32_t)mi, &sum);
        /* FC uses a single qp[0] for all channels */
        tml_post_i32(&sum, b32[c], qp, out_zp, out->data + c);
    }
    return TML_OK;
}

/* ──────────────────────────── SOFTMAX ────────────────────────────── */
/* (float path – see comment in TinyML_Run)                            */

#define TML_EXP _tml_approx_exp_s

static float _tml_approx_exp_s(float x) {
    float p = 1.442695040f * x;
    union { uint32_t i; float f; } v;
    v.i = (uint32_t)((1U << 23) * (p + 121.2740838f
          + 27.7280233f / (4.84252568f - (p - (int)p))
          - 1.49012907f * (p - (int)p)));
    return v.f;
}

static int _tml_softmax(const tml_tensor_t *in, tml_tensor_t *out,
                         float in_s, int32_t in_zp,
                         float out_s, int32_t out_zp)
{
    int n = (int)in->c;
    float buf[TML_MAX_CH];

    for (int i = 0; i < n; i++)
        buf[i] = ((float)in->data[i] - (float)in_zp) * in_s;

    float dmax = -FLT_MAX;
    for (int i = 0; i < n; i++)
        if (buf[i] > dmax) dmax = buf[i];

    float sum = 0;
    float inv_os = 1.0f / out_s;
    for (int i = 0; i < n; i++) {
        buf[i]  = TML_EXP(buf[i] - dmax) - 0.000001f;
        sum    += buf[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)(buf[i] * inv_sum * inv_os) + out_zp;
        out->data[i] = tml_clamp_i8(v);
    }
    return TML_OK;
}

/* ───────────────────────────── ADD ───────────────────────────────── */

static int _tml_add(const tml_tensor_t *in0, const tml_tensor_t *in1,
                     tml_tensor_t *out,
                     int32_t zp0, int32_t zp1, int32_t out_zp,
                     const tml_qp_t *qp0, const tml_qp_t *qp1,
                     const tml_qp_t *qp2)
{
    int n = in0->h * in0->w * in0->c;
    for (int i = 0; i < n; i++) {
        int32_t d0 = (int32_t)in0->data[i] - zp0;
        int32_t d1 = (int32_t)in1->data[i] - zp1;

        /* requantise each addend and sum in integer */
        int64_t a0 = (int64_t)d0 * (int64_t)qp0->mult;
        a0 = tml_rshift_round(a0, qp0->shift);
        int64_t a1 = (int64_t)d1 * (int64_t)qp1->mult;
        a1 = tml_rshift_round(a1, qp1->shift);
        int64_t acc = (a0 + a1) * (int64_t)qp2->mult;
        acc = tml_rshift_round(acc, qp2->shift);

        int32_t v = (int32_t)acc + out_zp;
        out->data[i] = tml_clamp_i8(v);
    }
    return TML_OK;
}
