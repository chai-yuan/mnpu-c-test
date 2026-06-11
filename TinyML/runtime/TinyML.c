/* TinyML Runtime – integer‑only (zero float types).
 *
 * All requantisation parameters are pre‑computed by the compiler and
 * stored inlined in the TMDL binary as (int32 mult, int32 shift) pairs.
 * No float parsing or scale arithmetic is needed at runtime.
 */
#include <stdlib.h>
#include <string.h>

#include "TinyML.h"
#include "internal.h"
#include "arch/soft.h"

/* ================================================================== */
/*  internal helpers                                                   */
/* ================================================================== */

static void *_tml_alloc(const memory_if *mem, size_t sz) {
    if (mem && mem->mem_alloc) return mem->mem_alloc(mem->self, sz);
    return malloc(sz);
}
static void _tml_free(const memory_if *mem, void *p) {
    if (mem && mem->mem_free) mem->mem_free(mem->self, p);
    else free(p);
}
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

static int _tml_argmax(const tml_tensor_t *in, tml_tensor_t *out);

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
    self->bin = bin;  self->own_buf = 0;  self->mem = mem;

    self->buf = (int8_t *)_tml_alloc(mem, (size_t)bin->buf_size);
    if (!self->buf) { _tml_free(mem, self); return NULL; }
    self->own_buf = 1;
    return self;
}

void TinyML_Destroy(TinyML *self)
{
    if (!self) return;
    if (self->own_buf) _tml_free(self->mem, self->buf);
    _tml_free(self->mem, self);
}

int TinyML_GetInputSize (TinyML *self) { const uint16_t *d=self->bin->in_dims;  return (int)d[1]*(int)d[2]*(int)d[3]; }
int TinyML_GetOutputSize(TinyML *self) { const uint16_t *d=self->bin->out_dims; return (int)d[1]*(int)d[2]*(int)d[3]; }

/* ================================================================== */
/*  inference loop                                                     */
/* ================================================================== */

int TinyML_Run(TinyML *self, const int8_t *input, int8_t *output)
{
    const tml_bin_t  *bin = self->bin;
    const uint8_t    *p   = (const uint8_t *)bin + TML_HEADER_SIZE;
    int8_t           *buf = self->buf;

    memcpy(buf, input, (size_t)TinyML_GetInputSize(self));

    tml_tensor_t in, out;
    in.dims = bin->in_dims[0]; in.h = bin->in_dims[1];
    in.w    = bin->in_dims[2]; in.c = bin->in_dims[3];
    in.data = buf;

    for (uint16_t li = 0; li < bin->layer_cnt; li++) {
        const tml_head_t *h = (const tml_head_t *)p;
        /* header QP is used by fc/gap/add; conv reads per‑channel from ws_oft */
        tml_qp_t hdr_qp = { h->qp_mult, h->qp_shift };

        if (li > 0) {
            memcpy(&in, h->in_dims, sizeof(uint16_t) * 4);
            in.data = buf + h->in_oft;
        }
        memcpy(&out, h->out_dims, sizeof(uint16_t) * 4);
        out.data = buf + h->out_oft;

        int rc;
        switch (h->type) {
        case TML_CONV2D: case TML_DWCONV2D: {
            const tml_conv_t *l = (const tml_conv_t *)p;
            /* per‑channel QP at (p + l->ws_oft) as int32[cho*2] */
            const tml_qp_t *cq = (const tml_qp_t *)(p + l->ws_oft);
            rc = _tml_conv2d_dw(&in, &out, p + l->w_oft, p + l->b_oft,
                l->kernel_w, l->kernel_h, l->stride_w, l->stride_h,
                l->dilation_w, l->dilation_h,
                l->act, l->pad, l->depth_mul,
                cq, h->in_zp, h->out_zp);
            break;
        }
        case TML_GAP:
            rc = _tml_gap(&in, &out, &hdr_qp, h->in_zp, h->out_zp);
            break;
        case TML_FC: {
            const tml_fc_t *l = (const tml_fc_t *)p;
            rc = _tml_fc(&in, &out, p + l->w_oft, p + l->b_oft,
                         &hdr_qp, h->out_zp, h->in_zp);
            break;
        }
        case TML_SOFTMAX:
            rc = _tml_argmax(&in, &out);
            break;
        case TML_RESHAPE:
            rc = 0;
            break;
        case TML_ADD: {
            const tml_add_t *l = (const tml_add_t *)p;
            tml_tensor_t in1;
            memcpy(&in1, h->in_dims, sizeof(uint16_t) * 4);
            in1.data = buf + l->in_oft1;
            tml_qp_t qp1 = { l->qp1_mult, l->qp1_shift };
            tml_qp_t qp2 = { l->qp2_mult, l->qp2_shift };
            rc = _tml_add(&in, &in1, &out,
                          h->in_zp, l->in_zp1, h->out_zp,
                          &hdr_qp, &qp1, &qp2);
            break;
        }
        default:
            return TML_ERR_LAYER;
        }
        if (rc) return rc;

        if (h->is_out) {
            int n = TinyML_GetOutputSize(self);
            memcpy(output, out.data, (size_t)n);
        }
        p += h->size;
    }
    return TML_OK;
}

/* ================================================================== */
/*  layer implementations  (100 % integer arithmetic)                  */
/* ================================================================== */

#define TML_MAX_KSIZE   (5 * 5)
#define TML_MAX_KCSIZE  (3 * 3 * 256)
static int32_t _tml_k_oft[TML_MAX_KSIZE];
static int8_t  _tml_sbuf[TML_MAX_KCSIZE];

/* ──────── CONV2D / DWCONV2D ──────── */

static int _tml_conv2d_dw(
    const tml_tensor_t *in, tml_tensor_t *out,
    const uint8_t *w, const uint8_t *b,
    int kw, int kh, int sx, int sy, int dx, int dy,
    int act, const uint8_t pad[4], int dmul,
    const tml_qp_t *qp, int32_t in_zp, int32_t out_zp)
{
    int maxk  = kw * kh;
    int chi   = dmul ? 1 : in->c;
    int cho   = out->c;
    int pt = (int)pad[0], pb = (int)pad[1], pl = (int)pad[2], pr = (int)pad[3];
    int pad_flag = (pt | pb | pl | pr) != 0;
    if (dx != 1 || dy != 1) return TML_ERR_PARAM;

    const int8_t  *w8  = (const int8_t  *)w;
    const int32_t *b32 = (const int32_t *)b;
    int8_t *outp = out->data;

    if (maxk == 1 && !pad_flag && !dmul) {
#define PW_BATCH 2
        int32_t sums[PW_BATCH];
        for (int y = 0; y < out->h; y++) {
            for (int x = 0; x < out->w; x++) {
                const int8_t *sp = TML_MATP(in, sy * y, sx * x, 0);
                const int8_t *kp = w8;
                int c = 0;
                for (; c + PW_BATCH <= cho; ) {
                    tml_dot_prod_pack2(sp, kp, kp + chi, (uint32_t)chi, sums);
                    tml_post_i32_n(PW_BATCH, sums, b32 + c, qp + c, out_zp,
                                   outp + c);
                    c   += PW_BATCH;  kp  += chi * PW_BATCH;
                }
                for (; c < cho; c++) {
                    tml_dot_prod(sp, kp, (uint32_t)chi, sums);
                    tml_post_i32(sums, b32[c], qp + c, out_zp, outp + c);
                    kp += chi;
                }
            }
        }
        return TML_OK;
    }

    int idx = 0;
    for (int y = 0; y < kh; y++)
        for (int x = 0; x < kw; x++)
            _tml_k_oft[idx++] = y * in->w * chi + x * chi;

    for (int y = 0; y < out->h; y++) {
        int sy0 = sy * y - pt;
        for (int x = 0; x < out->w; x++) {
            int sx0 = sx * x - pl;
            int slow = (sy0 < 0) | (sx0 < 0) | (sy0 + kh > in->h)
                     | (sx0 + kw > in->w);
            int cy = sy0<0?0:(sy0>in->h-1?in->h-1:sy0);
            int cx = sx0<0?0:(sx0>in->w-1?in->w-1:sx0);
            const int8_t *spb = TML_MATP(in, cy, cx, 0);

            if (!slow) {
                int sidx = 0; const int8_t *sp = spb;
                for (int cc = 0; cc < (dmul ? cho : chi); cc++) {
                    for (int k = 0; k < maxk; k++)
                        _tml_sbuf[sidx + k] = sp[_tml_k_oft[k]];
                    sidx += maxk;
                    sp = spb + (dmul ? (cc + 1) / dmul : cc + 1);
                }
            } else {
                memset(_tml_sbuf, (int)in_zp,
                       (size_t)(dmul ? cho * maxk : chi * maxk));
                int ky0 = sy0<0?-sy0:0, kx0 = sx0<0?-sx0:0;
                int ky1 = in->h-sy0>kh?kh:in->h-sy0;
                int kx1 = in->w-sx0>kw?kw:in->w-sx0;
                int sidx = 0; const int8_t *sp = spb;
                for (int cc = 0; cc < (dmul ? cho : chi); cc++) {
                    for (int ky = ky0; ky < ky1; ky++)
                        for (int kx = kx0; kx < kx1; kx++)
                            _tml_sbuf[sidx + ky * kw + kx]
                                = sp[_tml_k_oft[ky * kw + kx]];
                    sidx += maxk;
                    sp = spb + (dmul ? (cc + 1) / dmul : cc + 1);
                }
            }

            const int8_t *sp = _tml_sbuf;
            for (int c = 0; c < cho; c++) {
                int32_t sum;
                tml_dot_prod(sp, w8 + c * chi * maxk,
                             (uint32_t)(maxk * chi), &sum);
                if (act == TML_ACT_RELU)
                    tml_post_i32_n_relu(1, &sum, b32 + c, qp + c,
                                        out_zp, outp + c);
                else
                    tml_post_i32(&sum, b32[c], qp + c, out_zp, outp + c);
                if (dmul) sp += maxk;
            }
            outp += cho;
        }
    }
    return TML_OK;
}

/* ──────── GAP ──────── */

static int _tml_gap(const tml_tensor_t *in, tml_tensor_t *out,
                     const tml_qp_t *qp, int32_t in_zp, int32_t out_zp)
{
    int area = in->h * in->w;
    for (int c = 0; c < out->c; c++) {
        int32_t sum = 0;
        const int8_t *p = in->data + c;
        for (int y = 0; y < in->h; y++)
            for (int x = 0; x < in->w; x++) { sum += (int32_t)*p; p += out->c; }
        int64_t acc = (int64_t)(sum - in_zp * area) * (int64_t)qp->mult;
        acc = tml_rshift_round(acc, qp->shift);
        int32_t v = (int32_t)acc + out_zp;
        out->data[c] = tml_clamp_i8(v);
    }
    return TML_OK;
}

/* ──────── FC ──────── */

static int _tml_fc(const tml_tensor_t *in, tml_tensor_t *out,
                    const uint8_t *w, const uint8_t *b,
                    const tml_qp_t *qp, int32_t out_zp,
                    int32_t in_zp)
{
    (void)in_zp;
    int mi = in->c, mo = out->c;
    const int8_t *w8 = (const int8_t *)w;
    const int32_t *b32 = (const int32_t *)b;
    for (int c = 0; c < mo; c++) {
        int32_t sum;
        tml_dot_prod(in->data, w8 + c * mi, (uint32_t)mi, &sum);
        tml_post_i32(&sum, b32[c], qp, out_zp, out->data + c);
    }
    return TML_OK;
}

/* ──────── ARGMAX ──────── */

static int _tml_argmax(const tml_tensor_t *in, tml_tensor_t *out)
{
    int n = (int)in->c, imax = 0;
    int8_t vmax = in->data[0];
    for (int i = 1; i < n; i++)
        if (in->data[i] > vmax) { vmax = in->data[i]; imax = i; }
    for (int i = 0; i < n; i++)
        out->data[i] = (i == imax) ? (int8_t)127 : (int8_t)0;
    return TML_OK;
}

/* ──────── ADD ──────── */

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
