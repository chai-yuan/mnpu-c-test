/* TinyML Runtime – INT8 inference engine implementation.
 *
 * Handles model loading, ping‑pong buffer management, and layer dispatch.
 * Heavy compute operators are delegated to arch/soft.c.
 */
#include "tinyml.h"
#include "internal.h"
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Arch helpers – declared here, defined in arch/{soft,zve32x,...}.c  */
/* ------------------------------------------------------------------ */
int tml_arch_conv2d(const tml_tensor_t *in, tml_tensor_t *out, const int8_t *w, const int32_t *b, const int32_t *qp,
                    int8_t kw, int8_t kh, int8_t sx, int8_t sy, int8_t dx, int8_t dy, uint16_t act, uint8_t pt,
                    uint8_t pb, uint8_t pl, uint8_t pr, uint32_t dmul, int32_t in_zp, int32_t out_zp);

int tml_arch_fc(const tml_tensor_t *in, tml_tensor_t *out, const int8_t *w, const int32_t *b, int32_t qp_mult,
                int32_t qp_shift, int32_t out_zp);

int tml_arch_gap(const tml_tensor_t *in, tml_tensor_t *out, int32_t qp_mult, int32_t qp_shift, int32_t in_zp,
                 int32_t out_zp);

int tml_arch_softmax(const tml_tensor_t *in, tml_tensor_t *out, int32_t in_zp, int32_t out_zp);

int tml_arch_reshape(const tml_tensor_t *in, tml_tensor_t *out);

int tml_arch_add(const tml_tensor_t *in0, const tml_tensor_t *in1, tml_tensor_t *out, int32_t qp0_mult,
                 int32_t qp0_shift, int32_t in_zp0, int32_t qp1_mult, int32_t qp1_shift, int32_t in_zp1,
                 int32_t out_zp);

/* ------------------------------------------------------------------ */
/*  Create / Destroy                                                   */
/* ------------------------------------------------------------------ */
TinyMLHandle TinyML_Create(ArenaHandle arena, const uint8_t *model) {
    const tml_bin_t *bin = (const tml_bin_t *)model;

    /* validate magic "MAIX" */
    if (bin->magic != ((uint32_t)TML_MAGIC_3 << 24 | (uint32_t)TML_MAGIC_2 << 16 | (uint32_t)TML_MAGIC_1 << 8 |
                       (uint32_t)TML_MAGIC_0))
        return NULL;

    if (bin->mdl_type != 0) /* INT8 only */
        return NULL;

    /* allocate handle from arena */
    struct TinyML *ml = Arena_Alloc(arena, sizeof(struct TinyML));
    if (!ml)
        return NULL;

    ml->bin     = bin;
    ml->arena   = arena;
    ml->own_buf = 0;
    ml->buf     = NULL;

    /* allocate ping‑pong buffer */
    if (bin->buf_size > 0) {
        ml->buf = Arena_Alloc(arena, TML_ALIGN_UP(bin->buf_size));
        if (!ml->buf)
            return NULL;
        ml->own_buf = 1;
    }
    return ml;
}

void TinyML_Destroy(TinyMLHandle self) {
    /* Arena‑based: no manual free; user resets arena. */
    (void)self;
}

/* ------------------------------------------------------------------ */
/*  Size queries                                                       */
/* ------------------------------------------------------------------ */

int TinyML_GetInputSize(TinyMLHandle self) {
    return (int)self->bin->in_dims[1] * self->bin->in_dims[2] * self->bin->in_dims[3];
}

int TinyML_GetOutputSize(TinyMLHandle self) {
    return (int)self->bin->out_dims[1] * self->bin->out_dims[2] * self->bin->out_dims[3];
}

/* ------------------------------------------------------------------ */
/*  Helper: fill tensor descriptor from dims array                     */
/* ------------------------------------------------------------------ */
static void dims_to_tensor(tml_tensor_t *t, int8_t *data, const uint16_t d[4]) {
    t->dims = d[0];
    t->h    = d[1];
    t->w    = d[2];
    t->c    = d[3];
    t->data = data;
}

/* ------------------------------------------------------------------ */
/*  Run inference                                                      */
/* ------------------------------------------------------------------ */
int TinyML_Run(TinyMLHandle self, const int8_t *input, int8_t *output) {
    tml_tensor_t   tin, tin1, tout;
    const uint8_t *layer_body;
    int            ret = TML_OK;

    /* point to first layer (right after 64‑byte model header) */
    layer_body = (const uint8_t *)(self->bin) + TML_HEADER_SIZE;

    for (int li = 0; li < (int)self->bin->layer_cnt; li++) {

        const tml_head_t *h = (const tml_head_t *)layer_body;

        /* --- input tensor --- */
        if (li == 0) {
            /* first layer: copy user input → buffer */
            int in_sz = TinyML_GetInputSize(self);
            memcpy(self->buf + h->in_oft, input, (size_t)in_sz);
            dims_to_tensor(&tin, (int8_t *)(self->buf + h->in_oft), h->in_dims);
        } else {
            dims_to_tensor(&tin, (int8_t *)(self->buf + h->in_oft), h->in_dims);
        }

        /* --- output tensor --- */
        dims_to_tensor(&tout, (int8_t *)(self->buf + h->out_oft), h->out_dims);

        /* --- dispatch layer --- */
        switch (h->type) {

        case TML_CONV2D:
        case TML_DWCONV2D: {
            const tml_conv_t *c  = (const tml_conv_t *)h;
            const int8_t     *w  = (const int8_t *)(layer_body + c->w_oft);
            const int32_t    *b  = (const int32_t *)(layer_body + c->b_oft);
            const int32_t    *qp = (const int32_t *)(layer_body + c->ws_oft);

            ret = tml_arch_conv2d(&tin, &tout, w, b, qp, (int8_t)c->kernel_w, (int8_t)c->kernel_h, (int8_t)c->stride_w,
                                  (int8_t)c->stride_h, (int8_t)c->dilation_w, (int8_t)c->dilation_h, c->act, c->pad[0],
                                  c->pad[1], c->pad[2], c->pad[3], c->depth_mul, h->in_zp, h->out_zp);
            break;
        }

        case TML_GAP: {
            ret = tml_arch_gap(&tin, &tout, h->qp_mult, h->qp_shift, h->in_zp, h->out_zp);
            break;
        }

        case TML_FC: {
            const tml_fc_t *fc = (const tml_fc_t *)h;
            const int8_t   *w  = (const int8_t *)(layer_body + fc->w_oft);
            const int32_t  *b  = (const int32_t *)(layer_body + fc->b_oft);
            ret                = tml_arch_fc(&tin, &tout, w, b, h->qp_mult, h->qp_shift, h->out_zp);
            break;
        }

        case TML_SOFTMAX: {
            ret = tml_arch_softmax(&tin, &tout, h->in_zp, h->out_zp);
            break;
        }

        case TML_RESHAPE: {
            ret = tml_arch_reshape(&tin, &tout);
            break;
        }

        case TML_ADD: {
            const tml_add_t *a = (const tml_add_t *)h;
            /* second input tensor (KEEP) */
            dims_to_tensor(&tin1, (int8_t *)(self->buf + a->in_oft1), h->in_dims);
            ret = tml_arch_add(&tin, &tin1, &tout, h->qp_mult, h->qp_shift, h->in_zp, a->qp1_mult, a->qp1_shift,
                               a->in_zp1, h->out_zp);
            break;
        }

        default:
            ret = TML_ERR_LAYER;
            break;
        }

        if (ret != TML_OK)
            return ret;

        /* advance to next layer */
        layer_body += h->size;
    }

    /* copy final output to user buffer */
    {
        const uint8_t *final_body = (const uint8_t *)(self->bin) + TML_HEADER_SIZE;
        for (int li = 0; li < (int)self->bin->layer_cnt - 1; li++) {
            const tml_head_t *h = (const tml_head_t *)final_body;
            final_body += h->size;
        }
        const tml_head_t *last   = (const tml_head_t *)final_body;
        int               out_sz = (int)last->out_dims[1] * last->out_dims[2] * last->out_dims[3];
        memcpy(output, self->buf + last->out_oft, (size_t)out_sz);
    }

    return TML_OK;
}
