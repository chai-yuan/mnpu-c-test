#include "model.h"
#include <riscv_vector.h>
#include <string.h>

/* ==========================================================================
 * Widen-int8-to-int16 helpers
 *
 * __riscv_vwadd_vx(…, 0, …) is optimized by GCC to vsext.vf2, which is
 * NOT legal on Zve32x.  We add a non‑zero offset and then subtract it
 * back, which the compiler cannot fold into a single vsext.
 * ========================================================================== */
#define WIDEN_DELTA 1 /* any value 1…127, must fit in int8 */

static inline vint16m2_t _widen_i8_to_i16(vint8m1_t v, size_t vl) {
    vint16m2_t w = __riscv_vwadd_vx_i16m2(v, WIDEN_DELTA, vl);
    return __riscv_vsub_vx_i16m2(w, WIDEN_DELTA, vl);
}

static inline vint16m2_t _widen_sub_zp(vint8m1_t v, int16_t zp, size_t vl) {
    vint16m2_t w = __riscv_vwadd_vx_i16m2(v, WIDEN_DELTA, vl);
    return __riscv_vsub_vx_i16m2(w, (int16_t)(WIDEN_DELTA + zp), vl);
}
static void _int8_dense_layer(const int8_t *input, int32_t z_in, const int8_t *weight, const int32_t *bias,
                              const int32_t *q_mult, const int32_t *q_shift, int32_t z_out, int8_t *output, int in_c,
                              int out_c, int has_relu, Int8Timing *timing, int layer) {
    clock_t t_vec = 0; /* accumulated vector dot-product time */
    clock_t t_req = 0; /* accumulated reduction + requantize time */

    /* ── Fast path: zero-point fits in int16 ─────────────────────────── */
    if (z_in >= INT16_MIN && z_in <= INT16_MAX) {

        for (int c = 0; c < out_c; c++) {
            const int8_t *w_row = weight + c * in_c;

            /* Initialize int32 accumulator vector to zero */
            size_t     vl_max = __riscv_vsetvl_e16m2(in_c);
            vint32m4_t vacc   = __riscv_vmv_v_x_i32m4(0, vl_max);

            /* ── Vectorized dot-product strip-mining loop ─────────── */
            {
                clock_t t0 = clock();
                int     j  = 0;
                int     n  = in_c;
                size_t  vl;
                for (; n > 0; n -= vl, j += (int)vl) {
                    vl = __riscv_vsetvl_e16m2(n);

                    /* input int8 → widen to int16 + adjust zero-point */
                    vint8m1_t  v_in8  = __riscv_vle8_v_i8m1(input + j, vl);
                    vint16m2_t v_in16 = _widen_sub_zp(v_in8, (int16_t)z_in, vl);

                    /* weight int8 → widen to int16 (no zero-point) */
                    vint8m1_t  v_w8  = __riscv_vle8_v_i8m1(w_row + j, vl);
                    vint16m2_t v_w16 = _widen_i8_to_i16(v_w8, vl);

                    /* Widening MAC:  vacc[i] += v_in16[i] * v_w16[i]  */
                    vacc = __riscv_vwmacc_vv_i32m4(vacc, v_in16, v_w16, vl);
                }
                t_vec += (clock() - t0);
            }

            /* ── Reduce partial sums + bias ──────────────────────── */
            {
                clock_t t0 = clock();

                int32_t partials[8]; /* enough for VLMAX ≤ 8 */
                __riscv_vse32_v_i32m4(partials, vacc, vl_max);

                int64_t dot = (int64_t)bias[c];
                for (int k = 0; k < (int)vl_max; k++) {
                    dot += (int64_t)partials[k];
                }

                /* Requantize */
                int32_t rq  = _multiply_by_quant_mult((int32_t)dot, q_mult[c], q_shift[c]);
                int32_t val = rq + z_out;

                /* Clamp */
                int32_t lo = has_relu ? z_out : (int32_t)INT8_MIN;
                if (val < lo)
                    val = lo;
                if (val > INT8_MAX)
                    val = INT8_MAX;
                output[c] = (int8_t)val;

                t_req += (clock() - t0);
            }
        }

        if (timing) {
            timing->vec_dot[layer] = t_vec;
            timing->requant[layer] = t_req;
        }
        return;
    }

    /* ── Scalar fallback (unlikely: zero-point outside int16 range) ──── */
    for (int c = 0; c < out_c; c++) {
        clock_t t0  = clock();
        int64_t acc = (int64_t)bias[c];
        for (int j = 0; j < in_c; j++) {
            int32_t xv = (int32_t)input[j] - z_in;
            acc += (int64_t)xv * (int64_t)weight[c * in_c + j];
        }
        t_vec += (clock() - t0);

        t0          = clock();
        int32_t rq  = _multiply_by_quant_mult((int32_t)acc, q_mult[c], q_shift[c]);
        int32_t val = rq + z_out;
        int32_t lo  = has_relu ? z_out : (int32_t)INT8_MIN;
        if (val < lo)
            val = lo;
        if (val > INT8_MAX)
            val = INT8_MAX;
        output[c] = (int8_t)val;
        t_req += (clock() - t0);
    }

    if (timing) {
        timing->vec_dot[layer] = t_vec;
        timing->requant[layer] = t_req;
    }
}

/* ==========================================================================
 * Full forward pass
 * ========================================================================== */

int int8_forward(const Int8Model *model, Int8RunState *state, const int8_t *input, int8_t *output, Int8Timing *timing) {
    /* Ping-pong activation buffers: buf[0] = A, buf[1] = B */
    int8_t *act_a = state->buf;
    int8_t *act_b = state->buf + state->max_dim;

    const int8_t *layer_in = input;
    int32_t       z_in     = model->config.input_zero_point;

    if (timing) {
        memset(timing, 0, sizeof(*timing));
        timing->num_layers = model->num_layers;
    }

    for (int l = 0; l < model->num_layers; l++) {
        const Int8Layer *ly        = &model->layers[l];
        int8_t          *layer_out = (l == model->num_layers - 1) ? output : ((l % 2 == 0) ? act_b : act_a);
        int              has_relu  = (l < model->num_layers - 1);

        clock_t t0 = timing ? clock() : 0;
        _int8_dense_layer(layer_in, z_in, ly->weight, ly->bias, ly->requant_multiplier, ly->requant_shift,
                          ly->out_zero_point, layer_out, ly->in_features, ly->out_features, has_relu, timing, l);
        clock_t t1 = timing ? clock() : 0;

        if (timing) {
            timing->per_layer[l] = t1 - t0;
        }

        /* Prepare for next layer */
        layer_in = layer_out;
        z_in     = ly->out_zero_point;
    }

    return int8_argmax(output, model->layers[model->num_layers - 1].out_features);
}
