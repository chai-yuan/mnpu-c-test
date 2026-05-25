#include "model.h"

/* ==========================================================================
 * Single INT8 Dense layer (scalar implementation)
 * ========================================================================== */

static void _int8_dense_layer(const int8_t *input, int32_t z_in, const int8_t *weight, const int32_t *bias,
                              const int32_t *q_mult, const int32_t *q_shift, int32_t z_out, int8_t *output, int in_c,
                              int out_c, int has_relu, Int8Timing *timing, int layer) {
    /*
     * For each output channel c:
     *   1. dot-product: sum_j (input[j] - z_in) × weight[c*in_c + j]  +  bias[c]
     *      (全部用 int64 防止溢出)
     *   2. requantize via MultiplyByQuantizedMultiplier
     *   3. add z_out, clamp to int8
     *      - has_relu: clamp to [z_out, 127]  (ReLU fused layer)
     *      - no relu:  clamp to [-128, 127]
     */
    (void)timing;
    (void)layer;
    for (int c = 0; c < out_c; c++) {
        /* Dot product */
        int64_t acc = (int64_t)bias[c];
        for (int j = 0; j < in_c; j++) {
            int32_t xv = (int32_t)input[j] - z_in;
            acc += (int64_t)xv * (int64_t)weight[c * in_c + j];
        }

        /* Requantize */
        int32_t rq = _multiply_by_quant_mult((int32_t)acc, q_mult[c], q_shift[c]);

        /* Add output zero point */
        int32_t val = rq + z_out;

        /* Clamp */
        int32_t lo = has_relu ? z_out : (int32_t)INT8_MIN;
        if (val < lo)
            val = lo;
        if (val > INT8_MAX)
            val = INT8_MAX;
        output[c] = (int8_t)val;
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

    for (int l = 0; l < model->num_layers; l++) {
        const Int8Layer *ly        = &model->layers[l];
        int8_t          *layer_out = (l == model->num_layers - 1) ? output : ((l % 2 == 0) ? act_b : act_a);
        int              has_relu  = (l < model->num_layers - 1);

        _int8_dense_layer(layer_in, z_in, ly->weight, ly->bias, ly->requant_multiplier, ly->requant_shift,
                          ly->out_zero_point, layer_out, ly->in_features, ly->out_features, has_relu, timing, l);

        /* Prepare for next layer */
        layer_in = layer_out;
        z_in     = ly->out_zero_point;
    }

    return int8_argmax(output, model->layers[model->num_layers - 1].out_features);
}
