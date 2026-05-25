#include "model.h"
#include <string.h>

/* ==========================================================================
 * Binary header parser
 * ========================================================================== */

int int8_parse_header(const unsigned char *header, Int8Config *cfg) {
    int offset = 0;

    int32_t magic = *(int32_t *)(header + offset);
    offset += sizeof(int32_t);
    if (magic != INT8_MAGIC) return -1;

    cfg->version = *(int32_t *)(header + offset);
    offset += sizeof(int32_t);
    if (cfg->version != 1) return -2;

    cfg->input_dim = *(int32_t *)(header + offset);
    offset += sizeof(int32_t);
    cfg->num_layers = *(int32_t *)(header + offset);
    offset += sizeof(int32_t);
    if (cfg->num_layers > INT8_MAX_LAYERS) return -3;

    for (int i = 0; i < INT8_MAX_LAYERS; i++) {
        cfg->layer_out_features[i] = *(int32_t *)(header + offset);
        offset += sizeof(int32_t);
    }

    cfg->input_zero_point = *(int32_t *)(header + offset); offset += sizeof(int32_t);
    cfg->output_zero_point= *(int32_t *)(header + offset);

    return 0;
}

/* ==========================================================================
 * Layer setup: walk the binary data and set up pointers
 * ========================================================================== */

int int8_setup_layers(Int8Model *model, unsigned char *data) {
    Int8Config *cfg = &model->config;
    model->num_layers = cfg->num_layers;

    unsigned char *ptr = data + INT8_HEADER_SIZE;

    for (int i = 0; i < cfg->num_layers; i++) {
        int out_c = cfg->layer_out_features[i];
        int in_c  = (i == 0) ? cfg->input_dim : cfg->layer_out_features[i - 1];

        Int8Layer *l = &model->layers[i];
        l->in_features  = in_c;
        l->out_features = out_c;

        l->requant_multiplier = (int32_t *)ptr;       ptr += out_c * sizeof(int32_t);
        l->requant_shift      = (int32_t *)ptr;       ptr += out_c * sizeof(int32_t);
        l->out_zero_point     = *(int32_t *)ptr;      ptr += sizeof(int32_t);
        l->bias               = (int32_t *)ptr;       ptr += out_c * sizeof(int32_t);
        l->weight             = (int8_t *)ptr;        ptr += out_c * in_c * sizeof(int8_t);
    }
    return 0;
}

/* ==========================================================================
 * Run-state allocation (via runtime Arena)
 * ========================================================================== */

int int8_runstate_init(Int8RunState *state, const Int8Model *model,
                       Arena arena) {
    /* Compute max dimension width */
    int max_dim = model->config.input_dim;
    for (int i = 0; i < model->num_layers; i++) {
        int d = model->layers[i].out_features;
        if (d > max_dim) max_dim = d;
    }
    state->max_dim = max_dim;

    /* Allocate from arena (automatically 16‑byte aligned) */
    state->acc = Arena_Alloc(arena, (size_t)max_dim * sizeof(int32_t));
    if (!state->acc) return -1;

    /* Two activation buffers back-to-back for ping-pong */
    state->buf = Arena_Alloc(arena, (size_t)max_dim * sizeof(int8_t) * 2);
    if (!state->buf) return -1;

    return 0;
}

/* ==========================================================================
 * TFLite 纯整型 requantize 原语
 * ========================================================================== */

/** Saturating rounding doubling high-mul:  (a*b + 2^30) >> 31,饱和处理 */
static inline int32_t _sr_doubling_high_mul(int32_t a, int32_t b) {
    /* Handle the INT32_MIN × INT32_MIN edge case described by TFLite */
    if (a == INT32_MIN && b == INT32_MIN)
        return INT32_MAX;
    int64_t ab64  = (int64_t)a * (int64_t)b;
    int64_t nudge = ab64 >= 0 ? (1LL << 30) : (1LL - (1LL << 30));
    return (int32_t)((ab64 + nudge) >> 31);
}

/** Rounding divide by power of 2:  x / 2^exp  (round to nearest, ties even) */
static inline int32_t _rounding_divide_by_pot(int32_t x, int32_t exp) {
    if (exp == 0) return x;
    int32_t mask      = (1 << exp) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> exp) + (remainder > threshold ? 1 : 0);
}

/** TFLite MultiplyByQuantizedMultiplier — 纯整型从 int32 到 int32 */
static int32_t _multiply_by_quant_mult(int32_t x,
                                       int32_t quant_mult, int32_t shift) {
    int32_t left  = shift > 0 ? shift : 0;
    int32_t right = shift > 0 ? 0 : -shift;

    /* Left-shift with int64 to prevent overflow */
    int64_t x64 = (int64_t)x << left;
    /* Saturating doubling high mul */
    int32_t high = _sr_doubling_high_mul((int32_t)x64, quant_mult);
    /* Rounding right shift */
    return _rounding_divide_by_pot(high, right);
}

/* ==========================================================================
 * Single INT8 Dense layer + TFLite requantize
 * ========================================================================== */

static void _int8_dense_layer(const int8_t  *input,
                              int32_t        z_in,
                              const int8_t  *weight,
                              const int32_t *bias,
                              const int32_t *q_mult,
                              const int32_t *q_shift,
                              int32_t        z_out,
                              int8_t        *output,
                              int in_c, int out_c,
                              int has_relu) {
    /*
     * For each output channel c:
     *   1. dot-product: sum_j (input[j] - z_in) × weight[c*in_c + j]  +  bias[c]
     *      (全部用 int64 防止溢出)
     *   2. requantize via MultiplyByQuantizedMultiplier
     *   3. add z_out, clamp to int8
     *      - has_relu: clamp to [z_out, 127]  (ReLU fused layer)
     *      - no relu:  clamp to [-128, 127]
     */
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
        if (val < lo)       val = lo;
        if (val > INT8_MAX) val = INT8_MAX;
        output[c] = (int8_t)val;
    }
}

/* ==========================================================================
 * Full forward pass
 * ========================================================================== */

int int8_forward(const Int8Model *model, Int8RunState *state,
                 const int8_t *input, int8_t *output) {
    /* Ping-pong activation buffers: buf[0] = A, buf[1] = B */
    int8_t *act_a = state->buf;
    int8_t *act_b = state->buf + state->max_dim;

    const int8_t *layer_in  = input;
    int32_t       z_in      = model->config.input_zero_point;

    for (int l = 0; l < model->num_layers; l++) {
        const Int8Layer *ly  = &model->layers[l];
        int8_t *layer_out    = (l == model->num_layers - 1) ? output
                                : ((l % 2 == 0) ? act_b : act_a);
        int has_relu = (l < model->num_layers - 1);

        _int8_dense_layer(layer_in,
                          z_in,
                          ly->weight,
                          ly->bias,
                          ly->requant_multiplier,
                          ly->requant_shift,
                          ly->out_zero_point,
                          layer_out,
                          ly->in_features,
                          ly->out_features,
                          has_relu);

        /* Prepare for next layer */
        layer_in = layer_out;
        z_in     = ly->out_zero_point;
    }

    return int8_argmax(output, model->layers[model->num_layers - 1].out_features);
}

/* ==========================================================================
 * Helpers
 * ========================================================================== */

int int8_argmax(const int8_t *x, int n) {
    int   max_i   = 0;
    int8_t max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_i   = i;
        }
    }
    return max_i;
}
