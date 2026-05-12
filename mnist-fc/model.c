#include "model.h"
#include "lib/arena.h"
#include <string.h>

int parse_header(const unsigned char *header, Config *config) {
    int offset = 0;
    int magic  = *(int *)(header + offset);
    offset += sizeof(int);
    if (magic != 0x4d4c5046) {
        return -1;
    }

    config->version = *(int *)(header + offset);
    offset += sizeof(int);
    if (config->version != 3) {
        return -2;
    }

    config->input_dim = *(int *)(header + offset);
    offset += sizeof(int);
    config->output_dim = *(int *)(header + offset);
    offset += sizeof(int);
    config->num_layers = *(int *)(header + offset);
    offset += sizeof(int);

    if (config->num_layers > MAX_LAYERS) {
        return -3;
    }

    for (int i = 0; i < MAX_LAYERS; i++) {
        config->layer_out_features[i] = *(int *)(header + offset);
        offset += sizeof(int);
    }

    config->input_mean = *(float *)(header + offset);
    offset += sizeof(float);
    config->input_std = *(float *)(header + offset);
    offset += sizeof(float);

    return 0;
}

void setup_weights(const Config *config, MLPWeights *weights, float *weights_data, Arena *arena) {
    weights->num_layers   = config->num_layers;
    weights->weight       = (float **)arena_alloc(arena, config->num_layers * sizeof(float *));
    weights->bias         = (float **)arena_alloc(arena, config->num_layers * sizeof(float *));
    weights->in_features  = (int *)arena_alloc(arena, config->num_layers * sizeof(int));
    weights->out_features = (int *)arena_alloc(arena, config->num_layers * sizeof(int));

    float *ptr      = weights_data;
    int    prev_out = config->input_dim;

    for (int i = 0; i < config->num_layers; i++) {
        int out                  = config->layer_out_features[i];
        int in                   = prev_out;
        weights->in_features[i]  = in;
        weights->out_features[i] = out;
        weights->weight[i]       = ptr;
        ptr += out * in;
        weights->bias[i] = ptr;
        ptr += out;
        prev_out = out;
    }
}

int alloc_runstate(const Config *config, const MLPWeights *weights, RunState *state, Arena *arena) {
    int max_size = config->input_dim;
    for (int i = 0; i < config->num_layers; i++) {
        if (weights->out_features[i] > max_size)
            max_size = weights->out_features[i];
    }

    state->x   = (float *)arena_alloc(arena, max_size * sizeof(float));
    state->y   = (float *)arena_alloc(arena, max_size * sizeof(float));
    state->tmp = (float *)arena_alloc(arena, max_size * sizeof(float));

    if (!state->x || !state->y || !state->tmp) {
        state->x   = NULL;
        state->y   = NULL;
        state->tmp = NULL;
        return -1;
    }

    state->buf_size = max_size;
    return 0;
}

void matmul_add_bias(float *y, const float *x, const float *W, const float *b, int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d_in; j++) {
            sum += W[i * d_in + j] * x[j];
        }
        y[i] = sum + b[i];
    }
}

void relu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0.0f)
            x[i] = 0.0f;
    }
}

int argmax(const float *x, int n) {
    int   max_i   = 0;
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_i   = i;
        }
    }
    return max_i;
}

void forward(const Config *config, const MLPWeights *w, RunState *s, const float *input, float *output) {
    memcpy(s->x, input, config->input_dim * sizeof(float));

    int d_in = config->input_dim;
    for (int l = 0; l < config->num_layers - 1; l++) {
        int d_out = w->out_features[l];
        matmul_add_bias(s->y, s->x, w->weight[l], w->bias[l], d_in, d_out);
        relu(s->y, d_out);
        float *temp = s->x;
        s->x        = s->y;
        s->y        = temp;
        d_in        = d_out;
    }

    int last = config->num_layers - 1;
    matmul_add_bias(output, s->x, w->weight[last], w->bias[last], d_in, w->out_features[last]);
}
