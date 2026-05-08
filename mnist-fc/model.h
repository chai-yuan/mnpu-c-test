#ifndef MODEL_H
#define MODEL_H

typedef struct Arena Arena;

#define MAX_LAYERS 16
#define HEADER_SIZE 512

typedef struct {
    int   input_dim;
    int   output_dim;
    int   num_layers;
    int   layer_out_features[MAX_LAYERS];
    float input_mean;
    float input_std;
    int   version;
} Config;

typedef struct {
    float **weight;
    float **bias;
    int    *in_features;
    int    *out_features;
    int     num_layers;
} MLPWeights;

typedef struct {
    float *x;
    float *y;
    float *tmp;
    int    buf_size;
} RunState;

int  parse_header(const unsigned char *header, Config *config);
void setup_weights(const Config *config, MLPWeights *weights, float *weights_data, Arena *arena);
int  alloc_runstate(const Config *config, const MLPWeights *weights, RunState *state, Arena *arena);

void matmul_add_bias(float *y, const float *x, const float *W, const float *b, int d_in, int d_out);
void relu(float *x, int n);
int  argmax(const float *x, int n);
void forward(const Config *config, const MLPWeights *w, RunState *s, const float *input, float *output);

#endif
