#include "lib/arena.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.bin.h"
#include "model.bin.h"

static int load_image_from_memory(const unsigned char *data, size_t size, float *input, int expected_size, float mean,
                                  float std) {
    if ((size_t)expected_size != size) {
        fprintf(stderr, "Error: image size %zu != expected %d bytes\n", size, expected_size);
        return -1;
    }
    for (int i = 0; i < expected_size; i++) {
        float val = (float)data[i];
        input[i]  = (val / 255.0f - mean) / std;
    }
    return 0;
}

static int load_model_from_memory(const unsigned char *data, size_t size, Config *config, MLPWeights *weights,
                                  RunState *state, Arena *arena) {
    if (size < HEADER_SIZE) {
        fprintf(stderr, "Error: model data too small\n");
        return -1;
    }

    int ret = parse_header(data, config);
    if (ret != 0) {
        switch (ret) {
        case -1:
            fprintf(stderr, "Error: bad magic in model file\n");
            break;
        case -2:
            fprintf(stderr, "Error: unsupported model version %d (expected 3)\n", config->version);
            break;
        case -3:
            fprintf(stderr, "Error: too many layers %d > %d\n", config->num_layers, MAX_LAYERS);
            break;
        default:
            fprintf(stderr, "Error: failed to parse header (code %d)\n", ret);
            break;
        }
        return -1;
    }

    float *weights_ptr = (float *)(data + HEADER_SIZE);
    setup_weights(config, weights, weights_ptr, arena);

    if (alloc_runstate(config, weights, state, arena) != 0) {
        fprintf(stderr, "Error: failed to allocate run state buffers\n");
        return -1;
    }

    return 0;
}

int main(void) {
    Config     config;
    MLPWeights weights;
    RunState   state;

    memset(&config, 0, sizeof(config));
    memset(&weights, 0, sizeof(weights));
    memset(&state, 0, sizeof(state));

    enum { ARENA_SIZE = 65536 };
    unsigned char arena_buf[ARENA_SIZE];
    Arena        *arena = arena_create(arena_buf, ARENA_SIZE);
    if (!arena) {
        fprintf(stderr, "Error: failed to create arena\n");
        return 1;
    }

    if (load_model_from_memory(model_bin, model_bin_len, &config, &weights, &state, arena) != 0) {
        return 1;
    }

    float *image = (float *)arena_alloc(arena, config.input_dim * sizeof(float));
    if (!image) {
        fprintf(stderr, "Error: arena allocation failed for image\n");
        return 1;
    }

    if (load_image_from_memory(image_bin, image_bin_len, image, config.input_dim, config.input_mean,
                               config.input_std) != 0) {
        return 1;
    }

    float output[10];
    forward(&config, &weights, &state, image, output);

    int pred = argmax(output, config.output_dim);
    printf("%d\n", pred);

    arena_reset(arena);
    return 0;
}
