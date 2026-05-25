/*
 * INT8 MNIST inference — embedded test (pure integer, no dynamic allocation).
 * Model + image are baked-in as const arrays.
 */
#include "lib/arena.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "out/image.bin.h"
#include "out/model.bin.h"

static int load_model_from_memory(const unsigned char *data, size_t size,
                                  Int8Model *model, Arena arena,
                                  Int8RunState *state) {
    if (size < (size_t)INT8_HEADER_SIZE) {
        fprintf(stderr, "Error: model data too small\n");
        return -1;
    }

    memset(model, 0, sizeof(*model));
    int ret = int8_parse_header(data, &model->config);
    if (ret != 0) {
        switch (ret) {
        case -1: fprintf(stderr, "Error: bad magic in model data\n"); break;
        case -2: fprintf(stderr, "Error: unsupported model version\n"); break;
        case -3: fprintf(stderr, "Error: too many layers\n");          break;
        default: fprintf(stderr, "Error: parse header code %d\n", ret);break;
        }
        return -1;
    }

    /* setup_layers points into the model_bin[] data (cast away const) */
    int8_setup_layers(model, (unsigned char *)data);

    if (int8_runstate_init(state, model, arena) != 0) {
        fprintf(stderr, "Error: arena exhausted\n");
        return -1;
    }

    return 0;
}

int main(void) {
    Int8Model    model;
    Int8RunState state;
    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));

    /* Create arena */
    enum { ARENA_SIZE = 8192 };
    unsigned char arena_buf[ARENA_SIZE];
    Arena         arena = Arena_Create(arena_buf, ARENA_SIZE);
    if (!arena) {
        fprintf(stderr, "Error: arena create failed\n");
        return 1;
    }

    /* Load model */
    if (load_model_from_memory(model_bin, model_bin_len,
                               &model, arena, &state) != 0) {
        return 1;
    }

    /* Validate image size */
    if (model.config.input_dim != (int)image_bin_len) {
        fprintf(stderr, "Error: image size %u != config input_dim %d\n",
                image_bin_len, model.config.input_dim);
        return 1;
    }

    /* Run inference (pure integer path) */
    int8_t output[16];
    clock_t t0 = clock();
    int     pred = int8_forward(&model, &state, (const int8_t *)image_bin, output);
    clock_t t1 = clock();
    printf("%d (%ld us)\n", pred, (long)(t1 - t0));

    Arena_Reset(arena);
    return 0;
}
