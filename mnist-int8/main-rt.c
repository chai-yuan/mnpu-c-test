/*
 * INT8 MNIST inference — real-time processing from a fixed memory address.
 * The image data (pre-quantized int8, input_dim bytes) is expected to appear
 * at IMAGE_ADDR. The program loops, reads the image, runs inference, and
 * prints the prediction.
 *
 * Define IMAGE_ADDR to the memory address where image data resides.
 * Example: -DIMAGE_ADDR=0xa0000000
 */
#include "lib/arena.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "out/model.bin.h"

#ifndef IMAGE_ADDR
#define IMAGE_ADDR 0xa0000000
#endif

static int load_model_from_memory(const unsigned char *data, size_t size, Int8Model *model, Arena arena,
                                  Int8RunState *state) {
    if (size < (size_t)INT8_HEADER_SIZE) {
        fprintf(stderr, "Error: model data too small\n");
        return -1;
    }

    memset(model, 0, sizeof(*model));
    int ret = int8_parse_header(data, &model->config);
    if (ret != 0) {
        switch (ret) {
        case -1:
            fprintf(stderr, "Error: bad magic in model data\n");
            break;
        case -2:
            fprintf(stderr, "Error: unsupported model version\n");
            break;
        case -3:
            fprintf(stderr, "Error: too many layers\n");
            break;
        default:
            fprintf(stderr, "Error: parse header code %d\n", ret);
            break;
        }
        return -1;
    }

    int8_setup_layers(model, (unsigned char *)data);

    if (int8_runstate_init(state, model, arena) != 0) {
        fprintf(stderr, "Error: arena exhausted\n");
        return -1;
    }

    return 0;
}

int main(void) {
    printf("IMAGE_ADDR: 0x%x\n", IMAGE_ADDR);

    /* Load model */
    Int8Model    model;
    Int8RunState state;
    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));

    enum { ARENA_SIZE = 8192 };
    unsigned char arena_buf[ARENA_SIZE];
    Arena         arena = Arena_Create(arena_buf, ARENA_SIZE);
    if (!arena) {
        fprintf(stderr, "Error: arena create failed\n");
        return 1;
    }

    if (load_model_from_memory(model_bin, model_bin_len, &model, arena, &state) != 0) {
        return 1;
    }

    int input_dim = model.config.input_dim;

    /* Image data pointer — fixed memory address */
    const int8_t *image = (const int8_t *)IMAGE_ADDR;

    /* Loop: read image from fixed address, run inference, print result */
    int8_t output[16];
    for (int frame = 0;; frame++) {
        int pred = int8_forward(&model, &state, image, output, NULL);
        printf("[%d] prediction: %d\n", frame, pred);
    }

    /* NOTREACHED */
    Arena_Reset(arena);
    return 0;
}
