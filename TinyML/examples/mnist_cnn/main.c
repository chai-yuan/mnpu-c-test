/*
 * MNIST CNN inference example using the TinyML runtime.
 *
 * Architecture: Conv2D(8,3,s1) → Conv2D(8,3,s2) → Conv2D(16,3,s2)
 *               → GAP → Dense(10,softmax)
 *
 * Model + images are baked-in as const arrays (no file I/O, no malloc).
 * The runtime uses arena allocation – a single static buffer.
 */

#include "out/mnist_cnn_int8.bin.h"
#include "out/test_images.bin.h"
#include "tinyml/tinyml.h"
#include <stdio.h>
#include <string.h>

#define ARENA_SIZE 20480

static uint8_t arena_buf[ARENA_SIZE];

int main(void) {
    ArenaHandle arena = Arena_Create(arena_buf, ARENA_SIZE);
    if (!arena) {
        printf("ERR: arena creation failed\n");
        return 1;
    }

    TinyMLHandle ml = TinyML_Create(arena, mdl_data);
    if (!ml) {
        printf("ERR: model loading failed\n");
        return 1;
    }

    int in_sz  = TinyML_GetInputSize(ml);
    int out_sz = TinyML_GetOutputSize(ml);

    printf("TinyML MNIST CNN (INT8)\n");
    printf("  Input  : %d elements\n", in_sz);
    printf("  Output : %d elements\n", out_sz);
    printf("  Images : %d\n\n", TEST_IMAGE_COUNT);

    int    correct = 0;
    int8_t output[10];

    for (int i = 0; i < TEST_IMAGE_COUNT; i++) {
        const test_image_t *img = &test_images[i];

        int ret = TinyML_Run(ml, img->data, output);
        if (ret != TML_OK) {
            printf("[%d] ERR: inference returned %d\n", i, ret);
            continue;
        }

        /* softmax output uses argmax with 127/0 encoding */
        int predicted = -1;
        for (int j = 0; j < out_sz; j++) {
            if (output[j] == 127) {
                predicted = j;
                break;
            }
        }

        int match = (predicted == img->label);
        correct += match;

        printf("[%d] %s  pred=%d  label=%d  %s\n", i, img->name, predicted, img->label, match ? "OK" : "MISMATCH");
    }

    printf("\nAccuracy: %d/%d\n", correct, TEST_IMAGE_COUNT);

    TinyML_Destroy(ml);

    return correct == TEST_IMAGE_COUNT ? 0 : 1;
}
