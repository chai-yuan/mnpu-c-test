/*
 * Fruit CNN inference example (runtime/memory-mapped input).
 *
 * Architecture: Conv2D(16,3,s2) -> Conv2D(32,3,s2) -> Conv2D(64,3,s2)
 *               -> GAP -> Dense(5,softmax)
 *
 * Model is baked-in as a const array (no file I/O, no malloc).
 * A single input buffer is read from a fixed memory address on every loop.
 * The runtime uses arena allocation - a single static buffer.
 */

#include "out/fruit_cnn_int8.bin.h"
#include "tinyml/tinyml.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define ARENA_SIZE (70 * 1024)

/* Base address of the sensor / input memory buffer */
#ifndef INPUT_BASE_ADDR
#define INPUT_BASE_ADDR 0xa0000000
#endif

static const char *TEST_LABEL_NAMES[] = {
    "Apple", "Banana", "Grape", "Orange", "Strawberry"
};

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

    printf("TinyML Fruit CNN (INT8) – memory-mapped input\n");
    printf("  Classes: Apple, Banana, Grape, Orange, Strawberry\n");
    printf("  Input  : %d elements\n", in_sz);
    printf("  Output : %d elements\n", out_sz);
    printf("  Input base address: 0x%08X\n\n", (unsigned)INPUT_BASE_ADDR);

    int8_t output[5];
    unsigned long loop_cnt = 0;

    while (1) {
        const int8_t *input_data = (const int8_t *)INPUT_BASE_ADDR;

        clock_t start = clock();
        int ret = TinyML_Run(ml, input_data, output);
        clock_t end   = clock();
        clock_t ticks = end - start;

        if (ret != TML_OK) {
            printf("[loop %lu] ERR: inference returned %d\n", loop_cnt, ret);
        } else {
            /* softmax output uses argmax with 127/0 encoding */
            int predicted = -1;
            for (int j = 0; j < out_sz; j++) {
                if (output[j] == 127) {
                    predicted = j;
                    break;
                }
            }

            unsigned long us   = (unsigned long)ticks;
            unsigned long ms   = us / 1000;
            unsigned long frac = us % 1000;

            printf("[loop %lu] pred=%d (%s)  time=%lu.",
                   loop_cnt,
                   predicted, (predicted >= 0 && predicted < 5) ? TEST_LABEL_NAMES[predicted] : "?",
                   ms);
            if (frac < 100) printf("0");
            if (frac < 10)  printf("0");
            printf("%lu ms\n", frac);
        }

        loop_cnt++;
    }

    /* never reached */
    TinyML_Destroy(ml);
    return 0;
}
