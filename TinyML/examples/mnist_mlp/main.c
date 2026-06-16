/*
 * MNIST MLP inference example using the TinyML runtime.
 *
 * Model + images are baked-in as const arrays (no file I/O, no malloc).
 */
#include "lib/arena.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tinyml.h"
#include "out/mnist_mlp_int8.h"
#include "out/test_images.h"

/* Arena backing storage (MDL_BUF_LEN from the model header + headroom) */
static uint8_t s_arena_buf[MDL_BUF_LEN + 1024];

/* ------------------------------------------------------------------ */
/*  Helper: argmax on int8 array                                       */
/* ------------------------------------------------------------------ */
static int argmax_int8(const int8_t *data, int len) {
    int    idx     = 0;
    int8_t max_val = data[0];
    for (int i = 1; i < len; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            idx     = i;
        }
    }
    return idx;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */
int main(void) {
    ArenaHandle arena = Arena_Create(s_arena_buf, sizeof(s_arena_buf));
    if (!arena) {
        fprintf(stderr, "Error: arena create failed\n");
        return 1;
    }

    /* memory_if lives in main's frame — same lifetime as TinyML handle.
     * TinyML_Create stores the pointer (tinyml.c:77), so it must not be
     * a stack variable in a shorter-lived scope. */
    memory_if mem = Arena_GetMemoryIf(arena);
    TinyML   *ml  = TinyML_Create(&mem, mdl_data);
    if (!ml) {
        fprintf(stderr, "Error: failed to create TinyML runtime (bad model?)\n");
        return 1;
    }

    int input_size  = TinyML_GetInputSize(ml);
    int output_size = TinyML_GetOutputSize(ml);
    printf("Model loaded from memory\n");
    printf("  input size  = %d\n", input_size);
    printf("  output size = %d\n\n", output_size);

    int8_t output_buf[16];
    int    total = 0, correct = 0;

    for (int i = 0; i < TEST_IMAGE_COUNT; ++i) {
        const test_image_t *img = &test_images[i];

        if (img->size != input_size) {
            fprintf(stderr, "Warning: image '%s' size %d != expected %d\n", img->name, img->size, input_size);
        }

        int rc = TinyML_Run(ml, img->data, output_buf);
        if (rc != TML_OK) {
            fprintf(stderr, "Inference failed with error code %d\n", rc);
            continue;
        }

        int pred = argmax_int8(output_buf, output_size);
        int ok   = (pred == img->label) ? 1 : 0;
        if (ok)
            ++correct;
        ++total;

        printf("%-35s -> predicted %d (label %d %s)\n", img->name, pred, img->label, ok ? "OK" : "WRONG");
    }

    if (total > 0) {
        printf("\nSummary: %d/%d correct (%d%%)\n", correct, total,
               (correct * 100) / total);
    }

    TinyML_Destroy(ml);
    Arena_Reset(arena);
    return 0;
}
