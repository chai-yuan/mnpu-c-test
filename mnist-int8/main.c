/*
 * INT8 MNIST inference — command-line interface (pure integer).
 * Usage: mnist-int8 model.bin image.bin
 *
 * Both files must be pre-quantized int8 binary — no float operations.
 */
#include "lib/arena.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_file(const char *filename, size_t *out_sz) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror(filename);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) {
        perror(filename);
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);
    unsigned char *p = (unsigned char *)malloc((size_t)sz);
    if (!p) {
        fclose(f);
        return NULL;
    }
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) {
        free(p);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_sz = (size_t)sz;
    return p;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s model.bin image.bin\n", argv[0]);
        return 1;
    }
    const char *mp = argv[1];
    const char *ip = argv[2];

    /* Load model */
    size_t         msz = 0;
    unsigned char *md  = read_file(mp, &msz);
    if (!md)
        return 1;

    Int8Model m;
    memset(&m, 0, sizeof(m));
    if (int8_parse_header(md, &m.config) != 0) {
        fprintf(stderr, "Error: bad model header\n");
        free(md);
        return 1;
    }
    int8_setup_layers(&m, md);

    /* Run state */
    unsigned char arena_buf[8192];
    Arena         arena = Arena_Create(arena_buf, sizeof(arena_buf));
    if (!arena) {
        fprintf(stderr, "Error: arena create failed\n");
        free(md);
        return 1;
    }
    Int8RunState s;
    if (int8_runstate_init(&s, &m, arena) != 0) {
        fprintf(stderr, "Error: arena too small\n");
        free(md);
        return 1;
    }

    /* Load image (pre-quantized int8, no float ops) */
    int8_t img[784];
    {
        size_t         isz;
        unsigned char *id = read_file(ip, &isz);
        if (!id) {
            free(md);
            return 1;
        }
        if ((int)isz != m.config.input_dim) {
            fprintf(stderr, "Error: wrong image size %zu, expected %d\n", isz, m.config.input_dim);
            free(id);
            free(md);
            return 1;
        }
        memcpy(img, id, (size_t)m.config.input_dim);
        free(id);
    }

    int8_t out[16];
    int    pred = int8_forward(&m, &s, img, out);
    printf("%d\n", pred);

    free(md);
    return 0;
}
