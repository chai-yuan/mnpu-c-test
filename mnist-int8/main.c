/*
 * INT8 MNIST inference — command-line interface.
 * Usage: mnist-int8 model.bin image.bin
 *        mnist-int8 -r model.bin image.raw   (raw uint8, auto-quantize)
 */
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_file(const char *filename, size_t *out_sz) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror(filename); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) { perror(filename); fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    unsigned char *p = (unsigned char *)malloc((size_t)sz);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) {
        free(p); fclose(f); return NULL;
    }
    fclose(f);
    *out_sz = (size_t)sz;
    return p;
}

static void usage(const char *p) {
    fprintf(stderr, "Usage: %s [-r] model.bin image_file\n", p);
}

int main(int argc, char *argv[]) {
    int raw = 0;
    const char *mp, *ip;
    if (argc == 4 && strcmp(argv[1], "-r") == 0) {
        raw = 1; mp = argv[2]; ip = argv[3];
    } else if (argc == 3) {
        mp = argv[1]; ip = argv[2];
    } else {
        usage(argv[0]); return 1;
    }

    /* Load model */
    size_t msz = 0;
    unsigned char *md = read_file(mp, &msz);
    if (!md) return 1;

    Int8Model  m; memset(&m, 0, sizeof(m));
    if (int8_parse_header(md, &m.config) != 0) {
        fprintf(stderr, "Error: bad model header\n");
        free(md); return 1;
    }
    int8_setup_layers(&m, md);

    /* Run state */
    unsigned char arena[8192];
    Int8RunState  s;
    if (int8_runstate_init(&s, &m, arena, sizeof(arena)) != 0) {
        fprintf(stderr, "Error: arena too small\n");
        free(md); return 1;
    }

    /* Load image */
    int8_t img[784];
    if (raw) {
        size_t isz; unsigned char *id = read_file(ip, &isz);
        if (!id) { free(md); return 1; }
        if ((int)isz != m.config.input_dim) {
            fprintf(stderr, "Error: wrong image size\n");
            free(id); free(md); return 1;
        }
        for (int i = 0; i < m.config.input_dim; i++) {
            float v = (float)id[i];
            int8_quantize_input(&v, &img[i], 1,
                                m.config.input_scale,
                                m.config.input_zero_point);
        }
        free(id);
    } else {
        size_t isz; unsigned char *id = read_file(ip, &isz);
        if (!id) { free(md); return 1; }
        if ((int)isz != m.config.input_dim) {
            fprintf(stderr, "Error: wrong image size\n");
            free(id); free(md); return 1;
        }
        memcpy(img, id, (size_t)m.config.input_dim);
        free(id);
    }

    int8_t out[16];
    int pred = int8_forward(&m, &s, img, out);
    printf("%d\n", pred);

    free(md);
    return 0;
}
