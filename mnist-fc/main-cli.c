/*
 * Simple C inference for MNIST MLP model exported from export.py.
 * Reads a model.bin and a 28x28 raw image (784 bytes, 0‑255),
 * runs the forward pass and prints the predicted digit.
 *
 * Usage: main-cli model.bin image.raw
 */

#include "lib/arena.h"
#include "model.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int load_image(const char *filename, float *input, int expected_size, float mean, float std) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open image %s\n", filename);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    if (size != (size_t)expected_size) {
        fprintf(stderr, "Error: image size %zu != expected %d bytes\n", size, expected_size);
        fclose(f);
        return -1;
    }
    fseek(f, 0, SEEK_SET);
    unsigned char buf[784];
    if (fread(buf, 1, size, f) != size) {
        fprintf(stderr, "Error: failed to read image\n");
        fclose(f);
        return -1;
    }
    fclose(f);

    for (int i = 0; i < expected_size; i++) {
        float val = (float)buf[i];
        input[i]  = (val / 255.0f - mean) / std;
    }
    return 0;
}

static int load_model(const char *filename, Config *config, MLPWeights *weights, RunState *state, int *fd, float **data,
                      ssize_t *file_size, Arena *arena) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return -1;
    }

    unsigned char header[HEADER_SIZE];
    if (fread(header, 1, HEADER_SIZE, file) != HEADER_SIZE) {
        fprintf(stderr, "Error: failed to read %d-byte header\n", HEADER_SIZE);
        fclose(file);
        return -1;
    }
    fclose(file);

    int ret = parse_header(header, config);
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

    *fd = open(filename, O_RDONLY);
    if (*fd == -1) {
        perror("open");
        return -1;
    }
    *file_size = lseek(*fd, 0, SEEK_END);
    *data      = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        perror("mmap");
        close(*fd);
        *fd = -1;
        return -1;
    }

    float *weights_ptr = (float *)((char *)*data + HEADER_SIZE);
    setup_weights(config, weights, weights_ptr, arena);

    if (alloc_runstate(config, weights, state, arena) != 0) {
        fprintf(stderr, "Error: failed to allocate run state buffers\n");
        munmap(*data, *file_size);
        close(*fd);
        *fd = -1;
        return -1;
    }

    return 0;
}

static void free_model_resources(Config *config, MLPWeights *weights, RunState *state, int fd, float *data,
                                 ssize_t file_size) {
    (void)config;
    (void)weights;
    (void)state;
    if (data != MAP_FAILED && data != NULL)
        munmap(data, file_size);
    if (fd != -1)
        close(fd);
}

static void error_usage(void) {
    fprintf(stderr, "Usage: mnist-fc model.bin image.raw\n");
    fprintf(stderr, "       image.raw is a 28x28 grayscale file (784 bytes, values 0-255)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        error_usage();
    }
    const char *model_path = argv[1];
    const char *image_path = argv[2];

    Config     config;
    MLPWeights weights;
    RunState   state;
    int        fd        = -1;
    float     *data      = NULL;
    ssize_t    file_size = 0;

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

    if (load_model(model_path, &config, &weights, &state, &fd, &data, &file_size, arena) != 0) {
        return 1;
    }

    float *image = (float *)arena_alloc(arena, config.input_dim * sizeof(float));
    if (!image) {
        fprintf(stderr, "Error: arena allocation failed for image\n");
        free_model_resources(&config, &weights, &state, fd, data, file_size);
        return 1;
    }

    if (load_image(image_path, image, config.input_dim, config.input_mean, config.input_std) != 0) {
        free_model_resources(&config, &weights, &state, fd, data, file_size);
        return 1;
    }

    float output[10];
    forward(&config, &weights, &state, image, output);

    int pred = argmax(output, config.output_dim);
    printf("%d\n", pred);

    free_model_resources(&config, &weights, &state, fd, data, file_size);
    arena_reset(arena);
    return 0;
}
