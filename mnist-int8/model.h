#ifndef MODEL_INT8_H
#define MODEL_INT8_H

#include <stdint.h>
#include "lib/arena.h"

// Binary format constants
#define INT8_MAX_LAYERS  16
#define INT8_HEADER_SIZE 512
#define INT8_MAGIC       0x4D4C4938  /* "MLI8" */

// ---------------------------------------------------------------------------
// Model configuration (from binary header)
// ---------------------------------------------------------------------------
typedef struct {
    int32_t input_dim;
    int32_t num_layers;
    int32_t layer_out_features[INT8_MAX_LAYERS];
    int32_t input_zero_point;
    int32_t output_zero_point;
    int32_t version;
} Int8Config;

// ---------------------------------------------------------------------------
// Per-layer data (pointers into the loaded weights buffer)
// ---------------------------------------------------------------------------
typedef struct {
    int32_t  in_features;
    int32_t  out_features;
    int32_t *requant_multiplier;  /* [out_features] TFLite Q0.31 multipliers    */
    int32_t *requant_shift;       /* [out_features] right-shift amounts         */
    int32_t  out_zero_point;
    int32_t *bias;                /* [out_features] int32 biases                */
    int8_t  *weight;              /* [out_features * in_features] int8 weights  */
} Int8Layer;

// ---------------------------------------------------------------------------
// Complete model
// ---------------------------------------------------------------------------
typedef struct {
    Int8Config config;
    int32_t    num_layers;
    Int8Layer  layers[INT8_MAX_LAYERS];
} Int8Model;

// ---------------------------------------------------------------------------
// Run state (scratch buffers for inference)
// ---------------------------------------------------------------------------
typedef struct {
    int32_t *acc;       /* int32 accumulator, size = max(784, 128, 10) = 784  */
    int8_t  *buf;       /* two int8 buffers back-to-back, size = 784+784      */
    int32_t  max_dim;
} Int8RunState;

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/**
 * Parse binary header.  Returns 0 on success, <0 on error.
 *   -1 : bad magic
 *   -2 : unsupported version
 *   -3 : too many layers
 */
int int8_parse_header(const unsigned char *header, Int8Config *cfg);

/**
 * Set up per-layer pointers from the binary data (after header).
 * `data` must point to the start of the file buffer.
 * Must be called after int8_parse_header.
 */
int int8_setup_layers(Int8Model *model, unsigned char *data);

/**
 * Allocate run-state scratch buffers from an Arena.
 * Returns 0 on success, -1 if arena is exhausted.
 */
int int8_runstate_init(Int8RunState *state, const Int8Model *model,
                       Arena arena);

/**
 * Run INT8 inference.
 *   input:  int8 array of length model.config.input_dim
 *   output: int8 array of length model.layers[num_layers-1].out_features
 * Returns the predicted class index.
 */
int int8_forward(const Int8Model *model, Int8RunState *state,
                 const int8_t *input, int8_t *output);

/**
 * Helper: find argmax in int8 output (skips softmax).
 */
int int8_argmax(const int8_t *x, int n);

#endif /* MODEL_INT8_H */
