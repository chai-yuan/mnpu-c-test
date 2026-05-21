#include "test_framework.h"

#define VWMACC_N 10

static int16_t A[10], B[10];
static int32_t C[10];

static void vwmacc_init(void) {
    for (int i = 0; i < VWMACC_N; i++) {
        A[i] = 20000;
        B[i] = (i % 2 == 0) ? 20000 : -20000;
        C[i] = 100;
    }
}

static void vwmacc_vector(void) {
    size_t n = VWMACC_N;
    size_t vl;
    const int16_t *pa = A, *pb = B;
    int32_t *pc = C;
    for (; n > 0; n -= vl, pa += vl, pb += vl, pc += vl) {
        vl = __riscv_vsetvl_e16m1(n);
        vint16m1_t va  = __riscv_vle16_v_i16m1(pa, vl);
        vint16m1_t vb  = __riscv_vle16_v_i16m1(pb, vl);
        vint32m2_t vc  = __riscv_vle32_v_i32m2(pc, vl);
        vc = __riscv_vwmacc_vv_i32m2(vc, va, vb, vl);
        __riscv_vse32_v_i32m2(pc, vc, vl);
    }
}

static void vwmacc_print_result(void) {
    for (int i = 0; i < VWMACC_N; i++)
        printf("C[%d] = %d\n", i, C[i]);
}

const test_t test_info = {
    .name        = "vwmacc",
    .init        = vwmacc_init,
    .run_scalar  = NULL,
    .run_vector  = vwmacc_vector,
    .print_result = vwmacc_print_result,
};
