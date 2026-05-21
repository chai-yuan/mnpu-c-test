#include "test_framework.h"

#define N 32

static int32_t a[1000], b[1000], c_scalar[1000], c_rvv[1000];

static void add_init(void) {
    for (size_t i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 1000 - i;
    }
}

static void add_scalar(void) {
    for (size_t i = 0; i < N; i++)
        c_scalar[i] = a[i] + b[i];
}

static void add_vector(void) {
    size_t vl;
    size_t n = N;
    const int32_t *pa = a, *pb = b;
    int32_t *pc = c_rvv;
    for (; n > 0; n -= vl, pa += vl, pb += vl, pc += vl) {
        vl = __riscv_vsetvl_e32m1(n);
        vint32m1_t va = __riscv_vle32_v_i32m1(pa, vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(pb, vl);
        vint32m1_t vc = __riscv_vadd_vv_i32m1(va, vb, vl);
        __riscv_vse32_v_i32m1(pc, vc, vl);
    }
}

static void add_print_result(void) {
    printf("标量结果: ");
    for (int i = 0; i < N; i++) printf("%d ", c_scalar[i]);
    printf("\n矢量结果: ");
    for (int i = 0; i < N; i++) printf("%d ", c_rvv[i]);
    printf("\n");
}

const test_t test_info = {
    .name        = "vadd",
    .init        = add_init,
    .run_scalar  = add_scalar,
    .run_vector  = add_vector,
    .print_result = add_print_result,
};
