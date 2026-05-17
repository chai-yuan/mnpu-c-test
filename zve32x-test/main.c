#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

static inline uint32_t read_mcycle() {
    uint32_t val;
    __asm__ __volatile__ ("csrr %0, mcycle" : "=r" (val));
    return val;
}

void vector_add_scalar(const int32_t *a, const int32_t *b, int32_t *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_rvv(const int32_t *a, const int32_t *b, int32_t *c, size_t n) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m1(n);
        vint32m1_t va = __riscv_vle32_v_i32m1(a, vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(b, vl);
        vint32m1_t vc = __riscv_vadd_vv_i32m1(va, vb, vl);
        __riscv_vse32_v_i32m1(c, vc, vl);
    }
}

int main() {
    const size_t N = 1000; 
    int32_t a[1000], b[1000], c_scalar[1000], c_rvv[1000];

    for (size_t i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 1000 - i;
    }

    uint32_t start_inst, end_inst;
    uint32_t scalar_cost, rvv_cost;

    start_inst = read_mcycle();
    vector_add_scalar(a, b, c_scalar, N);
    end_inst = read_mcycle();
    scalar_cost = end_inst - start_inst;
    printf("标量版本执行周期数: %u\n", scalar_cost);

    start_inst = read_mcycle();
    vector_add_rvv(a, b, c_rvv, N);
    end_inst = read_mcycle();
    rvv_cost = end_inst - start_inst;
    printf("矢量版本执行周期数: %u\n", rvv_cost);
    return 0;
}