#include "test_framework.h"

#define SM_M 2
#define SM_K 3
#define SM_N 4

static int32_t A[] = { 1, 2, 3, 4, 5, 6 };
static int32_t B[] = {
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3
};
static int32_t C[SM_M * SM_N];

static void simple_matrix_init(void) {
    for (int i = 0; i < SM_M * SM_N; i++)
        C[i] = 0;
}

static void simple_matrix_vector(void) {
    for (size_t i = 0; i < SM_M; i++) {
        for (size_t k = 0; k < SM_K; k++) {
            int32_t a_ik = A[i * SM_K + k];
            size_t n = SM_N;
            size_t j = 0;
            size_t vl;
            for (; n > 0; n -= vl, j += vl) {
                vl = __riscv_vsetvl_e32m1(n);
                vint32m1_t vc = __riscv_vle32_v_i32m1(&C[i * SM_N + j], vl);
                vint32m1_t vb = __riscv_vle32_v_i32m1(&B[k * SM_N + j], vl);
                vc = __riscv_vmacc_vx_i32m1(vc, a_ik, vb, vl);
                __riscv_vse32_v_i32m1(&C[i * SM_N + j], vc, vl);
            }
        }
    }
}

static void simple_matrix_print_result(void) {
    printf("Result Matrix C:\n");
    for (size_t i = 0; i < SM_M; i++) {
        for (size_t j = 0; j < SM_N; j++)
            printf("%d ", C[i * SM_N + j]);
        printf("\n");
    }
}

const test_t test_info = {
    .name        = "simple_matrix",
    .init        = simple_matrix_init,
    .run_scalar  = NULL,
    .run_vector  = simple_matrix_vector,
    .print_result = simple_matrix_print_result,
};
