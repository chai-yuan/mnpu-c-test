#include "test_framework.h"
#include <stdlib.h>

#define SM_M 5
#define SM_K 6
#define SM_N 4

static int32_t A[SM_M * SM_K];
static int32_t B[SM_K * SM_N];
static int32_t C_scalar[SM_M * SM_N];
static int32_t C_vector[SM_M * SM_N];

static void simple_matrix_init(void) {
    for (int i = 0; i < SM_M * SM_K; i++) {
        A[i] = (int32_t)(rand() % 200 - 100); // 范围: -100 到 99
    }
    
    for (int i = 0; i < SM_K * SM_N; i++) {
        B[i] = (int32_t)(rand() % 200 - 100);
    }
    
    for (int i = 0; i < SM_M * SM_N; i++) {
        C_scalar[i] = 0;
        C_vector[i] = 0;
    }
}

static void simple_matrix_scalar(void) {
    for (size_t i = 0; i < SM_M; i++) {
        for (size_t k = 0; k < SM_K; k++) {
            int32_t a_ik = A[i * SM_K + k];
            for (size_t j = 0; j < SM_N; j++) {
                C_scalar[i * SM_N + j] += a_ik * B[k * SM_N + j];
            }
        }
    }
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
                vint32m1_t vc = __riscv_vle32_v_i32m1(&C_vector[i * SM_N + j], vl);
                vint32m1_t vb = __riscv_vle32_v_i32m1(&B[k * SM_N + j], vl);
                
                vc = __riscv_vmacc_vx_i32m1(vc, a_ik, vb, vl);
                __riscv_vse32_v_i32m1(&C_vector[i * SM_N + j], vc, vl);
            }
        }
    }
}

static void simple_matrix_print_result(void) {
    printf("Result Matrix C (%d x %d):\n", SM_M, SM_N);
    for (size_t i = 0; i < SM_M; i++) {
        for (size_t j = 0; j < SM_N; j++) {
            size_t idx = i * SM_N + j;
            printf("C[%u][%u] | Scalar: %d | Vector: %d %s\n", 
                   i, j, C_scalar[idx], C_vector[idx], 
                   (C_scalar[idx] == C_vector[idx]) ? "" : "<--- MISMATCH!");
        }
    }
}

const test_t test_info = {
    .name         = "simple_matrix",
    .init         = simple_matrix_init,
    .run_scalar   = simple_matrix_scalar,
    .run_vector   = simple_matrix_vector,
    .print_result = simple_matrix_print_result,
};