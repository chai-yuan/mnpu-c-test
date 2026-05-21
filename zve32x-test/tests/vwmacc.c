#include "test_framework.h"
#include "stdlib.h"

#define VWMACC_N 10

static int16_t A[VWMACC_N], B[VWMACC_N];
static int32_t C_scalar[VWMACC_N];
static int32_t C_vector[VWMACC_N];

static void vwmacc_init(void) {
    for (int i = 0; i < VWMACC_N; i++) {
        A[i] = (int16_t)((rand() % 32768) - 16384);
        B[i] = (int16_t)((rand() % 32768) - 16384);
        
        int32_t c_init = (int32_t)((rand() % 100000) - 50000);
        C_scalar[i] = c_init;
        C_vector[i] = c_init;
    }
}

static void vwmacc_scalar(void) {
    for (int i = 0; i < VWMACC_N; i++) {
        // 模拟 vwmacc 宽化乘加：必须先将 int16_t 强转为 int32_t 后再相乘
        C_scalar[i] = C_scalar[i] + (int32_t)A[i] * (int32_t)B[i];
    }
}

static void vwmacc_vector(void) {
    size_t n = VWMACC_N;
    size_t vl;
    const int16_t *pa = A, *pb = B;
    int32_t *pc = C_vector; // 指向专用于向量计算的结果数组
    
    for (; n > 0; n -= vl, pa += vl, pb += vl, pc += vl) {
        vl = __riscv_vsetvl_e16m1(n);
        vint16m1_t va  = __riscv_vle16_v_i16m1(pa, vl);
        vint16m1_t vb  = __riscv_vle16_v_i16m1(pb, vl);
        vint32m2_t vc  = __riscv_vle32_v_i32m2(pc, vl);
        
        // 向量宽化乘加指令：vc[i] = vc[i] + va[i] * vb[i]
        vc = __riscv_vwmacc_vv_i32m2(vc, va, vb, vl);
        __riscv_vse32_v_i32m2(pc, vc, vl);
    }
}

static void vwmacc_print_result(void) {
    for (int i = 0; i < VWMACC_N; i++) {
        printf("Index [%d] | Scalar: %d | Vector: %d %s\n", 
               i, C_scalar[i], C_vector[i], 
               (C_scalar[i] == C_vector[i]) ? "" : "<--- MISMATCH!");
    }
}

const test_t test_info = {
    .name         = "vwmacc",
    .init         = vwmacc_init,
    .run_scalar   = vwmacc_scalar,
    .run_vector   = vwmacc_vector,
    .print_result = vwmacc_print_result,
};