#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

static inline uint32_t read_mcycle() {
    uint32_t val;
    __asm__ __volatile__ ("csrr %0, mcycle" : "=r" (val));
    return val;
}

void matrix_multiply(const int32_t *A, const int32_t *B, int32_t *C, size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            int32_t a_ik = A[i * K + k];
            
            size_t n = N;          // 还剩 n 个元素要处理
            size_t j = 0;          // 当前处理到了第 j 列
            size_t vl;             // 当前能处理的矢量长度
            
            for (; n > 0; n -= vl, j += vl) {
                vl = __riscv_vsetvl_e32m1(n);
                vint32m1_t vc = __riscv_vle32_v_i32m1(&C[i * N + j], vl);
                vint32m1_t vb = __riscv_vle32_v_i32m1(&B[k * N + j], vl);
                vc = __riscv_vmacc_vx_i32m1(vc, a_ik, vb, vl);
                __riscv_vse32_v_i32m1(&C[i * N + j], vc, vl);
            }
        }
    }
}

int main() {
    size_t M = 2, K = 3, N = 4;
    
    // A (2x3)
    int32_t A[] = {
        1, 2, 3,
        4, 5, 6
    };
    // B (3x4)
    int32_t B[] = {
        1,  1,  1,  1,
        2,  2,  2,  2,
        3,  3,  3,  3
    };
    // C
    int32_t C[2 * 4] = {0};

    matrix_multiply(A, B, C, M, K, N);

    printf("Result Matrix C:\n");
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%d ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}