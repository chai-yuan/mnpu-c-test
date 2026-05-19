#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

void widening_mac(const int16_t *A, const int16_t *B, int32_t *C, size_t n) {
    size_t vl;

    for (; n > 0; n -= vl, A += vl, B += vl, C += vl) {
        
        // 1. 设置矢量长度。以 16位(e16)、单寄存器(m1) 为基准请求
        vl = __riscv_vsetvl_e16m1(n);

        // 2. 加载两个 16位 (m1) 的操作数
        vint16m1_t va = __riscv_vle16_v_i16m1(A, vl);
        vint16m1_t vb = __riscv_vle16_v_i16m1(B, vl);

        // 3. 加载 32位 (m2) 的累加器旧值
        // 因为 C 的位宽是 A/B 的两倍，所以它必须用两倍的寄存器 (m2) 来装这 vl 个元素
        vint32m2_t vc = __riscv_vle32_v_i32m2(C, vl);

        // 4. 执行加宽乘累加：vc[i] = vc[i] + (va[i] * vb[i])
        // 输入类型是 int16m1_t，输出类型是 int32m2_t
        vc = __riscv_vwmacc_vv_i32m2(vc, va, vb, vl);

        // 5. 将更新后的 32位累加结果写回内存
        __riscv_vse32_v_i32m2(C, vc, vl);
    }
}

int main() {
    size_t N = 10;
    
    int16_t A[10] = {20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000};
    int16_t B[10] = {20000, -20000, 20000, -20000, 20000, -20000, 20000, -20000, 20000, -20000};
    
    int32_t C[10] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};

    widening_mac(A, B, C, N);

    for(int i=0; i<N; i++){
        // 期望结果: 100 + (20000 * 20000) = 400000100 
        // 期望结果: 100 + (20000 * -20000) = -399999900
        printf("C[%d] = %d\n", i, C[i]);
    }
    
    return 0;
}