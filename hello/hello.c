#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h> // 必须包含此头文件，里面定义了所有 RVV 内联函数和类型

// 矢量加法函数：计算 c[i] = a[i] + b[i]
void vector_add(const int32_t *a, const int32_t *b, int32_t *c, size_t n) {
    size_t vl; // Vector Length: 保存每次循环硬件实际能处理的元素个数

    // 当还有剩余元素需要处理时继续循环
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        
        // 1. 设置矢量长度 (vsetvl)
        // 请求处理 n 个元素，元素大小为 32位(e32)，寄存器分组为1(m1)
        // 硬件会返回它当前这一轮实际能处理的最大个数给 vl
        vl = __riscv_vsetvl_e32m1(n);

        // 2. 从内存加载数据到矢量寄存器
        // 从指针 a 和 b 处，加载 vl 个 32位 整数到矢量寄存器 va 和 vb 中
        vint32m1_t va = __riscv_vle32_v_i32m1(a, vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(b, vl);

        // 3. 执行矢量加法
        // 执行 va + vb，处理 vl 个元素
        vint32m1_t vc = __riscv_vadd_vv_i32m1(va, vb, vl);

        // 4. 将结果存回内存
        // 将矢量寄存器 vc 中的 vl 个元素写回到指针 c 指向的内存
        __riscv_vse32_v_i32m1(c, vc, vl);
    }
}

int main() {
    const size_t N = 100; // 假设我们有 100 个元素
    int32_t a[100], b[100], c[100];

    // 初始化测试数据
    for (size_t i = 0; i < N; i++) {
        a[i] = i;        // 0, 1, 2...
        b[i] = 100 - i;  // 100, 99, 98...
    }

    // 调用矢量加法函数
    vector_add(a, b, c, N);

    // 打印前 5 个和最后 5 个结果验证
    printf("验证结果 (期望每个结果都是 100):\n");
    for (size_t i = 0; i < 5; i++) {
        printf("c[%zu] = %d\n", i, c[i]);
    }
    printf("...\n");
    for (size_t i = N - 5; i < N; i++) {
        printf("c[%zu] = %d\n", i, c[i]);
    }

    return 0;
}