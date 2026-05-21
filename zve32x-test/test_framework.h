#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

static inline uint32_t read_mcycle(void) {
    uint32_t val = 0;
    // __asm__ __volatile__("csrr %0, mcycle" : "=r"(val));
    return val;
}

typedef struct {
    const char *name;          /* 测试名称 */
    void (*init)(void);        /* 初始化测试数据 */
    void (*run_scalar)(void);  /* 标量版本 (可为 NULL) */
    void (*run_vector)(void);  /* 向量版本 */
    void (*print_result)(void);/* 打印 & 校验结果 */
} test_t;

extern const test_t test_info;

#endif /* TEST_FRAMEWORK_H */
