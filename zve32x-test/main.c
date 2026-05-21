#include "test_framework.h"

int main(void) {
    printf("=== Test: %s ===\n", test_info.name);

    if (test_info.init)
        test_info.init();

    uint32_t start, end;

    if (test_info.run_scalar) {
        start = read_mcycle();
        test_info.run_scalar();
        end   = read_mcycle();
        printf("标量版本执行周期数: %u\n", end - start);
    }

    if (test_info.run_vector) {
        start = read_mcycle();
        test_info.run_vector();
        end   = read_mcycle();
        printf("矢量版本执行周期数: %u\n", end - start);
    }

    if (test_info.print_result)
        test_info.print_result();

    return 0;
}
