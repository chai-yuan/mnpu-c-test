#include "port.h"
#include <stdint.h>

/* ------------------------------------------------------------------
 * UART output (TX) – works fine on both spike and qemu
 * ------------------------------------------------------------------ */
#define UART_BASE 0x10000000
#define UART_THR 0
#define UART_LSR 5
#define UART_LSR_THRE (1 << 5)

static void uart_putchar(char c) {
    volatile char *uart = (volatile char *)UART_BASE;
    while (!(uart[UART_LSR] & UART_LSR_THRE))
        ;
    uart[UART_THR] = c;
}

/* ------------------------------------------------------------------
 * spike on RV32 does not support interactive stdin.
 * Return EOF so fgets returns NULL immediately.
 * ------------------------------------------------------------------ */
static int uart_getchar(void) { return -1; }

/* ------------------------------------------------------------------ */

int main(void);

void _init(void) {
    struct port_functions port = {
        .putchar = uart_putchar,
        .getchar = uart_getchar,
    };
    port_init(port);
    main();
}

void c_trap_handler(uint32_t mcause, uint32_t mepc, uint32_t mtval) {
    printf("\n================= 致命异常发生 (FATAL TRAP) =================\n");
    printf("指令发生位置 (mepc)   : 0x%X\n", mepc);
    printf("异常附加信息 (mtval)  : 0x%X\n", mtval);
    printf("异常原因编码 (mcause) : 0x%X\n", mcause);
    
    uint32_t cause_code = mcause & 0x7FFFFFFF;
    printf("错误原因解析: ");
    
    switch (cause_code) {
        case 2: 
            printf("非法指令 (Illegal Instruction)!\n"); 
            break;
        case 4:
            printf("加载地址未对齐 (Load address misaligned)!\n");
            break;
        case 5: 
            printf("加载访问错误 (Load access fault)!\n"); 
            break;
        case 6:
            printf("存储地址未对齐 (Store/AMO address misaligned)!\n");
            break;
        case 7: 
            printf("存储访问错误 (Store/AMO access fault)!\n"); 
            break;
        case 8:
        case 9:
        case 11:
            printf("环境调用 (Environment Call) - 通常用于系统调用(Syscall)\n");
            break;
        default: 
            printf("未知异常 (代码: %u)\n", cause_code); 
            break;
    }
    printf("=============================================================\n\n");
    
    while(1) {
    }
}
