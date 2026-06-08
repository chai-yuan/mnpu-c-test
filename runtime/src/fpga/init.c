#include "port.h"
#include "stdint.h"
#include "stdio.h"

#define UART_BASE 0x10000000UL
#define UART_RX_FIFO (*(volatile unsigned int *)(UART_BASE + 0x0))
#define UART_TX_FIFO (*(volatile unsigned int *)(UART_BASE + 0x4))
#define UART_STAT_REG (*(volatile unsigned int *)(UART_BASE + 0x8))

#define STAT_RX_VALID (1 << 0)
#define STAT_TX_FULL (1 << 3)

static void uart_putchar(char c) {
    while (UART_STAT_REG & STAT_TX_FULL)
        ;
    UART_TX_FIFO = (unsigned int)c;
}

static int uart_getchar(void) {
    while (!(UART_STAT_REG & STAT_RX_VALID))
        ;
    unsigned int data = UART_RX_FIFO;
    return (int)(data & 0xFF);
}

#define TIMER_BASE 0x20000000UL
#define TIMER_READ (*(volatile unsigned int *)(TIMER_BASE + 0x0))

static uint64_t read_time(void) {
    // uint32_t val = TIMER_READ;
    // return (uint64_t)val;
    return 0;
}

int main(void);

void _init(void) {
    struct port_functions port = {
        .putchar  = uart_putchar,
        .getchar  = uart_getchar,
        .get_time = read_time,
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
}
