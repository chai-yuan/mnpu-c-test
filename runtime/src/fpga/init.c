#include "port.h"

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

int main(void);

void _init(void) {
    struct port_functions port = {
        .putchar = uart_putchar,
        .getchar = uart_getchar,
    };
    port_init(port);
    main();
}
