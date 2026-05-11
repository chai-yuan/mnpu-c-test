#include "port.h"

#define UART_BASE 0x10000000
#define UART_THR 0
#define UART_LSR 5
#define UART_LSR_THRE (1 << 5)
#define UART_LSR_DR   (1 << 0)

static void uart_putchar(char c) {
    volatile char *uart = (volatile char *)UART_BASE;
    while (!(uart[UART_LSR] & UART_LSR_THRE))
        ;
    uart[UART_THR] = c;
}

static int uart_getchar(void) {
    volatile char *uart = (volatile char *)UART_BASE;
    while (!(uart[UART_LSR] & UART_LSR_DR))
        ;
    return uart[UART_THR];
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
