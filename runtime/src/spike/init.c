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
