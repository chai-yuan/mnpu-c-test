#include "port.h"

static void uart_putchar(char c) {
}

static int uart_getchar(void) {
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
