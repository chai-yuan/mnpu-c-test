#include "port.h"

struct port_functions port_global;

void port_init(struct port_functions port) {
    port_global.putchar = port.putchar;
    port_global.getchar = port.getchar;
}
