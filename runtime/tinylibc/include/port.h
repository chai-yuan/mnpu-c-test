#ifndef _PORT_H
#define _PORT_H

typedef void (*putchar_t)(char);

struct port_functions {
    putchar_t putchar;
};

extern struct port_functions port_global;

void port_init(struct port_functions port);

#endif
