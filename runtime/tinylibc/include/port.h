#ifndef _PORT_H
#define _PORT_H

typedef void (*putchar_t)(char);
typedef int (*getchar_t)(void);

struct port_functions {
    putchar_t putchar;
    getchar_t getchar;
};

extern struct port_functions port_global;

void port_init(struct port_functions port);

#endif
