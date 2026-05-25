#ifndef _PORT_H
#define _PORT_H

#include <stdint.h>

typedef void (*putchar_t)(char);
typedef int (*getchar_t)(void);
typedef uint64_t (*get_time_t)(void);

struct port_functions {
    putchar_t  putchar;
    getchar_t  getchar;
    get_time_t get_time;
};

extern struct port_functions port_global;

void port_init(struct port_functions port);

#endif
