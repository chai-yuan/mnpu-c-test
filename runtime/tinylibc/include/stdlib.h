#ifndef _STDLIB_H
#define _STDLIB_H

#include <stddef.h>

unsigned long strtoul(const char *nptr, char **endptr, int base);

void srand(unsigned int seed);
int  rand(void);

#endif
