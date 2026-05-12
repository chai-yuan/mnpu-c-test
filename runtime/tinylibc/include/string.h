#ifndef _STRING_H
#define _STRING_H

#include <stddef.h>

void  *memset(void *s, int c, size_t n);
void  *memcpy(void *dst, const void *src, size_t n);
size_t strlen(const char *s);
int    strcmp(const char *s1, const char *s2);

#endif
