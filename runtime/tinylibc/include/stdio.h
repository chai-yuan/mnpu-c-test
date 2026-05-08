#ifndef _STDIO_H
#define _STDIO_H

#define stderr ((void *)0)

int printf(const char *fmt, ...);
int fprintf(void *stream, const char *fmt, ...);

#endif
