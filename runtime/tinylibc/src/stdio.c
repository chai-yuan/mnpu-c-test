#include "stdio.h"
#include "port.h"
#include <stdarg.h>

static void putchar(char c) {
    if (c == '\n')
        port_global.putchar('\r');
    port_global.putchar(c);
}

static void print_str(const char *s) {
    while (*s)
        putchar(*s++);
}

static void print_uint(unsigned int n) {
    if (n / 10)
        print_uint(n / 10);
    putchar('0' + n % 10);
}

static void print_int(int n) {
    if (n < 0) {
        putchar('-');
        print_uint((unsigned int)(-n));
    } else {
        print_uint((unsigned int)n);
    }
}

static void vprintf_impl(const char *fmt, va_list args) {
    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            switch (*fmt) {
            case 's':
                print_str(va_arg(args, const char *));
                break;
            case 'd':
                print_int(va_arg(args, int));
                break;
            case 'c':
                putchar((char)va_arg(args, int));
                break;
            case '%':
                putchar('%');
                break;
            case '\0':
                return;
            default:
                putchar('%');
                putchar(*fmt);
                break;
            }
        } else {
            putchar(*fmt);
        }
        fmt++;
    }
}

int printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf_impl(fmt, args);
    va_end(args);
    return 0;
}

int fprintf(void *stream, const char *fmt, ...) {
    (void)stream;
    va_list args;
    va_start(args, fmt);
    vprintf_impl(fmt, args);
    va_end(args);
    return 0;
}
