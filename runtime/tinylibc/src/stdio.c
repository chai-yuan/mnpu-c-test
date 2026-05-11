#include "stdio.h"
#include "port.h"
#include <stdarg.h>

static void putchar_impl(char c) {
    if (c == '\n')
        port_global.putchar('\r');
    port_global.putchar(c);
}

static void print_str(const char *s) {
    while (*s)
        putchar_impl(*s++);
}

static void print_uint(unsigned int n) {
    if (n / 10)
        print_uint(n / 10);
    putchar_impl('0' + n % 10);
}

static void print_ulong(unsigned long n) {
    if (n / 10)
        print_ulong(n / 10);
    putchar_impl('0' + n % 10);
}

static void print_hex(unsigned int n) {
    if (n / 16)
        print_hex(n / 16);
    int digit = n % 16;
    putchar_impl(digit < 10 ? '0' + digit : 'a' + digit - 10);
}

static void print_hex_upper(unsigned int n) {
    if (n / 16)
        print_hex_upper(n / 16);
    int digit = n % 16;
    putchar_impl(digit < 10 ? '0' + digit : 'A' + digit - 10);
}

static void print_hex_ulong(unsigned long n) {
    if (n / 16)
        print_hex_ulong(n / 16);
    int digit = n % 16;
    putchar_impl(digit < 10 ? '0' + digit : 'A' + digit - 10);
}

static void print_int(int n) {
    if (n < 0) {
        putchar_impl('-');
        print_uint((unsigned int)(-n));
    } else {
        print_uint((unsigned int)n);
    }
}

static void vprintf_impl(const char *fmt, va_list args) {
    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            int is_long = 0;
            if (*fmt == 'l') {
                is_long = 1;
                fmt++;
            }
            switch (*fmt) {
            case 's':
                print_str(va_arg(args, const char *));
                break;
            case 'd':
                if (is_long)
                    print_int((int)va_arg(args, long));
                else
                    print_int(va_arg(args, int));
                break;
            case 'u':
                if (is_long)
                    print_ulong(va_arg(args, unsigned long));
                else
                    print_uint(va_arg(args, unsigned int));
                break;
            case 'x':
                if (is_long)
                    print_hex((unsigned int)va_arg(args, unsigned long));
                else
                    print_hex(va_arg(args, unsigned int));
                break;
            case 'X':
                if (is_long)
                    print_hex_ulong(va_arg(args, unsigned long));
                else
                    print_hex_upper(va_arg(args, unsigned int));
                break;
            case 'p':
                print_str("0x");
                print_hex((unsigned int)va_arg(args, void *));
                break;
            case 'c':
                putchar_impl((char)va_arg(args, int));
                break;
            case '%':
                putchar_impl('%');
                break;
            case '\0':
                return;
            default:
                putchar_impl('%');
                if (is_long)
                    putchar_impl('l');
                putchar_impl(*fmt);
                break;
            }
        } else {
            putchar_impl(*fmt);
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

int getchar(void) {
    if (port_global.getchar)
        return port_global.getchar();
    return -1;
}

char *fgets(char *s, int size, void *stream) {
    (void)stream;
    int i = 0;
    int c;
    if (size <= 0)
        return 0;
    while (i < size - 1) {
        c = getchar();
        if (c < 0)
            break;
        if (c == '\r')
            c = '\n';
        s[i++] = (char)c;
        /* echo back to match native terminal behavior */
        if (c == '\n')
            putchar_impl('\r');
        putchar_impl((char)c);
        if (c == '\n')
            break;
    }
    if (i == 0 && c < 0)
        return 0;
    s[i] = '\0';
    return s;
}

int fflush(void *stream) {
    (void)stream;
    return 0;
}
