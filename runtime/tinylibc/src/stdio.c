#include "stdio.h"
#include "port.h"
#include <stdarg.h>
#include <stddef.h>

/* ---------- 输出上下文：支持串口输出或字符串缓冲 ---------- */
struct out_ctx {
    char  *buf;   /* 非 NULL 时写入缓冲区 */
    size_t len;   /* 缓冲区总大小 */
    size_t pos;   /* 当前已写入位置 */
};

static void out_putc(struct out_ctx *o, char c) {
    if (o->buf) {
        if (o->len > 0 && o->pos < o->len - 1)
            o->buf[o->pos] = c;
        o->pos++;
    } else {
        if (c == '\n')
            port_global.putchar('\r');
        port_global.putchar(c);
    }
}

static void out_str(struct out_ctx *o, const char *s) {
    while (*s)
        out_putc(o, *s++);
}

static void out_uint(struct out_ctx *o, unsigned int n) {
    if (n / 10)
        out_uint(o, n / 10);
    out_putc(o, '0' + n % 10);
}

static void out_ulong(struct out_ctx *o, unsigned long n) {
    if (n / 10)
        out_ulong(o, n / 10);
    out_putc(o, '0' + n % 10);
}

static void out_hex(struct out_ctx *o, unsigned int n) {
    if (n / 16)
        out_hex(o, n / 16);
    int digit = n % 16;
    out_putc(o, digit < 10 ? '0' + digit : 'a' + digit - 10);
}

static void out_hex_upper(struct out_ctx *o, unsigned int n) {
    if (n / 16)
        out_hex_upper(o, n / 16);
    int digit = n % 16;
    out_putc(o, digit < 10 ? '0' + digit : 'A' + digit - 10);
}

static void out_hex_ulong(struct out_ctx *o, unsigned long n) {
    if (n / 16)
        out_hex_ulong(o, n / 16);
    int digit = n % 16;
    out_putc(o, digit < 10 ? '0' + digit : 'A' + digit - 10);
}

static void out_int(struct out_ctx *o, int n) {
    if (n < 0) {
        out_putc(o, '-');
        out_uint(o, (unsigned int)(-n));
    } else {
        out_uint(o, (unsigned int)n);
    }
}

static void vformat(struct out_ctx *o, const char *fmt, va_list args) {
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
                out_str(o, va_arg(args, const char *));
                break;
            case 'd':
                if (is_long)
                    out_int(o, (int)va_arg(args, long));
                else
                    out_int(o, va_arg(args, int));
                break;
            case 'u':
                if (is_long)
                    out_ulong(o, va_arg(args, unsigned long));
                else
                    out_uint(o, va_arg(args, unsigned int));
                break;
            case 'x':
                if (is_long)
                    out_hex(o, (unsigned int)va_arg(args, unsigned long));
                else
                    out_hex(o, va_arg(args, unsigned int));
                break;
            case 'X':
                if (is_long)
                    out_hex_ulong(o, va_arg(args, unsigned long));
                else
                    out_hex_upper(o, va_arg(args, unsigned int));
                break;
            case 'p':
                out_str(o, "0x");
                out_hex(o, (unsigned int)va_arg(args, void *));
                break;
            case 'c':
                out_putc(o, (char)va_arg(args, int));
                break;
            case '%':
                out_putc(o, '%');
                break;
            case '\0':
                return;
            default:
                out_putc(o, '%');
                if (is_long)
                    out_putc(o, 'l');
                out_putc(o, *fmt);
                break;
            }
        } else {
            out_putc(o, *fmt);
        }
        fmt++;
    }
}

int printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    struct out_ctx o = {0};
    vformat(&o, fmt, args);
    va_end(args);
    return (int)o.pos;
}

int snprintf(char *buf, size_t len, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    struct out_ctx o = { buf, len, 0 };
    vformat(&o, fmt, args);
    if (len > 0)
        buf[o.pos < len ? o.pos : len - 1] = '\0';
    va_end(args);
    return (int)o.pos;
}

int fprintf(void *stream, const char *fmt, ...) {
    (void)stream;
    va_list args;
    va_start(args, fmt);
    struct out_ctx o = {0};
    vformat(&o, fmt, args);
    va_end(args);
    return (int)o.pos;
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
        if (port_global.putchar) {
            if (c == '\n')
                port_global.putchar('\r');
            port_global.putchar((char)c);
        }
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
