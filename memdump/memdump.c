#include <stdio.h>
#include <stdint.h>

#ifndef MEM_ADDR
#define MEM_ADDR  0xa0000000
#endif

#ifndef MEM_LEN
#define MEM_LEN   256
#endif

static void hexdump(const void *addr, uint32_t len) {
    const uint8_t *p = (const uint8_t *)addr;
    uint32_t i, j;

    for (i = 0; i < len; i += 16) {
        /* 地址列 */
        printf("%x  ", i);

        /* 十六进制列 */
        for (j = 0; j < 16; j++) {
            if (i + j < len)
                printf("%x ", p[i + j]);
            else
                printf("   ");
        }

        printf("|\n");
    }
}

int main() {
    printf("Memory Dump Tool\n");
    printf("Address: 0x%x\n", MEM_ADDR);
    printf("Length:  %u bytes\n\n", MEM_LEN);

    hexdump((const void *)MEM_ADDR, MEM_LEN);

    return 0;
}
