#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lib/arena.h"
#include "lib/ymodem.h"
#include "tinylibc/include/port.h"

/* ---------- 配置项 ---------- */
#ifndef TARGET_ADDR
#define TARGET_ADDR 0x80000000UL
#endif

#ifndef TARGET_SIZE
#define TARGET_SIZE (128 * 1024)
#endif

/* ---------- 静态资源 ---------- */
static uint8_t arena_buffer[2048];

static uint32_t g_file_size     = 0;
static uint32_t g_recv_len      = 0;
static char     g_file_name[64] = {0};
static bool     g_recv_success  = false;

/* ---------- 串口 IO 适配 ---------- */
static int32_t uart_read(void *self, uint8_t *buf, size_t len) {
    (void)self;
    if (!port_global.getchar)
        return IO_ERR_GENERIC;

    for (size_t i = 0; i < len; i++) {
        int ch = port_global.getchar();
        if (ch < 0)
            return IO_ERR_GENERIC;
        buf[i] = (uint8_t)ch;
    }
    return (int32_t)len;
}

static int32_t uart_write(void *self, const uint8_t *buf, size_t len) {
    (void)self;
    if (!port_global.putchar)
        return IO_ERR_GENERIC;

    for (size_t i = 0; i < len; i++) {
        port_global.putchar((char)buf[i]);
    }
    return (int32_t)len;
}

/* ---------- YMODEM 接收回调 (严禁在此处调用 printf) ---------- */
static int32_t on_file_begin(void *ctx, const char *file_name, uint32_t file_size) {
    (void)ctx;
    // 记录文件名和大小
    strncpy(g_file_name, file_name, sizeof(g_file_name) - 1);
    g_file_size = file_size;
    g_recv_len  = 0;

    // 如果文件超过了目标内存大小，拒绝接收
    if (file_size > TARGET_SIZE) {
        return -1; // 返回非 0 拒绝接收
    }
    return 0; /* 允许接收 */
}

static int32_t on_file_data(void *ctx, const uint8_t *data, uint32_t len) {
    (void)ctx;
    // 将数据直接写入目标内存，不作任何打印
    if (g_recv_len + len <= TARGET_SIZE) {
        memcpy((void *)(TARGET_ADDR + g_recv_len), data, len);
        g_recv_len += len;
        return 0; // 成功
    }
    return -1; // 溢出报错
}

static void on_file_end(void *ctx, bool is_success) {
    (void)ctx;
    g_recv_success = is_success;
}

/* ---------- 跳转到目标固件 ---------- */
static void jump_to_target(void) {
    printf("Jumping to 0x%08lX ...\n", (unsigned long)TARGET_ADDR);
    /* 若目标平台带有指令缓存，请在此处添加 fence.i 或缓存同步指令 */
    void (*target_entry)(void) = (void (*)(void))TARGET_ADDR;
    target_entry();
}

/* ---------- main ---------- */
int main(void) {
    /* 创建 Arena 分配器 */
    Arena arena = Arena_Create(arena_buffer, sizeof(arena_buffer));
    if (!arena)
        return 1;
    memory_if mem_if = Arena_GetMemoryIf(arena);

    /* IO 接口 */
    io_if io = {
        .self     = NULL,
        .io_read  = uart_read,
        .io_write = uart_write,
        .io_ctrl  = NULL,
    };

    /* YMODEM 配置（仅接收） */
    ymodem_config cfg = {
        .user_ctx         = NULL,
        .rx_on_file_begin = on_file_begin,
        .rx_on_file_data  = on_file_data,
        .rx_on_file_end   = on_file_end,
        .tx_on_file_begin = NULL,
        .tx_on_file_data  = NULL,
        .tx_on_file_end   = NULL,
    };

    Ymodem ym = Ymodem_Create(&mem_if, &io, &cfg);
    if (!ym)
        return 1;

    /* 阻塞接收，这期间无论终端发什么，只会安静地处理协议 */
    int32_t ret = Ymodem_Receive(ym, NULL);

    /* ======== 传输结束，终端恢复正常模式，此时可以安全地打印内容 ======== */
    printf("\n\n=============== Transfer Result ===============\n");
    switch (ret) {
    case YMODEM_OK:
        printf("[Status] finished: OK\n");
        if (g_recv_success && g_file_size > 0) {
            printf("[File] Name: %s\n", g_file_name);
            printf("[File] Size: %u bytes (Expected: %u bytes)\n", g_recv_len, g_file_size);
            printf("[Write] Flashed to 0x%x\n", (unsigned long)TARGET_ADDR);

            Ymodem_Destroy(ym);
            jump_to_target();
            /* 通常不会执行到这里 */
        }
        break;
    case YMODEM_ERR_TIMEOUT:
        printf("[Status] finished: TIMEOUT\n");
        break;
    case YMODEM_ERR_ABORT:
        printf("[Status] finished: ABORT\n");
        break;
    case YMODEM_ERR_CRC:
        printf("[Status] finished: CRC ERROR\n");
        break;
    case YMODEM_ERR_REJECT:
        printf("[Status] finished: REJECTED (File too large)\n");
        break;
    default:
        printf("[Status] finished: error code %d\n", ret);
        break;
    }

    Ymodem_Destroy(ym);
    printf("Bootloader halted.\n");
    return 0;
}