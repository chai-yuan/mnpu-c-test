#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lib/arena.h"
#include "lib/ymodem.h"
#include "tinylibc/include/port.h"

/* ---------- 静态资源 ---------- */
static uint8_t arena_buffer[4096];

/* 全局缓冲区：用来暂存收到的文件数据（为了测试，暂定最大缓存 2KB） */
static uint8_t g_file_buffer[2048];
static uint32_t g_file_size = 0;
static uint32_t g_recv_len = 0;
static char g_file_name[64] = {0};
static bool g_recv_success = false;

/* ---------- 串口 IO 适配 ---------- */
static int32_t uart_read(void *self, uint8_t *buf, size_t len) {
    (void)self;
    if (!port_global.getchar) return IO_ERR_GENERIC;

    for (size_t i = 0; i < len; i++) {
        int ch = port_global.getchar();
        if (ch < 0) return IO_ERR_GENERIC;
        buf[i] = (uint8_t)ch;
    }
    return (int32_t)len;
}

static int32_t uart_write(void *self, const uint8_t *buf, size_t len) {
    (void)self;
    if (!port_global.putchar) return IO_ERR_GENERIC;

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
    g_recv_len = 0;
    
    // 如果文件超过了我们的缓冲区大小，拒绝接收
    if (file_size > sizeof(g_file_buffer)) {
        return -1; // 返回非 0 拒绝接收
    }
    return 0; /* 允许接收 */
}

static int32_t on_file_data(void *ctx, const uint8_t *data, uint32_t len) {
    (void)ctx;
    // 将数据存入缓冲区，不作任何打印
    if (g_recv_len + len <= sizeof(g_file_buffer)) {
        memcpy(&g_file_buffer[g_recv_len], data, len);
        g_recv_len += len;
        return 0; // 成功
    }
    return -1; // 溢出报错
}

static void on_file_end(void *ctx, bool is_success) {
    (void)ctx;
    g_recv_success = is_success;
}

/* ---------- main ---------- */
int main(void) {
    /* 创建 Arena 分配器 */
    Arena arena = Arena_Create(arena_buffer, sizeof(arena_buffer));
    if (!arena) return 1;
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
    if (!ym) return 1;

    printf("Waiting for YMODEM transfer ...\n");

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
            printf("\n-------- content --------\n");
            for (uint32_t i = 0; i < g_file_size; i++) {
                printf("%c",g_file_buffer[i]); // 打印收到的文件内容
            }
            printf("\n-------------------------\n");
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
        printf("[Status] finished: REJECTED (File might be too large)\n");
        break;
    default:
        printf("[Status] finished: error code %d\n", ret);
        break;
    }

    Ymodem_Destroy(ym);
    return 0;
}