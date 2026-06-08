#ifndef YMODEM_H_
#define YMODEM_H_

#include "interface/io_if.h"
#include "interface/memory_if.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define YMODEM_OK 0
#define YMODEM_ERR_GENERIC -1 // 通用错误
#define YMODEM_ERR_TIMEOUT -2 // 传输或响应超时
#define YMODEM_ERR_ABORT -3   // 被对端终止 (收到连续的 CAN)
#define YMODEM_ERR_CRC -4     // 校验错误超限
#define YMODEM_ERR_REJECT -5  // 用户回调拒绝了当前传输
#define YMODEM_ERR_PARAM -6   // 参数错误
#define YMODEM_ERR_WRITE -7   // 写入/保存错误

typedef struct ymodem_config {
    void *user_ctx;
    /* 接收模式回调 */
    // 接收到文件头，提供文件名和大小。返回0允许接收，非0拒绝
    int32_t (*rx_on_file_begin)(void *ctx, const char *file_name, uint32_t file_size);
    // 接收到文件数据块，要求用户写入存储。返回0表示成功
    int32_t (*rx_on_file_data)(void *ctx, const uint8_t *data, uint32_t len);
    // 单个文件接收完成
    void (*rx_on_file_end)(void *ctx, bool is_success);
    /* 发送模式回调 */
    // 准备发送文件，要求填充文件名和大小。返回0表示有文件要发，非0表示结束发送队列
    int32_t (*tx_on_file_begin)(void *ctx, char *file_name_out, uint32_t *file_size_out);
    // 请求读取文件数据块以供发送。返回实际读取到的字节数 (<0表示读取错误)
    int32_t (*tx_on_file_data)(void *ctx, uint8_t *buf, uint32_t len);
    // 单个文件发送完成
    void (*tx_on_file_end)(void *ctx, bool is_success);
} ymodem_config;

// 确认是否超时的回调函数类型，返回 true 表示已超时
typedef bool (*ymodem_is_timeout_t)(void);

// 声明不透明结构体
typedef struct Ymodem *Ymodem;

// 依赖 memory_if 分配内存并构造 YMODEM 实例
Ymodem Ymodem_Create(memory_if *mem, io_if *io, const ymodem_config *config);
// 销毁 YMODEM 实例，释放相关内存
void Ymodem_Destroy(Ymodem self);

// 启动并阻塞执行 YMODEM 接收流程，直至结束或出错。is_timeout 传入 NULL 则永远阻塞
int32_t Ymodem_Receive(Ymodem self, ymodem_is_timeout_t is_timeout);
// 启动并阻塞执行 YMODEM 发送流程，直至结束或出错。is_timeout 传入 NULL 则永远阻塞
int32_t Ymodem_Transmit(Ymodem self, ymodem_is_timeout_t is_timeout);
// 主动发送终止信号 (CAN) 中断当前的传输
void Ymodem_Abort(Ymodem self);

#endif // YMODEM_H_