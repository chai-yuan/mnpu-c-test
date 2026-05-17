#ifndef IO_IF_H_
#define IO_IF_H_

#include <stddef.h>
#include <stdint.h>

/* 定义标准的错误码 */
#define IO_OK 0
#define IO_ERR_GENERIC -1  // 通用错误
#define IO_ERR_PARAM -2    // 参数错误
#define IO_ERR_NOT_SUPP -3 // 不支持的操作

// 读数据到buf中，返回读取的字节数
typedef int32_t (*io_read_t)(void *self, uint8_t *buf, size_t len);
// 写数据，返回写入的字节数
typedef int32_t (*io_write_t)(void *self, const uint8_t *buf, size_t len);
// IO控制/配置接口
typedef int32_t (*io_ctrl_t)(void *self, uint32_t cmd, void *arg);

typedef struct io_if {
    void      *self;
    io_read_t  io_read;
    io_write_t io_write;
    io_ctrl_t  io_ctrl;
} io_if;

static inline int32_t IOIf_Read(io_if *intf, uint8_t *buf, size_t len) {
    if (!intf || !intf->io_read || !buf)
        return IO_ERR_PARAM;
    return intf->io_read(intf->self, buf, len);
}

static inline int32_t IOIf_Write(io_if *intf, const uint8_t *buf, size_t len) {
    if (!intf || !intf->io_write || !buf)
        return IO_ERR_PARAM;
    return intf->io_write(intf->self, buf, len);
}

static inline int32_t IOIf_Ctrl(io_if *intf, uint32_t cmd, void *arg) {
    if (!intf)
        return IO_ERR_PARAM;
    if (!intf->io_ctrl)
        return IO_ERR_NOT_SUPP;
    return intf->io_ctrl(intf->self, cmd, arg);
}

#endif // IO_IF_H_