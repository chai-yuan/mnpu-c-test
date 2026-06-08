#include "ymodem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* YMODEM 控制字符 */
#define YM_SOH 0x01 // 128字节数据包起始
#define YM_STX 0x02 // 1024字节数据包起始
#define YM_EOT 0x04 // 传输结束
#define YM_ACK 0x06 // 确认
#define YM_NAK 0x15 // 否认
#define YM_CAN 0x18 // 取消传输
#define YM_C 0x43   // 字符 'C'，请求 CRC 校验

/* YMODEM 数据包大小定义 */
#define YM_PACKET_SIZE 128
#define YM_PACKET_1K_SIZE 1024
#define YM_PACKET_OVERHEAD 5 // SEQ(1) + ~SEQ(1) + CRC(2) + TYPE(1 被独立读取)
#define YM_FILE_NAME_MAX 256

/* 内部多项式 CRC 校验 */
#define YM_CRC_POLY 0x1021

/* YMODEM 实例结构体 */
struct Ymodem {
    memory_if    *mem;
    io_if        *io;
    ymodem_config cfg;
    uint8_t       buf[YM_PACKET_1K_SIZE + YM_PACKET_OVERHEAD]; // 收发复用缓冲区
};

/* 计算 CRC16-CCITT */
static uint16_t CalculateCRC16(const uint8_t *data, size_t size) {
    uint16_t crc = 0;
    for (size_t i = 0; i < size; i++) {
        crc ^= (uint16_t)(data[i] << 8);
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000)
                crc = (crc << 1) ^ YM_CRC_POLY;
            else
                crc = (crc << 1);
        }
    }
    return crc;
}

/* 阻塞读取指定长度的数据，支持超时检查 */
static int32_t ReadData(Ymodem self, uint8_t *buf, size_t len, ymodem_is_timeout_t is_timeout) {
    size_t rx_len = 0;
    while (rx_len < len) {
        if (is_timeout && is_timeout())
            return YMODEM_ERR_TIMEOUT;
        int32_t ret = IOIf_Read(self->io, buf + rx_len, len - rx_len);
        if (ret < 0)
            return YMODEM_ERR_GENERIC;
        rx_len += ret;
    }
    return YMODEM_OK;
}

/* 阻塞发送单字节命令 */
static void SendCmd(Ymodem self, uint8_t cmd) { IOIf_Write(self->io, &cmd, 1); }

/* 创建 YMODEM 实例 */
Ymodem Ymodem_Create(memory_if *mem, io_if *io, const ymodem_config *config) {
    if (!mem || !io || !config)
        return NULL;
    Ymodem self = MemoryIf_Alloc(mem, sizeof(struct Ymodem));
    if (!self)
        return NULL;

    self->mem = mem;
    self->io  = io;
    self->cfg = *config;
    return self;
}

/* 销毁 YMODEM 实例 */
void Ymodem_Destroy(Ymodem self) {
    if (self)
        MemoryIf_Free(self->mem, self);
}

/* 中断传输 */
void Ymodem_Abort(Ymodem self) {
    if (!self)
        return;
    uint8_t can_buf[2] = {YM_CAN, YM_CAN};
    IOIf_Write(self->io, can_buf, 2);
}

/* YMODEM 接收主流程 */
int32_t Ymodem_Receive(Ymodem self, ymodem_is_timeout_t is_timeout) {
    if (!self || !self->cfg.rx_on_file_begin || !self->cfg.rx_on_file_data)
        return YMODEM_ERR_PARAM;

    uint8_t  type;
    uint8_t  seq_expected   = 0;
    uint32_t file_size      = 0;
    uint32_t bytes_received = 0;
    bool     eot_received   = false;

    // 发送 'C' 启动传输
    SendCmd(self, YM_C);

    while (1) {
        if (ReadData(self, &type, 1, is_timeout) != YMODEM_OK)
            return YMODEM_ERR_TIMEOUT;

        if (type == YM_SOH || type == YM_STX) {
            uint16_t packet_size = (type == YM_SOH) ? YM_PACKET_SIZE : YM_PACKET_1K_SIZE;

            // 读取 Seq(1) + ~Seq(1) + Data(size) + CRC(2)
            if (ReadData(self, self->buf, packet_size + 4, is_timeout) != YMODEM_OK)
                return YMODEM_ERR_TIMEOUT;

            uint8_t  seq      = self->buf[0];
            uint8_t  seq_comp = self->buf[1];
            uint8_t *data     = &self->buf[2];
            uint16_t crc_rx   = (self->buf[packet_size + 2] << 8) | self->buf[packet_size + 3];

            // 检查序号与 CRC
            if (seq != (uint8_t)~seq_comp || CalculateCRC16(data, packet_size) != crc_rx) {
                SendCmd(self, YM_NAK);
                continue;
            }

            if (seq == 0) {
                // Block 0: 文件信息包
                if (data[0] == '\0') {
                    // 空文件名表示全部传输结束
                    SendCmd(self, YM_ACK);
                    return YMODEM_OK;
                }

                // 解析文件名与大小 (格式: "filename.txt\0 1234\0")
                char *file_name = (char *)data;
                char *size_str  = (char *)(data + strlen(file_name) + 1);
                file_size       = strtoul(size_str, NULL, 10);

                if (self->cfg.rx_on_file_begin(self->cfg.user_ctx, file_name, file_size) != 0) {
                    Ymodem_Abort(self);
                    return YMODEM_ERR_REJECT;
                }

                SendCmd(self, YM_ACK);
                SendCmd(self, YM_C); // 准备接收文件数据
                seq_expected   = 1;
                bytes_received = 0;
                eot_received   = false;
            } else if (seq == seq_expected) {
                // 数据包
                uint32_t chunk_len = packet_size;
                // 去除最后一包可能填充的 0x1A (EOF) 冗余数据
                if (file_size > 0 && (bytes_received + packet_size) > file_size) {
                    chunk_len = file_size - bytes_received;
                }

                if (self->cfg.rx_on_file_data(self->cfg.user_ctx, data, chunk_len) != 0) {
                    Ymodem_Abort(self);
                    return YMODEM_ERR_WRITE; // 假设宏存在，或用 YMODEM_ERR_GENERIC
                }

                bytes_received += chunk_len;
                seq_expected++;
                SendCmd(self, YM_ACK);
            } else {
                // 重复包或失序包，直接回复 ACK 让发送方继续
                SendCmd(self, YM_ACK);
            }
        } else if (type == YM_EOT) {
            // YMODEM 规范: 第一个 EOT 回复 NAK，第二个 EOT 回复 ACK 并发 'C' 请求结束包
            if (!eot_received) {
                eot_received = true;
                SendCmd(self, YM_NAK);
            } else {
                SendCmd(self, YM_ACK);
                SendCmd(self, YM_C);
                if (self->cfg.rx_on_file_end)
                    self->cfg.rx_on_file_end(self->cfg.user_ctx, true);
            }
        } else if (type == YM_CAN) {
            if (self->cfg.rx_on_file_end)
                self->cfg.rx_on_file_end(self->cfg.user_ctx, false);
            return YMODEM_ERR_ABORT;
        } else {
            // 杂乱数据过滤
        }
    }
}

/* 内部数据包封装与发送辅助函数 */
static int32_t TransmitPacket(Ymodem self, uint8_t seq, const uint8_t *data, uint16_t size,
                              ymodem_is_timeout_t is_timeout) {
    uint8_t ack;
    self->buf[0] = (size == YM_PACKET_SIZE) ? YM_SOH : YM_STX;
    self->buf[1] = seq;
    self->buf[2] = ~seq;
    memcpy(&self->buf[3], data, size);

    uint16_t crc        = CalculateCRC16(data, size);
    self->buf[size + 3] = (crc >> 8) & 0xFF;
    self->buf[size + 4] = crc & 0xFF;

    do {
        IOIf_Write(self->io, self->buf, size + 5);
        if (ReadData(self, &ack, 1, is_timeout) != YMODEM_OK)
            return YMODEM_ERR_TIMEOUT;
    } while (ack != YM_ACK && ack != YM_CAN);

    return (ack == YM_ACK) ? YMODEM_OK : YMODEM_ERR_ABORT;
}

/* YMODEM 发送主流程 */
int32_t Ymodem_Transmit(Ymodem self, ymodem_is_timeout_t is_timeout) {
    if (!self || !self->cfg.tx_on_file_begin || !self->cfg.tx_on_file_data)
        return YMODEM_ERR_PARAM;

    uint8_t  c;
    char     file_name[YM_FILE_NAME_MAX];
    uint32_t file_size;

    while (1) {
        // 等待接收方发来的 'C'
        do {
            if (ReadData(self, &c, 1, is_timeout) != YMODEM_OK)
                return YMODEM_ERR_TIMEOUT;
        } while (c != YM_C);

        // 获取要发送的文件信息
        if (self->cfg.tx_on_file_begin(self->cfg.user_ctx, file_name, &file_size) != 0) {
            // 发送空包表示发送队列结束
            memset(self->buf, 0, YM_PACKET_SIZE);
            TransmitPacket(self, 0, self->buf, YM_PACKET_SIZE, is_timeout);
            return YMODEM_OK;
        }

        // 格式化 Block 0 数据包 (文件名 + 大小)
        uint8_t block0[YM_PACKET_SIZE] = {0};
        snprintf((char *)block0, YM_PACKET_SIZE, "%s%c%lu", file_name, 0, (unsigned long)file_size);
        if (TransmitPacket(self, 0, block0, YM_PACKET_SIZE, is_timeout) != YMODEM_OK)
            return YMODEM_ERR_ABORT;

        // 等待数据接收确认 'C'
        do {
            if (ReadData(self, &c, 1, is_timeout) != YMODEM_OK)
                return YMODEM_ERR_TIMEOUT;
        } while (c != YM_C);

        // 开始分块读取与发送
        uint8_t  seq        = 1;
        uint32_t bytes_sent = 0;
        uint8_t  data_buf[YM_PACKET_1K_SIZE];

        while (bytes_sent < file_size) {
            uint16_t packet_size = ((file_size - bytes_sent) >= YM_PACKET_1K_SIZE) ? YM_PACKET_1K_SIZE : YM_PACKET_SIZE;
            memset(data_buf, 0x1A, packet_size); // 尾部默认填充 EOF (0x1A)

            int32_t read_len = self->cfg.tx_on_file_data(self->cfg.user_ctx, data_buf, packet_size);
            if (read_len < 0) {
                Ymodem_Abort(self);
                if (self->cfg.tx_on_file_end)
                    self->cfg.tx_on_file_end(self->cfg.user_ctx, false);
                return YMODEM_ERR_GENERIC;
            }

            if (TransmitPacket(self, seq++, data_buf, packet_size, is_timeout) != YMODEM_OK)
                return YMODEM_ERR_ABORT;
            bytes_sent += read_len;
        }

        // 发送完毕，处理两段 EOT 握手
        SendCmd(self, YM_EOT);
        ReadData(self, &c, 1, is_timeout); // 接收第一段 NAK
        SendCmd(self, YM_EOT);
        ReadData(self, &c, 1, is_timeout); // 接收第二段 ACK

        if (self->cfg.tx_on_file_end)
            self->cfg.tx_on_file_end(self->cfg.user_ctx, true);
        // 循环进入下一个文件的发送（YMODEM 支持多文件）
    }
}