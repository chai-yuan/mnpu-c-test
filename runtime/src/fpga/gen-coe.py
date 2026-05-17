#!/usr/bin/env python3
"""
将二进制文件 (.bin) 转换为 Xilinx COE 内存初始化文件 (.coe)。

默认数据宽度为 16 位，小端字节序。
用法示例:
    python bin2coe.py input.bin
    python bin2coe.py input.bin -o output.coe --width 32 --endian big
"""

import argparse
import os
import sys

def bin2coe(input_path, output_path, data_width=16, endian='little'):
    # 计算每个数据占用的字节数
    bytes_per_word = data_width // 8
    if data_width % 8 != 0 or bytes_per_word == 0:
        raise ValueError("数据宽度 (--width) 必须是 8 的倍数且大于 0")

    with open(input_path, 'rb') as f:
        raw = f.read()

    total_bytes = len(raw)
    if total_bytes == 0:
        raise ValueError("输入文件为空")

    # 检查是否能被完整切分
    remainder = total_bytes % bytes_per_word
    if remainder != 0:
        print(f"警告: 文件大小 ({total_bytes} 字节) 不是 {bytes_per_word} 字节的整数倍，"
              f"末尾 {remainder} 个字节将被忽略", file=sys.stderr)
        raw = raw[:total_bytes - remainder]

    # 将字节切分为数据字
    words = []
    for i in range(0, len(raw), bytes_per_word):
        chunk = raw[i:i + bytes_per_word]
        if endian == 'little':
            value = int.from_bytes(chunk, byteorder='little')
        else:
            value = int.from_bytes(chunk, byteorder='big')
        words.append(value)

    # 生成 COE 文件内容
    hex_digits = data_width // 4  # 每个数据需要的十六进制位数
    fmt = f"{{:0{hex_digits}X}}"  # 大写十六进制，固定宽度

    with open(output_path, 'w') as f:
        f.write(f"MEMORY_INITIALIZATION_RADIX=16;\n")
        f.write(f"MEMORY_INITIALIZATION_VECTOR=\n")
        if len(words) == 1:
            f.write(f"{fmt.format(words[0])};\n")
        else:
            for word in words[:-1]:
                f.write(f"{fmt.format(word)},\n")
            f.write(f"{fmt.format(words[-1])};\n")

    print(f"成功生成 COE 文件: {output_path} (共 {len(words)} 个数据字)")

def main():
    parser = argparse.ArgumentParser(description="将 .bin 转换为 .coe 内存初始化文件")
    parser.add_argument("input", help="输入的 .bin 文件路径")
    parser.add_argument("-o", "--output", help="输出的 .coe 文件路径 (默认与输入同名，扩展名为 .coe)")
    parser.add_argument("-w", "--width", type=int, default=32,
                        help="数据宽度（位），默认为 32")
    parser.add_argument("-e", "--endian", choices=['little', 'big'], default='little',
                        help="字节序，little 或 big，默认为 little")
    args = parser.parse_args()

    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = base + ".coe"

    try:
        bin2coe(args.input, args.output, args.width, args.endian)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()