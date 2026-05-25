# INT8 MNIST C Inference - Progress

## 任务概述

将 `train.py` 训练的 TFLite INT8 量化模型导出为自定义二进制格式，并用纯 C 语言实现 INT8 推理（模仿 mnist-fc 的模式）。

## 设计思路

### 二进制格式

```
Header (512 bytes):
  Offset  Size   Field
  0       4      magic: uint32 = 0x4D4C4938 ("MLI8")
  4       4      version: int32 = 1
  8       4      input_dim: int32 (784)
  12      4      num_layers: int32 (2)
  16      64     layer_out_features[16]: int32
  80      4      input_scale: float32
  84      4      input_zero_point: int32
  88      4      output_scale: float32
  92      4      output_zero_point: int32
  96      416    reserved (padding to 512)
  512     ...    per-layer data

Per layer:
  float32   weight_scale[out_c]       (per-channel)
  int32     requant_multiplier[out_c] (TFLite Q0.31 定点乘数)
  int32     requant_shift[out_c]      (右移位数)
  int32     out_zero_point
  int32     bias[out_c]
  int8      weight[out_c * in_c]
```

### 模型架构

```
Input(28x28, INT8) → Flatten(784) → Dense(784→128) + ReLU → Dense(128→10)
```

### 关键算法

1. **INT8 矩阵乘**：`x_int32 = input_int8 - z_in` → INT64 点积 + bias（用 int64 防溢出）
2. **Requantize**：TFLite `MultiplyByQuantizedMultiplier` — Q0.31 定点乘法 + 移位 + zp + 夹紧
3. **ReLU**：对首层输出夹紧到 `[z_out, 127]`，末层输出夹紧到 `[-128, 127]`

## 文件清单

| 文件 | 说明 |
|------|------|
| `export.py` | 从 TFLite 模型导出参数到 `model.bin` |
| `model.h` | C 头文件，定义 Config / Layer / RunState 结构体 |
| `model.c` | C 实现，包含解析器 + 纯整型推理核心 |
| `main.c` | CLI 测试程序，支持预量化 INT8 图像和原始 uint8 图像 |
| `Makefile` | 集成到 runtime 构建系统 |
| `compare.py` | 对比 Python manual_inference 与 C 推理结果 |

## 验证结果

✅ **15/15 全部一致** — C 语言 INT8 推理结果与 Python 手动推理完全匹配。

```
============================================================
                          file  lbl  py  clang  match?
------------------------------------------------------------
     test_img_1605_label_3.bin    3    3     3  Y
     test_img_2261_label_1.bin    1    1     1  Y
     test_img_2794_label_0.bin    0    0     0  Y
     test_img_2954_label_0.bin    0    0     0  Y
     test_img_4449_label_6.bin    6    6     6  Y
     test_img_4732_label_1.bin    1    1     1  Y
     test_img_5366_label_2.bin    2    2     2  Y
      test_img_551_label_7.bin    7    7     7  Y
     test_img_5602_label_9.bin    9    9     9  Y
     test_img_6170_label_7.bin    7    7     7  Y
      test_img_632_label_2.bin    2    2     2  Y
     test_img_6712_label_1.bin    1    1     1  Y
     test_img_6964_label_5.bin    5    5     5  Y
     test_img_7589_label_4.bin    4    4     4  Y
     test_img_9257_label_7.bin    7    7     7  Y
------------------------------------------------------------
Agreement: 15/15
```

## 使用方法

```bash
# 1. 训练模型并导出 TFLite
python train.py

# 2. 导出二进制模型
python export.py

# 3. 编译 C 推理程序
make

# 4. 推理（预量化 INT8 图像）
./build/native/mnist-int8 out/model.bin out/test_img_551_label_7.bin

# 5. 推理（原始 uint8 图像，自动量化）
./build/native/mnist-int8 -r out/model.bin image.raw
```
