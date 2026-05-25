import os
import glob
import numpy as np
import tensorflow as tf

model_path = 'out/mnist_full_int8.tflite'
bin_dir = 'out'

# =============================================================================
# 1. 深入解析 TFLite 计算图，精准定位层与张量
# =============================================================================
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()
ops = interpreter._get_ops_details() # 获取真实执行图的算子列表

def get_quant_params(tensor_idx):
    """根据张量索引提取量化参数，同时处理 per-tensor 与 per-channel 量化"""
    detail = details[tensor_idx]
    # Per-channel 量化参数存储在 quantization_parameters 中
    qp = detail.get('quantization_parameters', None)
    if qp is not None and qp['scales'].size > 0:
        scales = qp['scales'].copy()
        zps = qp['zero_points'].copy()
        return np.array(scales, dtype=np.float32), np.array(zps, dtype=np.int32)
    # Per-tensor 量化回退到 quantization 字段
    scale, zp = detail['quantization']
    return np.array(scale, dtype=np.float32), np.array(zp, dtype=np.int32)

# 动态找到网络中的两个 FULLY_CONNECTED 算子
fc_ops = [op for op in ops if op['op_name'] == 'FULLY_CONNECTED']
if len(fc_ops) != 2:
    raise ValueError(f"期望找到 2 个 FULLY_CONNECTED 算子，但找到 {len(fc_ops)} 个！")

# --- 提取 Layer 1 对应的真实张量 ---
# Fully Connected 的输入列表通常是: [input, weights, bias]
l1_in_idx, l1_w_idx, l1_b_idx = fc_ops[0]['inputs']
l1_out_idx = fc_ops[0]['outputs'][0]

in_scale, in_zp = get_quant_params(l1_in_idx)
w1 = interpreter.get_tensor(l1_w_idx)
b1 = interpreter.get_tensor(l1_b_idx)
w1_scale, w1_zp = get_quant_params(l1_w_idx)
out1_scale, out1_zp = get_quant_params(l1_out_idx)

# --- 提取 Layer 2 对应的真实张量 ---
l2_in_idx, l2_w_idx, l2_b_idx = fc_ops[1]['inputs']
l2_out_idx = fc_ops[1]['outputs'][0]

_ , _ = get_quant_params(l2_in_idx) # 它必定与 out1_scale, out1_zp 一致
w2 = interpreter.get_tensor(l2_w_idx)
b2 = interpreter.get_tensor(l2_b_idx)
w2_scale, w2_zp = get_quant_params(l2_w_idx)
out2_scale, out2_zp = get_quant_params(l2_out_idx)

print("✅ 基于计算图，权重和量化参数精准提取成功！\n")

# =============================================================================
# 2. TFLite 原生纯整型 requantize 原语（与 tensorflow/lite/kernels/internal 一致）
# =============================================================================

def _quantize_multiplier(double_multiplier):
    """TFLite QuantizeMultiplier: 将浮点乘数分解为 Q0.31 定点乘数 + 右移位数。
       （只在模型加载时执行一次，推理时不再涉及浮点）"""
    import math
    if double_multiplier == 0.0:
        return np.int32(0), np.int32(0)
    # frexp 返回 m ∈ [0.5, 1.0), 原始值 = m * 2^shift
    m, shift = math.frexp(double_multiplier)
    # q_fixed 是 Q0.31 定点值, 范围 [2^30, 2^31]
    q_fixed = int(round(m * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        shift = 0
        q_fixed = 0
    return np.int32(q_fixed), np.int32(shift)


def _saturating_rounding_doubling_high_mul(a, b):
    """int32 × int32 → 取高 32 位（带舍入），饱和处理 INT32_MIN 边界情况。
       等价于 (a * b + 2^30) >> 31 的舍入高半段乘法。"""
    a64 = np.int64(a)
    b64 = np.int64(b)
    ab64 = a64 * b64
    nudge = np.where(ab64 >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    result = ((ab64 + nudge) >> 31).astype(np.int32)
    i32min = np.int32(-2147483648)
    i32max = np.int32(2147483647)
    return np.where((a == i32min) & (b == i32min), i32max, result)


def _rounding_divide_by_pot(x, exponent):
    """x / 2^exponent 带舍入（舍入到最近，半值时向偶数舍入）。"""
    if exponent == 0:
        return x
    mask = (np.int32(1) << np.int32(exponent)) - np.int32(1)
    remainder = x & mask
    threshold = (mask >> np.int32(1)) + np.where(x < np.int32(0), np.int32(1), np.int32(0))
    return (x >> np.int32(exponent)) + np.where(remainder > threshold, np.int32(1), np.int32(0))


def _multiply_by_quantized_multiplier(x, quantized_multiplier, shift):
    """TFLite MultiplyByQuantizedMultiplier: 纯整型 requantize。"""
    left_shift  = np.where(shift > 0, shift, np.int32(0))
    right_shift = np.where(shift > 0, np.int32(0), -shift)
    x_s = np.left_shift(x, left_shift)
    high = _saturating_rounding_doubling_high_mul(x_s, quantized_multiplier)
    return _rounding_divide_by_pot(high, right_shift)


# --- 预计算各层的整数 multiplier / shift（一次性，推理时不涉及浮点） ---
def _compute_int_multipliers(s_in, s_w, s_out):
    """为 per-channel 权重预计算 (int32 q_mult, int32 shift) 对。"""
    s_in_val = float(np.squeeze(s_in))
    s_out_val = float(np.squeeze(s_out))
    eff_scale = (s_in_val * s_w.astype(np.float64)) / s_out_val
    q_mults = np.zeros(s_w.size, dtype=np.int32)
    shifts = np.zeros(s_w.size, dtype=np.int32)
    for i in range(s_w.size):
        q_mults[i], shifts[i] = _quantize_multiplier(eff_scale[i])
    return q_mults, shifts

# Layer 1
l1_qmult, l1_shift = _compute_int_multipliers(in_scale, w1_scale, out1_scale)
# Layer 2
l2_qmult, l2_shift = _compute_int_multipliers(out1_scale, w2_scale, out2_scale)

print("🔢 纯整型 requantize 参数预计算完成\n")

# =============================================================================
# 3. 纯整型 Dense 算子 — 矩阵乘（INT8×INT8→INT32） + TFLite 定点 requantize
# =============================================================================
def manual_dense_int8_pure_integer(input_int8, weight_int8, bias_int32,
                                   z_in, z_w, z_out,
                                   q_mults, shifts):
    """所有算术均在 INT8 / INT32 / INT64 域完成，不含浮点。"""
    # 1. 输入去零点
    x_int32 = input_int8.flatten().astype(np.int32) - int(np.squeeze(z_in))
    x_int32 = x_int32.reshape(1, -1)

    # 2. 权重去零点（per-channel INT8 对称量化 → zp=0，本行实际无变化）
    if z_w.ndim == 0 or z_w.size == 1:
        w_int32 = weight_int8.astype(np.int32) - int(np.squeeze(z_w))
    else:
        w_int32 = weight_int8.astype(np.int32) - z_w.reshape(-1, 1)

    # 3. INT8×INT8 → INT32 矩阵乘 + INT32 偏置
    acc_int32 = np.dot(x_int32, w_int32.T) + bias_int32  # shape (1, out_c)

    # 4. 纯整型 requantize（逐通道），保持 (1, out_c) 形状
    out_vals = np.zeros((1, acc_int32.shape[1]), dtype=np.int32)
    for c in range(acc_int32.shape[1]):
        out_vals[0, c] = _multiply_by_quantized_multiplier(
            np.int32(acc_int32[0, c]), q_mults[c], shifts[c]
        )
    out = out_vals + int(np.squeeze(z_out))

    # 5. INT8 饱和（隐含 ReLU 融合）
    return np.clip(out, -128, 127).astype(np.int8)

# =============================================================================
# 4. 批量推理测试 — 对比浮点近似与纯整型两种实现
# =============================================================================
bin_files = glob.glob(os.path.join(bin_dir, "*.bin"))
if not bin_files:
    raise FileNotFoundError("未在 out 目录下找到 .bin 测试文件！")

print(f"{'='*60}")
print(f"{'图片':>30s}  真实  浮点预测  整型预测  一致?")
print(f"{'-'*60}")

all_correct = 0
all_match = 0
total = 0

for test_file in sorted(bin_files):
    true_label = int(os.path.basename(test_file).split("_label_")[1].split(".")[0])
    total += 1

    with open(test_file, 'rb') as f:
        img_data = f.read()
    img_int8 = np.frombuffer(img_data, dtype=np.int8).reshape(1, 784)

    # ── 纯整型推理 ──
    l1_out = manual_dense_int8_pure_integer(
        input_int8=img_int8, weight_int8=w1, bias_int32=b1,
        z_in=in_zp, z_w=w1_zp, z_out=out1_zp,
        q_mults=l1_qmult, shifts=l1_shift,
    )
    l2_out = manual_dense_int8_pure_integer(
        input_int8=l1_out, weight_int8=w2, bias_int32=b2,
        z_in=out1_zp, z_w=w2_zp, z_out=out2_zp,
        q_mults=l2_qmult, shifts=l2_shift,
    )
    pred_int = np.argmax(l2_out[0])

    # ── 浮点近似推理（用于对比） ──
    def _manual_dense_float(x, w, b, s_in, z_in, s_w, s_out, z_out):
        xv = x.flatten().astype(np.int32) - int(np.squeeze(z_in))
        acc = np.dot(xv.reshape(1,-1), w.astype(np.int32).T) + b
        eff = (float(np.squeeze(s_in)) * s_w.astype(np.float64)) / float(np.squeeze(s_out))
        out = np.round(acc.astype(np.float64) * eff) + int(np.squeeze(z_out))
        return np.clip(out.astype(np.int32), -128, 127).astype(np.int8)

    l1_float = _manual_dense_float(img_int8, w1, b1, in_scale, in_zp, w1_scale, out1_scale, out1_zp)
    l2_float = _manual_dense_float(l1_float, w2, b2, out1_scale, out1_zp, w2_scale, out2_scale, out2_zp)
    pred_float = np.argmax(l2_float[0])

    match = "✅" if np.array_equal(l2_out, l2_float) else "❌"
    name = os.path.basename(test_file)
    
    if pred_int == true_label:
        all_correct += 1
    if np.array_equal(l2_out, l2_float):
        all_match += 1

    print(f"{name:>30s}  {true_label:>3d}  {pred_float:>8d}  {pred_int:>8d}  {match}")

print(f"{'-'*60}")
print(f"准确率: {all_correct}/{total}")
print(f"浮点/整型一致: {all_match}/{total}")