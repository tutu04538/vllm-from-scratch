"""测试专用 W4 位运算参考，不进入生产推理路径。"""
import torch

AWQ_REVERSE = (0, 4, 1, 5, 2, 6, 3, 7)


def unpack_standard_word(value):
    word = int(value) & 0xffffffff
    return [(word >> (4 * i)) & 15 for i in range(8)]


def pack_standard_word(values):
    if len(values) != 8 or any(type(v) is not int or not 0 <= v <= 15 for v in values):
        raise ValueError('每字必须有八个 0..15 的整数')
    word = sum(v << (4 * i) for i, v in enumerate(values))
    return word if word < 2**31 else word - 2**32


def unpack_awq_cols(packed):
    if packed.ndim != 2 or packed.dtype != torch.int32 or packed.device.type != 'cpu':
        raise ValueError('需要二维 CPU int32 AWQ packed 张量')
    rows = []
    for row in packed.tolist():
        result = []
        for word in row:
            physical = unpack_standard_word(word)
            result.extend(physical[i] for i in AWQ_REVERSE)
        rows.append(result)
    return torch.tensor(rows, dtype=torch.int32)


def unpack_standard_rows(packed):
    if packed.ndim != 2 or packed.dtype != torch.int32 or packed.device.type != 'cpu':
        raise ValueError('需要二维 CPU int32 标准 packed 张量')
    k_words, n = packed.shape
    out = torch.empty((k_words * 8, n), dtype=torch.int32)
    for i in range(k_words):
        for j in range(n):
            out[i*8:i*8+8, j] = torch.tensor(unpack_standard_word(packed[i, j]))
    return out


def pack_standard_rows(logical):
    if logical.ndim != 2 or logical.shape[0] % 8:
        raise ValueError('标准行打包要求 K 被 8 整除')
    k, n = logical.shape
    result = torch.empty((k // 8, n), dtype=torch.int32)
    for i in range(k // 8):
        for j in range(n):
            result[i, j] = pack_standard_word(logical[i*8:i*8+8, j].tolist())
    return result


def pack_standard_cols(logical):
    if logical.ndim != 2 or logical.shape[1] % 8:
        raise ValueError('标准列打包要求 N 被 8 整除')
    k, n = logical.shape
    result = torch.empty((k, n // 8), dtype=torch.int32)
    for i in range(k):
        for j in range(n // 8):
            result[i, j] = pack_standard_word(logical[i, j*8:j*8+8].tolist())
    return result


def unpack_standard_cols(packed):
    if packed.ndim != 2 or packed.dtype != torch.int32 or packed.device.type != 'cpu':
        raise ValueError('需要二维 CPU int32 标准 packed 张量')
    return torch.tensor([sum((unpack_standard_word(word) for word in row), [])
                         for row in packed.tolist()], dtype=torch.int32)


def awq_to_standard(qweight, qzeros):
    q = unpack_awq_cols(qweight)
    z = unpack_awq_cols(qzeros)
    if q.shape[1] != z.shape[1]:
        raise ValueError('qweight/qzeros 的 N 不匹配')
    return pack_standard_rows(q), pack_standard_rows(z.T.contiguous()), q, z


def dequantize(q, scale, zero, *, group_size, valid_k=None, valid_n=None):
    """scale 为反量化乘数；只返回明确声明的有效区域。"""
    if q.ndim != 2 or scale.ndim != 2 or zero.shape != scale.shape or group_size <= 0:
        raise ValueError('错误的 W4 group 参数')
    k, n = q.shape
    if k % group_size or scale.shape != (k // group_size, n):
        raise ValueError('scale/zero shape 与物理 K、N、group 不匹配')
    if bool(((q < 0) | (q > 15)).any()) or bool(((zero < 0) | (zero > 15)).any()):
        raise ValueError('q 和逻辑 zero 必须在 0..15')
    vk = k if valid_k is None else valid_k
    vn = n if valid_n is None else valid_n
    if type(vk) is not int or type(vn) is not int or not 0 <= vk <= k or not 0 <= vn <= n:
        raise ValueError('无效的有效区域')
    result = torch.empty((vk, vn), dtype=torch.float32)
    for row in range(vk):
        for col in range(vn):
            group = row // group_size
            result[row, col] = (int(q[row, col]) - int(zero[group, col])) * float(scale[group, col])
    return result
