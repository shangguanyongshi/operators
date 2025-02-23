#include "common_cpu.h"

/**
 * @brief 将 F16 格式的浮点数转换为 F32 格式的浮点数，后续针对 F16 的数据计算时，
 *        都是先转换为 F32 计算后再转换回去
 * @param h 要转换的数字
 * @return F32 类型的转换后的格式
 */
float f16_to_f32(uint16_t h) {
    // 将保存在 uint16_t 中的浮点数通过转换保存到 uint32_t 中
    // uint16_t 共 16 位，第 1 位是符号位，接下来 5 位是指数，剩下的 10 位是尾数
    uint32_t sign = (h & 0x8000) << 16; // Extract the sign bit
    int32_t exponent = (h >> 10) & 0x1F;// Extract the exponent
    uint32_t mantissa = h & 0x3FF;      // Extract the mantissa (fraction part)

    // IEEE 754 标准中，第 1 位为符号位，之后 8 位是阶码，尾数占最后 23 位
    // 阶码全 1，尾数全 0 的 float 表示无穷大、尾数非 0 的 float 为 NaN
    // 阶码全 0，尾数全 0 的 float 表示机器 0、尾数非 0 的 float 表示非规格化数
    if (exponent == 31) {// Special case for Inf and NaN
        // 如果指数位为31（即全 1），表示可能是无穷大或 NaN
        if (mantissa != 0) {
            // NaN: Set float32 NaN. 尾数非 0，表示 NaN
            // 将 16 位浮点数的后 10 位在 32 位浮点数的后 23 位中从前向后保存
            uint32_t f32 = sign | 0x7F800000 | (mantissa << 13);
            return *(float *) &f32;
        } else {
            // Infinity，如果尾数为 0，为无穷大
            // 与符号位按位或，是为了确定是正无穷还是负无穷
            uint32_t f32 = sign | 0x7F800000;
            return *(float *) &f32;
        }
    } else if (exponent == 0) {// Subnormal float16 or zero
        if (mantissa == 0) {
            // Zero (positive or negative)
            uint32_t f32 = sign;// Just return signed zero
            return *(float *) &f32;
        } else {
            // Subnormal: Convert to normalized float32
            exponent = -14;                  // Set exponent for subnormal numbers
            while ((mantissa & 0x400) == 0) {// Normalize mantissa
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;// Clear the leading 1 bit
            // 阶码为移码，在 float16 中阶码为非移码，因此需要做转换
            uint32_t f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
            return *(float *) &f32;
        }
    } else {
        // Normalized float16，-15 是为了删除后三位的值
        uint32_t f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        return *(float *) &f32;
    }
}

uint16_t f32_to_f16(float val) {
    uint32_t f32 = *(uint32_t *) &val;            // Read the bits of the float32
    uint16_t sign = (f32 >> 16) & 0x8000;         // Extract the sign bit
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;// Extract and de-bias the exponent
    uint32_t mantissa = f32 & 0x7FFFFF;           // Extract the mantissa (fraction part)

    if (exponent >= 31) {// Special cases for Inf and NaN
        // NaN
        if (exponent == 128 && mantissa != 0) {
            return sign | 0x7E00;
        }
        // Infinity
        return sign | 0x7C00;
    } else if (exponent >= -14) {// Normalized case
        return sign | ((exponent + 15) << 10) | (mantissa >> 13);
    } else if (exponent >= -24) {
        mantissa |= 0x800000;// Add implicit leading 1
        mantissa >>= (-14 - exponent);
        return sign | (mantissa >> 13);
    } else {
        // Too small for subnormal: return signed zero
        return sign;
    }
}

// 根据给定的一维索引 flat_index、维度数量 ndim、源步长数组 src_strides 和目标步长数组 dst_strides 来计算目标偏移量
uint64_t getDstOffset(uint64_t flat_index, uint64_t ndim, int64_t const *src_strides, int64_t const *dst_strides) {
    uint64_t res = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        res += flat_index / src_strides[i] * dst_strides[i];
        flat_index %= src_strides[i];
    }
    return res;
}

// 根据给定的一维扁平索引 flat_index、维度数量 ndim、各维度的形状数组 shape 和步长数组 strides 来计算多维数组中对应元素在内存中的偏移量
uint64_t getOffset(uint64_t flat_index, uint64_t ndim, uint64_t const *shape, int64_t const *strides) {
    uint64_t res = 0;
    for (long i = ndim - 1; i >= 0; --i) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

// 根据给定的维度数量 ndim、各维度的形状数组 shape 和步长数组 strides 来计算多维数组的总大小
uint64_t getPaddedSize(uint64_t ndim, uint64_t *shape, uint64_t const *pads) {
    uint64_t total_size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_size *= shape[i] + (i < 2 ? 0 : 2 * pads[i - 2]);
    }
    return total_size;
}

// 根据给定的维度数量 ndim、各维度的形状数组 shape 和步长数组 strides 来计算出填充后的形状
void getPaddedShape(uint64_t ndim, uint64_t const *shape, uint64_t const *pads, uint64_t *padded_shape) {
    memcpy(padded_shape, shape, ndim * sizeof(uint64_t));
    for (size_t i = 2; i < ndim; ++i) {
        padded_shape[i] += 2 * pads[i - 2];
    }
}
