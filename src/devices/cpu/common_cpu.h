#ifndef __COMMON_CPU_H__
#define __COMMON_CPU_H__

#include <cmath>
#include <cstdint>
#include <cstring>

// return a mask with the specified number of low bits set to 1
constexpr static uint16_t mask_low(int bits) noexcept {
    return (1 << bits) - 1;
}

// convert half-precision float to single-precision float
/**
 * @brief 将 16 位浮点数转换为 32 位浮点数
 * @param code 在 uint16_t 类型中保存的 16 位浮点数
 * @return 转换后的 32 位浮点数保存为 float 类型
 */
float f16_to_f32(uint16_t code);

// convert single-precision float to half-precision float
/**
 * @brief 将 32 位浮点数转换为 16 位浮点数
 * @param val 32 位浮点数以 float 类型保存
 * @return 转换后的 16 位浮点数保存在 uint16_t 类型中
 */
uint16_t f32_to_f16(float val);

// get the corresponding offset in the destination given the flat index of the source (for element mapping in shape broadcast)
uint64_t getDstOffset(uint64_t flat_index, uint64_t ndim, int64_t const *src_strides, int64_t const *dst_strides);

// get the memory offset of the given element in a tensor given its flat index
uint64_t getOffset(uint64_t flat_index, uint64_t ndim, uint64_t const *shape, int64_t const *strides);

/**
 * get the total array size (element count) after applying padding for a 
 * ndim-ary tensor with the given shape
 */
uint64_t getPaddedSize(uint64_t ndim, uint64_t *shape, uint64_t const *pads);

// calculate the padded shape and store the result in padded_shape
void getPaddedShape(uint64_t ndim, uint64_t const *shape, uint64_t const *pads, uint64_t *padded_shape);

#endif// __COMMON_CPU_H__
