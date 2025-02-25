#ifndef __UTILS_H__
#define __UTILS_H__

#include "data_type.h"
#include "tensor.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/* This file contains some useful macros and helper functions */

// check if an expression is true, and if not, print an error message and abort the program
inline void assert_true(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(expr) assert_true(expr, #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != " #b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a) != nullptr, #a " is nullptr", __FILE__, __LINE__)

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(EXIT_FAILURE)

#define ROUND_UP_DIV(x, y) ((x + y - 1) / y)

#define CHECK_ERROR(call, target, errCode)                   \
    do {                                                     \
        if (auto value = (call); value == (target)) {        \
            std::cerr << "Error: expected " << (target)      \
                      << " but got " << value                \
                      << " in file " << __FILE__             \
                      << ", function " << __func__           \
                      << ", line " << __LINE__ << std::endl; \
            return (errCode);                                \
        }                                                    \
    } while (0)

#define CREATE_CHECK_ERROR(expr, value, target, errCode) \
    expr;                                                \
    CHECK_ERROR(value, target, errCode)

#define CHECK_STATUS(call, target)                           \
    do {                                                     \
        if (auto value = (call); value != (target)) {        \
            std::cerr << "Error: expected " << (target)      \
                      << " but got " << value                \
                      << " in file " << __FILE__             \
                      << ", function " << __func__           \
                      << ", line " << __LINE__ << std::endl; \
            return value;                                    \
        }                                                    \
    } while (0)

// check if two data layouts (types) are equal
/**
 * @brief 比较两个 DataLayout 是否相等
 * @param a 第一个操作数
 * @param b 第二个操作数
 * @return 如果相等，返回 true，否则返回 false
 */
inline bool dtype_eq(DataLayout a, DataLayout b) {
    union TypePun {
        DataLayout layout;
        int i;
    } pun;
    pun.layout = a;
    auto a_ = pun.i;
    pun.layout = b;
    auto b_ = pun.i;
    return a_ == b_;
}

/**
 * @brief 返回每个维度偏移量所占字节大小
 * @param desc 张量的描述符
 * @return 每个维度偏移量所占字节大小
 */
inline std::vector<int64_t> get_byte_strides(infiniopTensorDescriptor_t desc) {
    // 获取单个元素的字节数
    int64_t dsize = desc->dt.size;
    // 每个维度偏移量乘以每个元素所占字节数，就是该维度的偏移量所占字节数
    std::vector<int64_t> strides(desc->ndim);
    for (uint64_t i = 0; i < desc->ndim; i++) {
        strides[i] = dsize * desc->strides[i];
    }

    return strides;
}

// calculate the broadcasted shape for two tensors
/**
 * @brief 计算两个张量的广播后的形状
 * @param shape1 第一个张量的实际形状
 * @param ndim1 第一个张量的实际维度数
 * @param shape2 第二个张量的实际形状
 * @param ndim2 第二个张量的实际维度数
 * @param broadcast_shape 广播后的形状保存到该数组中
 * @param padded_shape1 在第一个张量前面添加多个维度 1 达到 max_rank 阶后的形状
 * @param padded_shape2 在第二个张量前面添加多个维度 1 达到 max_rank 阶后的形状
 * @param max_rank 广播后的最大维度个数
 * @return 如果广播成功，返回 true，否则返回 false
 */
inline bool getBroadcastShape(const uint64_t *shape1, uint64_t ndim1,
                              const uint64_t *shape2, uint64_t ndim2,
                              uint64_t *broadcast_shape, uint64_t *padded_shape1,
                              uint64_t *padded_shape2, uint64_t max_rank) {
    // prepending and initializing
    std::fill(padded_shape1, padded_shape1 + max_rank, 1);
    std::fill(padded_shape2, padded_shape2 + max_rank, 1);
    std::copy(shape1, shape1 + ndim1, padded_shape1 + max_rank - ndim1);
    std::copy(shape2, shape2 + ndim2, padded_shape2 + max_rank - ndim2);

    // compute broadcasted shape
    for (size_t i = 0; i < max_rank; ++i) {
        if (padded_shape1[i] == padded_shape2[i] || padded_shape1[i] == 1 || padded_shape2[i] == 1) {
            broadcast_shape[i] = std::max(padded_shape1[i], padded_shape2[i]);
        } else {
            return false;
        }
    }

    return true;
}

// check if the shape of tensor c is valid after broadcasting tensors a and b and also get the broadcasted shapes
// 检查张量 c 的形状是否有效，即是否可以通过广播张量 a 和 b 得到，且广播后的阶为 broadcast_ndim
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a, infiniopTensorDescriptor_t b, infiniopTensorDescriptor_t c,
                                  uint64_t broadcast_ndim) {
    std::vector<uint64_t>
        broadcast_shape_(broadcast_ndim),
        padded_shape1_(broadcast_ndim),
        padded_shape2_(broadcast_ndim);
    auto broadcast_shape = broadcast_shape_.data(),
         padded_shape1 = padded_shape1_.data(),
         padded_shape2 = padded_shape2_.data();
    if (broadcast_ndim != c->ndim || !getBroadcastShape(a->shape, a->ndim, b->shape, b->ndim, broadcast_shape, padded_shape1, padded_shape2, broadcast_ndim)) {
        return false;
    }
    return std::equal(broadcast_shape, broadcast_shape + broadcast_ndim, c->shape);
}

// check if the shape of tensor src can be validly broadcasted to that of the tensor dst
// 检查张量 src 的形状是否可以广播到张量 dst 的形状
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t dst, infiniopTensorDescriptor_t src) {
    if (dst->ndim < src->ndim) {
        return false;
    }
    std::vector<uint64_t> padded_shape_(dst->ndim);
    auto padded_shape = padded_shape_.data();
    std::fill(padded_shape, padded_shape + dst->ndim, 1);
    std::copy(src->shape, src->shape + src->ndim, padded_shape + dst->ndim - src->ndim);
    for (size_t i = 0; i < dst->ndim; ++i) {
        if (padded_shape[i] != dst->shape[i] && padded_shape[i] != 1) {
            return false;
        }
    }
    return true;
}

// check if the shape of tensor c is valid after broadcasting tensors a and b
// 检查张量 c 的形状是否有效，即是否可以通过广播张量 a 和 b 得到
inline bool isValidBroadcastShape(infiniopTensorDescriptor_t a, infiniopTensorDescriptor_t b, infiniopTensorDescriptor_t c) {
    return isValidBroadcastShape(a, b, c, std::max(a->ndim, b->ndim));
}

/**
 * @brief 返回给定格式的张量所占总字节数
 * @param desc 张量的描述符
 * @return 该张量所占的总字节数
 */
inline uint64_t get_byte_size(infiniopTensorDescriptor_t desc) {
    // 获取张量中一个元素所占的字节数
    uint64_t dsize = desc->dt.size;
    // 计算张量所占字节的总个数
    uint64_t size = 1;
    for (uint64_t i = 0; i < desc->ndim; i++) {
        size *= desc->shape[i];
    }
    return size * dsize;
}

// permute the dimensions of a tensor descriptor
inline infiniopTensorDescriptor_t permute(infiniopTensorDescriptor_t desc, const std::vector<uint64_t> &order) {
    uint64_t ndim = desc->ndim;
    if (order.size() != ndim) {
        return nullptr;
    }
    uint64_t *shape = new uint64_t[ndim];
    int64_t *strides = new int64_t[ndim];
    for (uint64_t i = 0; i < ndim; i++) {
        if (std::find(order.begin(), order.end(), i) == order.end()) {
            return nullptr;
        }
        shape[i] = desc->shape[order[i]];
        strides[i] = desc->strides[order[i]];
    }
    return new TensorDescriptor{
        desc->dt, ndim, shape, strides};
}

// check if the dimensions [dim_start, dim_end] of a tensor descriptor are contiguous
// 检查张量的维度是否连续（判断张量的 strides 是否合法）
inline bool is_contiguous(const infiniopTensorDescriptor_t &desc, uint64_t dim_start, uint64_t dim_end) {
    for (size_t i = dim_start + 1; i <= dim_end; i++) {
        if (desc->strides[i - 1] != static_cast<int64_t>(desc->shape[i]) * desc->strides[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 检查张量的 strides 是否合法
 * @param desc 待检查的张量描述符
 * @return 维度合法时返回 true，否则返回 false
 */
inline bool is_contiguous(const infiniopTensorDescriptor_t &desc) {
    if (desc->ndim == 0) {
        return true;
    }
    return is_contiguous(desc, 0, desc->ndim - 1);
}

// merge the dimensions [dim_start, dim_end] of a tensor descriptor
// 将张量的维度 [dim_start, dim_end] 合并为一个维度
inline infiniopTensorDescriptor_t dim_merge(infiniopTensorDescriptor_t desc, uint64_t dim_start, uint64_t dim_end) {
    uint64_t ndim = desc->ndim;
    if (dim_start > dim_end || dim_end >= ndim) {
        return nullptr;
    }

    uint64_t new_ndim = ndim - (dim_end - dim_start);
    uint64_t *new_shape = new uint64_t[new_ndim];
    int64_t *new_strides = new int64_t[new_ndim];
    uint64_t index = 0;
    for (size_t i = 0; i < dim_start; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    if (!is_contiguous(desc, dim_start, dim_end)) {
        return nullptr;
    }
    new_shape[index] = 1;
    for (size_t i = dim_start; i <= dim_end; i++) {
        new_shape[index] *= desc->shape[i];
    }
    new_strides[index] = desc->strides[dim_end];
    index++;
    for (size_t i = dim_end + 1; i < ndim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    return new TensorDescriptor{
        desc->dt, new_ndim, new_shape, new_strides};
}

// split the dimension dim of a tensor descriptor into multiple dimensions
inline infiniopTensorDescriptor_t dim_split(infiniopTensorDescriptor_t desc, uint64_t dim, const std::vector<uint64_t> &dims) {
    uint64_t ndim = desc->ndim;
    if (desc->shape[dim] != std::accumulate(dims.begin(), dims.end(), (uint64_t)1, std::multiplies<uint64_t>{})) {
        return nullptr;
    }
    uint64_t new_ndim = ndim + dims.size() - 1;
    uint64_t *new_shape = new uint64_t[new_ndim];
    int64_t *new_strides = new int64_t[new_ndim];
    uint64_t index = 0;
    for (size_t i = 0; i < dim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    for (size_t i = 0; i < dims.size(); i++) {
        new_shape[index] = dims[i];
        new_strides[index] = desc->strides[dim] * desc->shape[dim] / std::accumulate(dims.begin(), dims.begin() + i + 1, 1, std::multiplies<uint64_t>());
        index++;
    }
    for (size_t i = dim + 1; i < ndim; i++) {
        new_shape[index] = desc->shape[i];
        new_strides[index] = desc->strides[i];
        index++;
    }
    return new TensorDescriptor{
        desc->dt, new_ndim, new_shape, new_strides};
}

#endif// __UTILS_H__
