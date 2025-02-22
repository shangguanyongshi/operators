#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "data_type.h"
#include <stdint.h>

/**
 * @brief 张量的描述符
 */
struct TensorDescriptor {
    // Datatype
    DT dt;
    // Number of dimensions
    uint64_t ndim;
    // Shape of the tensor, ndim elements
    uint64_t *shape;
    // Stride of each dimension in elements, ndim elements
    // 张量是一维存储的，对于三阶张量来说，(i, j, k) 索引定位到的张量元素位置可以通过
    // i * strides[0] + j * strides[1] + k * strides[2] 计算得到
    // 每一维的 strides 为相对于后面几维的偏移量（将后面所有维度的总元素个数看做一块整体，确定在第一个整体，见 learn_cxx exam _22）
    // 对于一个 3x4x5 的张量来说，strides = [20, 5, 1]
    int64_t *strides;
};

/**
 * @brief 张量描述符的指针
 */
typedef struct TensorDescriptor *infiniopTensorDescriptor_t;

#endif// __TENSOR_H__
