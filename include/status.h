/**
 * @file status.h
 * @brief 定义状态码相关的枚举值
 * @author 上官永石
 * @date 2025-02-21
 */
#ifndef INFINIOP_STATUS_H
#define INFINIOP_STATUS_H

/**
 * @brief 操作的状态
 */
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_EXECUTION_FAILED = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_BAD_TENSOR_DTYPE = 3,
    STATUS_BAD_TENSOR_SHAPE = 4,
    STATUS_BAD_TENSOR_STRIDES = 5,
    STATUS_MEMORY_NOT_ALLOCATED = 6,
    STATUS_INSUFFICIENT_WORKSPACE = 7,
    STATUS_BAD_DEVICE = 8,
} infiniopStatus_t;

#endif
