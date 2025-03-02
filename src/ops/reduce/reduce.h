#ifndef REDUCE_H
#define REDUCE_H

#include "export.h"
#include "operators.h"

typedef struct ReduceDescriptor {
    Device device;
} ReduceDescriptor;

typedef ReduceDescriptor *infiniopReduceDescriptor_t;

/**
 * @brief 根据传入的参数创建执行 Reduce 操作的描述符
 * @param handle infiniop 句柄
 * @param desc_ptr Reduce 描述符指针
 * @param reduced 结果张量描述符
 * @param data 输入张量描述符
 * @param axes 要进行 Reduce 的维度
 * @param keepdims 是否保留 Reduce 后的维度
 * @param noop_with_empty_axes 在 axes 为空时是否还执行 Reduce 操作
 * @param reduce_type 1: ReduceMax, 2: ReduceMin, 3: ReduceMean
 * @return 描述符是否创建成功
 */
__C infiniopStatus_t infiniopCreateReduceDescriptor(infiniopHandle_t handle,
                                                    infiniopReduceDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t reduced,
                                                    infiniopTensorDescriptor_t data,
                                                    infiniopTensorDescriptor_t axes,
                                                    int keepdims,
                                                    int noop_with_empty_axes,
                                                    int reduce_type);

/**
 * @brief 进行 F16 类型的 ReduceMean 操作时，将数据频繁的从 F32 转到 F16 会丢失精度，需要额外的空间，
 *        将和表示为 F32 类型，最后再将结果转换为 F16 类型
 * @param desc infiniopReduceDescriptor 句柄
 * @param size 保存所分配空间的大小
 * @return 空间是否分配成功
 */
__C infiniopStatus_t infiniopGetReduceWorkspaceSize(infiniopReduceDescriptor_t desc, uint64_t *size);

__C infiniopStatus_t infiniopReduce(infiniopReduceDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *reduced,
                                    void const *data,
                                    void const *axes,
                                    void *stream);

__C infiniopStatus_t infiniopDestroyReduceDescriptor(infiniopReduceDescriptor_t desc);

#endif